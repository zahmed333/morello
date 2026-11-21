"""Fit and check a monitor against a table in a warehouse.

A scheduled job calls check_table. Each run checks the batches that arrived since the last
run, appends one row per check to a results table, and saves the monitor's history.

What happens when a run fails part-way:

- Nothing is saved until the results are written. A run that dies starts again from the same
  batch, so no batch is skipped.
- The results table is an append-only log. A retried run can append a batch a second time.
  Rows are identified by (table_name, batch, segment); read the one with the latest checked_at.
- Learning a batch twice has no effect: the history keeps one entry per batch.
- An alert outlives the run that raised it. The batch stays in unresolved_alerts until a person
  accepts it or it is checked again and passes, so a task retried after an alert fails again.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import state
from .distributed import distributed_statistics
from .monitor import MIN_FIT_BATCHES, Monitor, TableConfig
from .report import Report
from .sql import Source, TableSampler, connect, literal
from .store import Store, open_store

log = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "checked_at",
    "table_name",
    "batch",
    "segment",
    "status",
    "rows_checked",
    "n_alerts",
    "suspect_columns",
    "top_check",
    "top_column",
    "top_ratio",
    "details",
]
MIN_SLICE_CHECK_ROWS = 50  # a slice with fewer sampled rows than this is not checked
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ACCEPTED = "accepted"  # status written to the results table when a person accepts a batch

SliceCounts = dict[str, dict[str, int]]  # batch -> slice value -> rows


def default_key(table: str) -> str:
    """The name a table's monitor is saved under when none is given."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(table)).strip("_").lower()


def result_rows(table: str, reports: list[Report]) -> pd.DataFrame:
    """One row per report, in the layout of the results table."""
    checked_at = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    rows = []
    for report in reports:
        top = report.alerts[0] if report.alerts else None
        details = {
            "findings": [finding.summary for finding in report.findings[:10]],
            "notes": [_finding_note(finding) for finding in report.findings[:10]],
            "alerts": report.to_dict()["alerts"][:10],
            "examples": report.examples,
        }
        rows.append(
            {
                "checked_at": checked_at,
                "table_name": str(table),
                "batch": report.batch,
                "segment": report.segment,
                "status": report.status,
                "rows_checked": report.rows_checked,
                "n_alerts": len(report.alerts),
                "suspect_columns": ", ".join(report.suspect_columns[:8]),
                "top_check": top.check if top else "",
                "top_column": top.column if top else "",
                "top_ratio": top.ratio if top else 0.0,
                "details": json.dumps(details, default=str),
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _finding_note(finding: Any) -> str:
    """Where a finding's rows are and what else it affected, on one line."""
    affects = f"also affected: {', '.join(finding.affects)}" if finding.affects else ""
    return "; ".join(part for part in (finding.where, affects) if part)


def fit_table(
    source: Any,
    table: str,
    batch_col: str,
    store: str | Store,
    key: str | None = None,
    as_date: bool = True,
    where: str | None = None,
    until: str | None = None,
    slice_col: str | None = None,
    max_slices: int = 8,
    min_slice_rows: int = 500,
    slice_sample_rows: int = 2000,
    train_fraction: float = 0.5,
    score_rows: int | None = None,
    refit_every: int | None = None,
    exclude_batches: list[str] | tuple[str, ...] = (),
    dialect: str | None = None,
    **monitor_settings: Any,
) -> Monitor:
    """Train and calibrate a monitor on a table's history and save it to `store`.

    source       a connection, session or URL (see morello.sql.connect)
    as_date      batches are calendar days of batch_col; False uses its distinct values
    where        optional SQL filter, inserted as written
    until        only use batches up to and including this key (default: all of them)
    slice_col    also keep a separate alert history for each large value of this column
    score_rows   Spark only: score this many rows per batch on the cluster, not sample_rows
    refit_every  check_table fits again after this many checked batches (see refit_table)
    exclude_batches  batches to leave out of training and calibration
    monitor_settings are passed to Monitor (margin, sample_rows and so on)
    """
    source = connect(source, dialect)
    sampler = TableSampler(source, table, batch_col, as_date, where)
    if exclude_batches:
        excluded = ", ".join(literal(batch) for batch in exclude_batches)
        clean_only = f"{sampler.key_expression} NOT IN ({excluded})"
        sampler.where = f"({where}) AND {clean_only}" if where else clean_only
    counts = sampler.batch_counts()
    if until is not None:
        counts = counts[counts["batch"] <= str(until)]
    batches = list(counts["batch"])
    n_rows = dict(zip(counts["batch"], counts["n_rows"], strict=True))
    if len(batches) < MIN_FIT_BATCHES:
        raise ValueError(
            f"need at least {MIN_FIT_BATCHES} batches of history to calibrate alerts, "
            f"got {len(batches)}"
        )
    n_train = max(1, int(len(batches) * train_fraction))
    train_batches, calibration_batches = batches[:n_train], batches[n_train:]

    monitor = Monitor(**monitor_settings)
    monitor.choose_sample_rows(counts["n_rows"])
    train = sampler.sample(
        train_batches[0],
        train_batches[-1],
        sum(n_rows[batch] for batch in train_batches),
        monitor.max_train_rows,
        monitor.seed,
    )
    monitor.fit_model(train, _actual_name(train, batch_col))
    monitor.table_config = TableConfig(
        table=str(table),
        batch_col=batch_col,
        as_date=as_date,
        where=where,
        slice_col=slice_col,
        slices=_large_slices(sampler, slice_col, train_batches, min_slice_rows, max_slices),
        slice_sample_rows=slice_sample_rows,
        score_rows=score_rows,
        max_slices=max_slices,
        min_slice_rows=min_slice_rows,
        train_fraction=train_fraction,
        refit_every=refit_every,
    )

    slice_counts = _slice_counts(sampler, monitor.table_config, since=calibration_batches[0])
    for batch in calibration_batches:
        _check_batch(monitor, source, sampler, batch, n_rows[batch], "always", slice_counts)
    monitor.last_checked = batches[-1]
    monitor.save(open_store(store), key or default_key(table))
    return monitor


def check_table(
    source: Any,
    store: str | Store,
    key: str | None = None,
    table: str | None = None,
    batch: str | None = None,
    results_table: str | None = None,
    update: str = "auto",
    max_batches: int = 31,
    dialect: str | None = None,
) -> list[Report]:
    """Check the batches that arrived since the last run, oldest first, or one named batch.

    Returns the reports and, when results_table is given, appends one row per report to it.
    At most max_batches are checked per run; a later run continues where this one stopped.

    A batch that alerts stays unresolved (see unresolved_alerts) until it is accepted or is
    checked again without alerting.
    """
    return _run(source, store, key, table, batch, results_table, update, max_batches, dialect)


def accept_batch(
    source: Any,
    store: str | Store,
    batch: str,
    key: str | None = None,
    table: str | None = None,
    results_table: str | None = None,
    dialect: str | None = None,
) -> list[Report]:
    """Record that an alerting batch was reviewed and is fine.

    Tolerances widen to include the batch and it stops being unresolved. With results_table,
    one row per report is appended with status "accepted", so jobs that read that table see
    the decision.
    """
    return _run(
        source, store, key, table, batch, results_table, "always", 1, dialect, status=ACCEPTED
    )


def refit_table(
    source: Any,
    store: str | Store,
    key: str | None = None,
    table: str | None = None,
    dialect: str | None = None,
) -> Monitor:
    """Train again on the table as it is now, with the settings of the saved monitor.

    Use it when the table has changed for good: new columns, new categories, a new normal.
    Batches with an unresolved alert are left out, so a fault that nobody accepted is not
    learned as normal. They stay unresolved, and checking continues after the last batch
    already checked.
    """
    store = open_store(store)
    key = _key(key, table)
    old = Monitor.load(store, key)
    config = old.table_config
    if config is None:
        raise ValueError(f"monitor {key!r} was not fitted with fit_table")
    monitor = fit_table(
        source,
        table or config.table,
        config.batch_col,
        store,
        key,
        as_date=config.as_date,
        where=config.where,
        until=old.last_checked,
        slice_col=config.slice_col,
        max_slices=config.max_slices,
        min_slice_rows=config.min_slice_rows,
        slice_sample_rows=config.slice_sample_rows,
        train_fraction=config.train_fraction,
        score_rows=config.score_rows,
        refit_every=config.refit_every,
        exclude_batches=old.unresolved_alerts,
        dialect=dialect,
        **{name: getattr(old, name) for name in Monitor.SETTINGS},
    )
    monitor.last_checked, monitor.unresolved_alerts = old.last_checked, old.unresolved_alerts
    monitor.save(store, key, history_only=True)
    return monitor


def unresolved_alerts(
    store: str | Store, key: str | None = None, table: str | None = None
) -> list[str]:
    """Batches whose latest check alerted and that nobody has accepted, oldest first.

    A gate should stop while this list is not empty. Asking again gives the same answer, so a
    task that is retried after an alert fails again where it would otherwise find nothing new
    to check and pass. Reads only the small history document.
    """
    data = open_store(store).get(f"{_key(key, table)}.history")
    if data is None:
        return []
    return list(state.unpack(data, state.HISTORY).get("unresolved_alerts", []))


def alert_summary(reports: list[Report], unresolved: list[str]) -> str:
    """What should stop a pipeline, as text. Empty when nothing should.

    Covers alerts in `reports` and batches still unresolved from earlier runs.
    """
    alerting = [report for report in reports if report.alert]
    reported = {report.batch for report in alerting}
    lines = [report.text() for report in alerting]
    lines += [
        f"batch {batch}: alerted in an earlier run and has not been accepted"
        for batch in unresolved
        if batch not in reported
    ]
    return "\n".join(lines)


def _run(
    source: Any,
    store: str | Store,
    key: str | None,
    table: str | None,
    batch: str | None,
    results_table: str | None,
    update: str,
    max_batches: int,
    dialect: str | None,
    status: str | None = None,
) -> list[Report]:
    """check_table and accept_batch. `status`, when given, replaces the status written to
    the results table."""
    source, store = connect(source, dialect), open_store(store)
    key = _key(key, table)
    monitor = Monitor.load(store, key)
    config = monitor.table_config
    if config is None:
        raise ValueError(f"monitor {key!r} was not fitted with fit_table")
    table = table or config.table
    one_batch = batch is not None
    refit_is_due = config.refit_every and monitor.checked_since_fit >= config.refit_every
    if refit_is_due and not one_batch and update != "never":
        log.info("fitting again after %d checked batches", monitor.checked_since_fit)
        monitor = refit_table(source, store, key, table)
    sampler = TableSampler(source, table, config.batch_col, config.as_date, config.where)

    since = str(batch) if one_batch else monitor.last_checked
    counts = sampler.batch_counts(since=since)
    n_rows = dict(zip(counts["batch"], counts["n_rows"], strict=True))
    if one_batch:
        pending = [str(batch)] if str(batch) in n_rows else []
    else:
        pending = [b for b in counts["batch"] if since is None or b > since]
        if len(pending) > max_batches:
            log.warning(
                "%d batches are waiting; checking the oldest %d this run", len(pending), max_batches
            )
            pending = pending[:max_batches]

    slice_counts = _slice_counts(sampler, config, since=pending[0]) if pending else {}
    reports: list[Report] = []
    unresolved = set(monitor.unresolved_alerts)
    for pending_batch in pending:
        batch_reports = _check_batch(
            monitor, source, sampler, pending_batch, n_rows[pending_batch], update, slice_counts
        )
        reports += batch_reports
        if update != "always" and any(report.alert for report in batch_reports):
            unresolved.add(pending_batch)
        else:
            unresolved.discard(pending_batch)  # accepted, or checked again and clean
        if not one_batch:
            monitor.last_checked = pending_batch
            monitor.checked_since_fit += 1
    monitor.unresolved_alerts = sorted(unresolved)

    if reports and results_table:
        rows = result_rows(table, reports)
        if status is not None:
            rows["status"] = status
        source.append(results_table, rows)
    if update != "never":
        monitor.save(store, key, history_only=True)
    return reports


def _key(key: str | None, table: str | None) -> str:
    if key is not None:
        return key
    if table is None:
        raise ValueError("give key or table")
    return default_key(table)


def _check_batch(
    monitor: Monitor,
    source: Source,
    sampler: TableSampler,
    batch: str,
    n_rows: int,
    update: str,
    slice_counts: SliceCounts,
) -> list[Report]:
    """Check one batch of the whole table, then each configured slice of it."""
    config = monitor.table_config
    rows = sampler.sample(batch, batch, n_rows, monitor.sample_rows, monitor.seed)
    if len(rows) == 0:
        return []

    if config.score_rows and hasattr(source, "frame"):
        # Score a larger sample on the cluster. The driver's small sample supplies the two
        # statistics that need all rows in one place, and the example rows.
        large_sample = source.frame(sampler.sample_query(batch, batch, n_rows, config.score_rows))
        stats = distributed_statistics(monitor, large_sample)
        encoded = monitor.schema.transform(rows)
        _, scores = monitor.statistics(encoded)
        sample = monitor.checks.sample_statistics(encoded)
        reports = [monitor.evaluate(stats, sample, batch, update, rows=rows, scores=scores)]
    else:
        reports = [monitor.check(rows, batch=batch, update=update)]

    if config.slices:
        samples = sampler.sample_slices(
            batch,
            config.slice_col,
            slice_counts.get(batch, {}),
            config.slice_sample_rows,
            monitor.seed,
        )
        for value, slice_rows in samples.items():
            if len(slice_rows) >= MIN_SLICE_CHECK_ROWS:
                segment = f"{config.slice_col}={value}"
                reports.append(
                    monitor.check(slice_rows, batch=batch, update=update, segment=segment)
                )
    return reports


def _large_slices(
    sampler: TableSampler,
    slice_col: str | None,
    train_batches: list[str],
    min_rows_per_batch: int,
    max_slices: int,
) -> list[str]:
    """Values of slice_col with enough rows per training batch to be checked on their own."""
    if not slice_col:
        return []
    counts = sampler.batch_counts(slice_col)
    counts = counts[counts["batch"].isin(train_batches) & counts["slice"].notna()]
    rows_per_batch = counts.groupby("slice")["n_rows"].sum() / len(train_batches)
    large = rows_per_batch[rows_per_batch >= min_rows_per_batch].sort_values(ascending=False)
    return [str(value) for value in large.index[:max_slices]]


def _slice_counts(sampler: TableSampler, config: TableConfig, since: str | None) -> SliceCounts:
    if not config.slices:
        return {}
    counts = sampler.batch_counts(config.slice_col, since=since)
    by_batch: SliceCounts = {}
    for batch, value, n_rows in zip(
        counts["batch"], counts["slice"], counts["n_rows"], strict=True
    ):
        if value in config.slices:
            by_batch.setdefault(batch, {})[value] = int(n_rows)
    return by_batch


def _actual_name(rows: pd.DataFrame, column: str) -> str:
    """The column's name as the engine returned it (engines differ on upper and lower case)."""
    return next((name for name in rows.columns if name.lower() == column.lower()), column)
