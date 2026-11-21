"""Replay a table's history to see which past batches would have alerted.

The first part of the history fits the monitor, exactly as `fit` would. Every later batch is
then checked in order, the way a scheduled job would have checked it on the day. On history
that is believed to be clean, each alert is a false alarm, so the replay measures how noisy
the monitor would be on this table. On history with known incidents, it shows which of them
the monitor would have caught.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .monitor import MIN_FIT_BATCHES, Monitor, batch_keys
from .report import Report
from .sql import TableSampler, connect
from .table import check_table, fit_table
from .term import paint


@dataclass
class Backtest:
    fit_batches: int  # batches at the start of the history used to fit
    reports: list[Report]  # one per replayed batch (and per slice, for tables with slices)

    @property
    def batches(self) -> list[str]:
        return list(dict.fromkeys(report.batch for report in self.reports))

    @property
    def alerting(self) -> list[Report]:
        return [report for report in self.reports if report.alert]

    @property
    def alerting_batches(self) -> list[str]:
        return list(dict.fromkeys(report.batch for report in self.alerting))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_batches": self.fit_batches,
            "replayed_batches": len(self.batches),
            "alerting_batches": self.alerting_batches,
            "reports": [report.to_dict() for report in self.alerting],
        }

    def text(self, color: bool = False) -> str:
        replayed, alerted = len(self.batches), len(self.alerting_batches)
        share = f" ({alerted / replayed:.0%})" if replayed else ""
        lines = [
            f"fitted on the first {self.fit_batches} batches, replayed the next {replayed}",
            f"{alerted} of {replayed} batches would have alerted{share}",
        ]
        for report in self.alerting:
            segment = f" [{report.segment}]" if report.segment else ""
            first = report.findings[0].summary if report.findings else ""
            more = f" (+{len(report.findings) - 1} more)" if len(report.findings) > 1 else ""
            lines.append(f"  {paint(str(report.batch), 'alert', color)}{segment}  {first}{more}")
        if alerted:
            lines.append(
                "An alerting batch is not learned, as in production. Check each one: a real "
                "incident is a catch, a normal day is a false alarm."
            )
        return "\n".join(lines)


def default_fit_batches(n_batches: int, min_history: int) -> int:
    """Half the history, and never fewer batches than alerts need to switch on."""
    return max(2 * min_history, MIN_FIT_BATCHES, n_batches // 2)


def backtest(
    table: pd.DataFrame,
    batch_col: str,
    freq: str = "D",
    fit_batches: int | None = None,
    **monitor_settings: Any,
) -> Backtest:
    """Replay a history held in a DataFrame. monitor_settings are passed to Monitor."""
    keys = batch_keys(table[batch_col], freq)
    batches = sorted(keys.dropna().unique())
    monitor = Monitor(**monitor_settings)
    n_fit = _fit_batches(fit_batches, len(batches), monitor.min_history)
    monitor.fit(table[keys.isin(set(batches[:n_fit]))], batch_col, freq)
    reports = [monitor.check(table[keys == batch], batch=batch) for batch in batches[n_fit:]]
    return Backtest(n_fit, reports)


def backtest_table(
    source: Any,
    table: str,
    batch_col: str,
    fit_batches: int | None = None,
    as_date: bool = True,
    where: str | None = None,
    dialect: str | None = None,
    **fit_options: Any,
) -> Backtest:
    """Replay the history of a table in a warehouse. Nothing is saved.

    fit_options are passed to fit_table (slice_col, sample_rows, margin and so on).
    """
    source = connect(source, dialect)
    counts = TableSampler(source, table, batch_col, as_date, where).batch_counts()
    batches = list(counts["batch"])
    min_history = fit_options.get("min_history", Monitor().min_history)
    n_fit = _fit_batches(fit_batches, len(batches), min_history)
    with tempfile.TemporaryDirectory() as scratch:
        fit_table(
            source,
            table,
            batch_col,
            scratch,
            as_date=as_date,
            where=where,
            until=batches[n_fit - 1],
            **fit_options,
        )
        reports = check_table(source, scratch, table=table, max_batches=len(batches))
    return Backtest(n_fit, reports)


def _fit_batches(requested: int | None, n_batches: int, min_history: int) -> int:
    n_fit = requested or default_fit_batches(n_batches, min_history)
    if n_batches <= n_fit:
        raise ValueError(
            f"a backtest needs more than {n_fit} batches of history "
            f"({n_fit} to fit, the rest to replay), got {n_batches}"
        )
    return n_fit
