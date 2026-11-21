"""Look at a table before fitting: which column defines its batches, and is there enough
history. `morello init` prints the result and the command to run next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .monitor import MIN_FIT_BATCHES, Monitor, batch_keys, suggest_sample_rows
from .schema import TableSchema, as_datetime
from .sql import TableSampler, connect, identifier

BATCH_NAME_HINTS = ("batch", "load", "partition", "run", "date", "day")
MAX_CANDIDATES = 3
TABLE_SAMPLE_ROWS = 20_000
PEEK_ROWS = 1_000


@dataclass
class BatchColumn:
    column: str
    as_date: bool
    n_batches: int
    typical_rows: int
    unevenness: float  # spread of batch sizes relative to their median; 0 is perfectly even


@dataclass
class Profile:
    rows: int
    candidates: list[BatchColumn]  # best first
    modelled: list[str]
    format_only: dict[str, str]
    skipped: dict[str, str]
    sample_rows: int = 5000
    notes: list[str] = field(default_factory=list)

    @property
    def batch_column(self) -> BatchColumn | None:
        return self.candidates[0] if self.candidates else None

    @property
    def ready(self) -> bool:
        best = self.batch_column
        return best is not None and best.n_batches >= MIN_FIT_BATCHES

    def text(self) -> str:
        lines = [f"{self.rows:,} rows looked at"]
        best = self.batch_column
        if best is None:
            lines.append(
                "no batch column found: the table needs a date, a timestamp or a load id that "
                "says which rows arrived together"
            )
            return "\n".join(lines)
        unit = "daily batches" if best.as_date else "batches"
        lines.append(
            f"batch column: {best.column} ({best.n_batches} {unit}, "
            f"about {best.typical_rows:,} rows each)"
        )
        if len(self.candidates) > 1:
            others = ", ".join(f"{c.column} ({c.n_batches})" for c in self.candidates[1:])
            lines.append(f"other candidates: {others}")
        lines.append(f"modelled columns: {', '.join(self.modelled) or 'none'}")
        for label, columns in (
            ("format and length checks only", self.format_only),
            ("skipped", self.skipped),
        ):
            if columns:
                lines.append(f"{label}: " + ", ".join(f"{c} ({why})" for c, why in columns.items()))
        scored = max(1, min(self.sample_rows, best.typical_rows))
        floor = Monitor(sample_rows=scored).smallest_detectable_share
        lines.append(
            f"sample: up to {self.sample_rows:,} rows per batch, so a fault has to touch about "
            f"{floor:.2%} of a batch to be noticed"
        )
        needed = 2 * Monitor().min_history
        if best.n_batches < MIN_FIT_BATCHES:
            lines.append(
                f"not enough history: {best.n_batches} batches, and fitting needs "
                f"{MIN_FIT_BATCHES}. Wait for more batches."
            )
        elif best.n_batches < needed:
            lines.append(
                f"enough history to fit. Alerts switch on after {needed - best.n_batches} more "
                "clean batches."
            )
        else:
            lines.append("enough history to fit, and alerts are on from the first check")
        lines += self.notes
        return "\n".join(lines)


def profile_frame(table: pd.DataFrame) -> Profile:
    """Profile a table held in a DataFrame."""
    candidates = []
    for column in table.columns:
        as_date = as_datetime(table[column]) is not None
        if not as_date and not _named_like_a_batch(column):
            continue
        sizes = batch_keys(table[column]).value_counts()
        if 2 <= len(sizes) <= len(table) / 10:
            candidates.append(_candidate(column, as_date, sizes.to_numpy()))
    return _profile(table, len(table), candidates)


def profile_table(source: Any, table: str, dialect: str | None = None) -> Profile:
    """Profile a table in a warehouse with a few grouped counts and one sample of rows."""
    source = connect(source, dialect)
    peek = source.query(f"SELECT * FROM {identifier(table)} LIMIT {PEEK_ROWS}")
    candidates, total_rows = [], len(peek)
    for column in peek.columns:
        as_date = as_datetime(peek[column]) is not None
        if not as_date and not _named_like_a_batch(column):
            continue
        counts = TableSampler(source, table, column, as_date).batch_counts()
        total_rows = max(total_rows, int(counts["n_rows"].sum()))
        if 2 <= len(counts) <= counts["n_rows"].sum() / 10:
            candidates.append(_candidate(column, as_date, counts["n_rows"].to_numpy()))
    rows = peek
    if candidates:
        best = min(candidates, key=_rank)
        counts = TableSampler(source, table, best.column, best.as_date).batch_counts()
        sampler = TableSampler(source, table, best.column, best.as_date)
        first, last = counts["batch"].iloc[0], counts["batch"].iloc[-1]
        rows = sampler.sample(first, last, total_rows, TABLE_SAMPLE_ROWS)
    return _profile(rows, total_rows, candidates)


def _profile(rows: pd.DataFrame, total_rows: int, candidates: list[BatchColumn]) -> Profile:
    candidates = sorted(candidates, key=_rank)[:MAX_CANDIDATES]
    exclude = [candidates[0].column] if candidates else []
    schema = TableSchema().fit(rows, exclude=exclude)
    sample_rows = suggest_sample_rows([candidates[0].typical_rows]) if candidates else 5000
    notes = []
    if candidates and candidates[0].unevenness > 1.0:
        notes.append(
            "batch sizes vary a lot. If small batches are partial loads, checks on them are "
            "noisier."
        )
    return Profile(
        rows=total_rows,
        candidates=candidates,
        modelled=[plan.column for plan in schema.plans if plan.column not in schema.format_only],
        format_only=dict(schema.format_only),
        skipped=dict(schema.skipped),
        sample_rows=sample_rows,
        notes=notes,
    )


def _candidate(column: str, as_date: bool, sizes: np.ndarray) -> BatchColumn:
    typical = float(np.median(sizes))
    unevenness = float(np.std(sizes) / typical) if typical else 0.0
    return BatchColumn(column, as_date, len(sizes), int(typical), unevenness)


def _rank(candidate: BatchColumn) -> tuple:
    """Enough batches first, then dates before ids, then the most even batch sizes."""
    enough = candidate.n_batches >= MIN_FIT_BATCHES
    return (not enough, not candidate.as_date, candidate.unevenness)


def _named_like_a_batch(column: str) -> bool:
    return any(hint in str(column).lower() for hint in BATCH_NAME_HINTS)
