"""Check that the keys in one table exist in another.

The row model reads one table. An order that points at a customer who is missing from the
customers table looks normal row by row, so it needs a check across the two tables.

The check keeps no state. Each run counts, per batch of the child table, the rows whose key
has no match in the parent table, and compares the newest batch with the batches before it
using the same rule as every other check: the change from the usual value, measured against
the largest change seen on an earlier batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .history import RATE_FLOOR_ROWS
from .report import ALERT, OK, WARMING_UP, Alert, Finding, Report
from .sql import TableSampler, connect, identifier, literal

CHECK = "missing reference"
PARENT_KEY = "morello_parent_key"
EXAMPLE_ROWS = 3


def check_reference(
    source: Any,
    child: str,
    child_col: str,
    parent: str,
    parent_col: str,
    batch_col: str,
    batch: str | None = None,
    as_date: bool = True,
    margin: float = 3.0,
    window: int = 28,
    min_history: int = 14,
    dialect: str | None = None,
) -> Report:
    """Check the newest batch of `child` (or the named one) for keys missing from `parent`.

    child_col   the column of `child` that should match parent_col of `parent`
    batch_col   the column of `child` that defines its batches, as in fit_table
    """
    source = connect(source, dialect)
    sampler = TableSampler(source, child, batch_col, as_date)
    joined = (
        f"FROM {sampler.table} LEFT JOIN "
        f"(SELECT DISTINCT {identifier(parent_col)} AS {PARENT_KEY} FROM {identifier(parent)}) p "
        f"ON {identifier(child_col)} = p.{PARENT_KEY}"
    )
    counts = source.query(
        f"SELECT {sampler.key_expression} AS batch, COUNT(*) AS n_rows, "
        f"SUM(CASE WHEN p.{PARENT_KEY} IS NULL THEN 1 ELSE 0 END) AS missing "
        f"{joined} WHERE {sampler.batch_col} IS NOT NULL AND {identifier(child_col)} IS NOT NULL "
        "GROUP BY 1 ORDER BY 1"
    )
    counts.columns = [name.lower() for name in counts.columns]
    counts["batch"] = counts["batch"].astype(str)
    if batch is not None:
        counts = counts[counts["batch"] <= str(batch)]
    if counts.empty or (batch is not None and counts["batch"].iloc[-1] != str(batch)):
        raise ValueError(f"no rows found in {child} for batch {batch!r}")

    shares = (counts["missing"] / counts["n_rows"]).to_numpy(dtype=float)
    newest, earlier = shares[-1], shares[:-1][-window:]
    rows = int(counts["n_rows"].iloc[-1])
    report = Report(
        batch=counts["batch"].iloc[-1],
        status=WARMING_UP,
        rows_checked=rows,
        history_batches=len(earlier),
        segment=f"{child_col} -> {parent}.{parent_col}",
    )
    if len(earlier) < min_history:
        return report

    usual = float(np.median(earlier))
    largest = float(np.max(earlier - usual))
    ratio = (newest - usual) / (max(largest, RATE_FLOOR_ROWS / max(rows, 1)) + 1e-12)
    if ratio <= margin:
        report.status = OK
        return report

    report.status = ALERT
    report.alerts = [Alert(CHECK, child_col, float(newest), usual, float(ratio))]
    report.suspect_columns = [child_col]
    report.findings = [
        Finding(
            column=child_col,
            summary=(
                f"{child_col}: {newest:.2%} of rows point at a {parent}.{parent_col} that does "
                f"not exist (usual {usual:.2%})"
            ),
            ratio=float(ratio),
            checks=[CHECK],
            check=CHECK,
        )
    ]
    examples = source.query(
        f"SELECT {sampler.table}.* {joined} WHERE p.{PARENT_KEY} IS NULL "
        f"AND {identifier(child_col)} IS NOT NULL "
        f"AND {sampler.key_expression} = {literal(report.batch)} LIMIT {EXAMPLE_ROWS}"
    )
    report.examples[child_col] = [
        {name: _plain(value) for name, value in row.items()} for row in examples.to_dict("records")
    ]
    return report


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)
