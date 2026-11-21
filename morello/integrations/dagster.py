"""Dagster asset check. NOT RUN INSIDE A DAGSTER DEPLOYMENT YET; the function it calls is tested.

@asset_check(asset=orders)
def orders_quality():
    return morello_check_result(
        my_connection(), table="analytics.orders", store="s3://bucket/morello"
    )
"""

from __future__ import annotations

from typing import Any

from ..store import Store
from ..table import check_table

MAX_REPORT_CHARS = 5000


def morello_check_result(
    source: Any,
    table: str,
    store: str | Store,
    key: str | None = None,
    results_table: str | None = None,
    dialect: str | None = None,
) -> Any:
    """Check new batches of a table and return a dagster AssetCheckResult."""
    from dagster import AssetCheckResult

    reports = check_table(
        source, store, key=key, table=table, results_table=results_table, dialect=dialect
    )
    alerting = [report for report in reports if report.alert]
    suspects = sorted({column for report in alerting for column in report.suspect_columns[:5]})
    return AssetCheckResult(
        passed=not alerting,
        metadata={
            "batches_checked": len(reports),
            "alerts": len(alerting),
            "suspect_columns": ", ".join(suspects),
            "report": "\n".join(report.text() for report in alerting)[:MAX_REPORT_CHARS],
        },
    )
