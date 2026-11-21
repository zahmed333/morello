"""Dagster asset check. Run on Dagster with a DuckDB table (see docs/integrations.md).

    @asset_check(asset=orders)
    def orders_quality():
        return morello_check_result(
            my_connection(), table="analytics.orders", store="s3://bucket/morello"
        )

The check does not pass while a batch has an unresolved alert, including on re-execution.
"""

from __future__ import annotations

from typing import Any

from ..store import Store
from ..table import alert_summary, check_table, unresolved_alerts

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
    unresolved = unresolved_alerts(store, key, table)
    problems = alert_summary(reports, unresolved)
    alerting = [report for report in reports if report.alert]
    suspects = sorted({column for report in alerting for column in report.suspect_columns[:5]})
    return AssetCheckResult(
        passed=not problems,
        metadata={
            "batches_checked": len(reports),
            "alerts": len(alerting),
            "unresolved_batches": ", ".join(unresolved),
            "suspect_columns": ", ".join(suspects),
            "report": problems[:MAX_REPORT_CHARS],
        },
    )
