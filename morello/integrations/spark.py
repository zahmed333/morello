"""Spark and Databricks.

    from morello.integrations import spark as morello_spark

    morello_spark.fit(spark, "catalog.schema.orders", "order_ts", store="dbfs:/morello")
    reports = morello_spark.check(
        spark, "catalog.schema.orders", store="dbfs:/morello", results_table="ops.morello_results"
    )
    morello_spark.fail_on_alert(reports)

Sampling runs as Spark SQL, so only the sampled rows reach the driver. With score_rows set,
scoring itself runs on the executors and only sums and counts come back.
"""

from __future__ import annotations

from typing import Any

from ..distributed import distributed_statistics
from ..monitor import Monitor
from ..report import Report
from ..sql import SparkSource
from ..store import Store, TableStore
from ..table import check_table, fit_table

__all__ = [
    "check",
    "check_dataframe",
    "distributed_statistics",
    "fail_on_alert",
    "fit",
    "fit_dataframe",
]
SOURCE_VIEW = "_morello_src"


def fit(
    spark: Any,
    table: str,
    batch_col: str,
    store: str | Store | None = None,
    state_table: str | None = None,
    **options: Any,
) -> Monitor:
    """Fit a monitor on a table or view.

    Keep the saved monitor in cloud storage (store=...) or in a table (state_table=...).
    Other options are those of morello.table.fit_table.
    """
    return fit_table(
        SparkSource(spark), table, batch_col, _store(spark, store, state_table), **options
    )


def check(
    spark: Any,
    table: str,
    store: str | Store | None = None,
    state_table: str | None = None,
    **options: Any,
) -> list[Report]:
    """Check new batches of a table or view. Options are those of morello.table.check_table."""
    return check_table(
        SparkSource(spark), _store(spark, store, state_table), table=table, **options
    )


def fit_dataframe(frame: Any, batch_col: str, **options: Any) -> Monitor:
    """Same as fit, for a DataFrame that is not a table."""
    frame.createOrReplaceTempView(SOURCE_VIEW)
    return fit(frame.sparkSession, SOURCE_VIEW, batch_col, **options)


def check_dataframe(frame: Any, **options: Any) -> list[Report]:
    frame.createOrReplaceTempView(SOURCE_VIEW)
    return check(frame.sparkSession, SOURCE_VIEW, **options)


def fail_on_alert(reports: list[Report]) -> list[Report]:
    """Raise if any report alerts, to stop a job or a workflow task."""
    alerting = [report for report in reports if report.alert]
    if alerting:
        raise RuntimeError("morello alert:\n" + "\n".join(report.text() for report in alerting))
    return reports


def _store(spark: Any, store: str | Store | None, state_table: str | None) -> str | Store:
    if state_table:
        return TableStore(SparkSource(spark), state_table)
    if store is None:
        raise ValueError("give store or state_table")
    return store
