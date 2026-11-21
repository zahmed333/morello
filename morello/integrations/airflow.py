"""Airflow operator. Run on Airflow 3.4 with a DuckDB table (see docs/integrations.md).

    MorelloCheckOperator(
        task_id="orders_quality",
        connect=lambda: my_connection(),
        table="analytics.orders",
        store="s3://bucket/morello",
        results_table="ops.morello_results",
    )

The task fails when a batch alerts, and keeps failing on retries and later runs until the batch
is accepted (`morello accept`) or passes a re-check. It returns the reports as XCom.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..store import Store
from ..table import alert_summary, check_table, unresolved_alerts

try:
    from airflow.sdk import BaseOperator  # Airflow 3
except ImportError:
    try:
        from airflow.models import BaseOperator  # Airflow 2
    except ImportError:  # lets the module be imported, and the operator tested, without Airflow
        BaseOperator = object
try:
    from airflow.exceptions import AirflowException
except ImportError:
    AirflowException = RuntimeError


class MorelloCheckOperator(BaseOperator):
    """Checks new batches of a table and fails the task when one alerts."""

    def __init__(
        self,
        connect: Callable[[], Any],
        table: str,
        store: str | Store,
        key: str | None = None,
        results_table: str | None = None,
        dialect: str | None = None,
        fail_on_alert: bool = True,
        **operator_options: Any,
    ) -> None:
        if BaseOperator is not object:
            super().__init__(**operator_options)
        self.connect = connect  # called on the worker, so no connection is pickled into the DAG
        self.table = table
        self.store = store
        self.key = key
        self.results_table = results_table
        self.dialect = dialect
        self.fail_on_alert = fail_on_alert

    def execute(self, context: Any = None) -> list[dict[str, Any]]:
        reports = check_table(
            self.connect(),
            self.store,
            key=self.key,
            table=self.table,
            results_table=self.results_table,
            dialect=self.dialect,
        )
        problems = alert_summary(reports, unresolved_alerts(self.store, self.key, self.table))
        if problems and self.fail_on_alert:
            raise AirflowException("morello alert:\n" + problems)
        return [report.to_dict() for report in reports]
