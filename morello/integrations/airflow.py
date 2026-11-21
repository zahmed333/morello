"""Airflow operator. NOT RUN INSIDE AN AIRFLOW DEPLOYMENT YET; the function it calls is tested.

MorelloCheckOperator(
    task_id="orders_quality",
    connect=lambda: my_connection(),
    table="analytics.orders",
    store="s3://bucket/morello",
    results_table="ops.morello_results",
)
"""

from __future__ import annotations

from typing import Any, Callable

from ..store import Store
from ..table import check_table

try:
    from airflow.exceptions import AirflowException
    from airflow.models import BaseOperator
except ImportError:  # lets the module be imported, and the operator tested, without Airflow
    AirflowException = RuntimeError
    BaseOperator = object


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
        alerting = [report for report in reports if report.alert]
        if alerting and self.fail_on_alert:
            raise AirflowException(
                "morello alert:\n" + "\n".join(report.text() for report in alerting)
            )
        return [report.to_dict() for report in reports]
