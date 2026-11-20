"""Airflow operator. NOT RUN INSIDE AN AIRFLOW DEPLOYMENT YET; the function it calls is tested.

    MorelloCheckOperator(task_id="orders_quality", conn_factory=lambda: my_connection(),
                           table="analytics.orders", store="s3://bucket/morello",
                           results_table="ops.morello_results")
"""
from ..table import check_table

try:
    from airflow.models import BaseOperator
    from airflow.exceptions import AirflowException
except Exception:                                        # Airflow not installed
    BaseOperator, AirflowException = object, RuntimeError


class MorelloCheckOperator(BaseOperator):
    def __init__(self, conn_factory, table, store, key=None, results_table=None, dialect=None,
                 fail_on_alert=True, **kwargs):
        if BaseOperator is not object:
            super().__init__(**kwargs)
        self.conn_factory, self.table, self.store, self.key = conn_factory, table, store, key
        self.results_table, self.dialect, self.fail_on_alert = results_table, dialect, fail_on_alert

    def execute(self, context=None):
        reports = check_table(self.conn_factory(), self.store, key=self.key, table=self.table,
                              results_table=self.results_table, dialect=self.dialect)
        bad = [r for r in reports if r.alert]
        if bad and self.fail_on_alert:
            raise AirflowException("morello alert:\n" + "\n".join(r.text() for r in bad))
        return [r.to_dict() for r in reports]
