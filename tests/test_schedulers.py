"""The Airflow operator and the Dagster asset check.

The first test runs everywhere. The other two run the adapters inside Airflow and Dagster and
are skipped when those are not installed:

    pip install apache-airflow dagster
"""

import importlib
import os
import subprocess
import sys
import tempfile

import duckdb
import pytest

from conftest import make_orders
from morello.table import accept_batch, fit_table

# Airflow reads these when it is first imported, which importing the operator does.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp(prefix="morello-airflow-"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
os.environ.setdefault("AIRFLOW__CORE__DAGS_FOLDER", tempfile.mkdtemp(prefix="morello-dags-"))

from morello.integrations.airflow import AirflowException, MorelloCheckOperator  # noqa: E402
from morello.integrations.dagster import morello_check_result  # noqa: E402

BROKEN_DAY = "2024-03-03"


class Warehouse:
    """A DuckDB file with orders through 2 March and a monitor fitted through February."""

    def __init__(self, folder) -> None:
        self.database = str(folder / "wh.duckdb")
        self.store = str(folder / "state")
        with self.connect() as connection:
            connection.register("staged", make_orders(n_days=62))
            connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
            fit_table(
                connection, "orders", "order_ts", self.store, until="2024-02-29", sample_rows=1000
            )

    def connect(self):
        return duckdb.connect(self.database)

    def add_broken_day(self) -> None:
        day = make_orders(n_days=63).query("order_ts >= @BROKEN_DAY").copy()
        day.loc[day["region"] == "US", "currency"] = "EUR"
        with self.connect() as connection:
            connection.register("new_day", day)
            connection.execute("INSERT INTO orders SELECT * FROM new_day")

    def accept_broken_day(self) -> None:
        with self.connect() as connection:
            accept_batch(connection, self.store, BROKEN_DAY, table="orders")

    def operator(self, **options) -> MorelloCheckOperator:
        return MorelloCheckOperator(
            task_id="orders_quality",
            connect=self.connect,
            table="orders",
            store=self.store,
            results_table="morello_results",
            **options,
        )


@pytest.fixture
def warehouse(tmp_path) -> Warehouse:
    return Warehouse(tmp_path)


def test_operator_fails_on_an_alert_and_on_every_retry(warehouse):
    reports = warehouse.operator().execute()
    assert [(report["batch"], report["status"]) for report in reports] == [
        ("2024-03-01", "ok"),
        ("2024-03-02", "ok"),
    ]

    warehouse.add_broken_day()
    with pytest.raises(AirflowException, match="currency: "):
        warehouse.operator().execute()
    with pytest.raises(AirflowException, match="alerted in an earlier run"):
        warehouse.operator().execute()
    assert warehouse.operator(fail_on_alert=False).execute() == []

    warehouse.accept_broken_day()
    assert warehouse.operator().execute() == []


DAG_FILE = """
import duckdb
from airflow.sdk import DAG

from morello.integrations.airflow import MorelloCheckOperator

with DAG(dag_id="morello_orders", schedule=None, catchup=False) as dag:
    MorelloCheckOperator(
        task_id="orders_quality",
        connect=lambda: duckdb.connect({database!r}),
        table="orders",
        store={store!r},
        results_table="morello_results",
    )
"""


def test_operator_inside_airflow(warehouse, monkeypatch):
    pytest.importorskip("airflow")
    # Airflow only runs DAGs it finds in its DAG folder, so the DAG is written there as a file.
    dag_folder = os.environ["AIRFLOW__CORE__DAGS_FOLDER"]
    with open(os.path.join(dag_folder, "morello_orders_dag.py"), "w") as file:
        file.write(DAG_FILE.format(database=warehouse.database, store=warehouse.store))
    migrate = [sys.executable, "-m", "airflow", "db", "migrate"]
    subprocess.run(migrate, check=True, capture_output=True)
    monkeypatch.syspath_prepend(dag_folder)
    dag = importlib.import_module("morello_orders_dag").dag

    def run() -> tuple[str, str]:
        dag_run = dag.test()
        task = dag_run.get_task_instance("orders_quality")
        return str(dag_run.state), str(task.state)

    assert run() == ("success", "success")
    warehouse.add_broken_day()
    assert run() == ("failed", "failed")
    assert run() == ("failed", "failed")  # a retry with nothing new to check
    warehouse.accept_broken_day()
    assert run() == ("success", "success")


def test_asset_check_inside_dagster(warehouse):
    dagster = pytest.importorskip("dagster")

    @dagster.asset
    def orders() -> None:
        """Stands in for the job that loads the orders table."""

    @dagster.asset_check(asset=orders)
    def orders_quality():
        with warehouse.connect() as connection:
            return morello_check_result(connection, table="orders", store=warehouse.store)

    def run() -> tuple[bool, dict]:
        result = dagster.materialize([orders, orders_quality], raise_on_error=False)
        evaluation = result.get_asset_check_evaluations()[0]
        metadata = {name: value.value for name, value in evaluation.metadata.items()}
        return evaluation.passed, metadata

    passed, metadata = run()
    assert passed and metadata["batches_checked"] == 2

    warehouse.add_broken_day()
    passed, metadata = run()
    assert not passed
    assert metadata["suspect_columns"] == "currency"
    assert metadata["unresolved_batches"] == BROKEN_DAY

    passed, metadata = run()  # executed again with nothing new to check
    assert not passed and "alerted in an earlier run" in metadata["report"]

    warehouse.accept_broken_day()
    passed, _ = run()
    assert passed
