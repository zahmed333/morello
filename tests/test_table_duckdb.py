"""The table runner on DuckDB: in-engine sampling, state in a table, results, slices, reruns."""

import duckdb
import pytest

from conftest import make_orders
from morello import Monitor, TableStore
from morello.sql import connect
from morello.table import accept_batch, check_table, fit_table


def rows_in(connection, table: str) -> int:
    return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]


def add_day(connection, n_days: int, since: str, break_us_currency: bool = False) -> None:
    """Append the days from `since` on, generated the same way as the original table."""
    day = make_orders(n_days=n_days).query("order_ts >= @since").copy()
    if break_us_currency:
        day.loc[day["region"] == "US", "currency"] = "EUR"
    connection.register("new_day", day)
    connection.execute("INSERT INTO orders SELECT * FROM new_day")
    connection.unregister("new_day")


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    """A database with 62 days of orders and a monitor fitted on January and February."""
    path = tmp_path_factory.mktemp("warehouse") / "wh.duckdb"
    connection = duckdb.connect(str(path))
    connection.register("staged", make_orders())
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    store = TableStore(connect(connection), "morello_state")
    monitor = fit_table(
        connection,
        "orders",
        "order_ts",
        store,
        until="2024-02-29",
        slice_col="region",
        min_slice_rows=200,
        slice_sample_rows=500,
        sample_rows=1000,
    )
    yield connection, store, monitor, path
    connection.close()


def test_table_runner_end_to_end(warehouse):
    connection, store, monitor, _ = warehouse
    assert monitor.table_config.slices == ["EU", "US", "APAC"]
    assert "order_id" in monitor.schema.skipped
    assert monitor.batches_until_ready == 0

    # The two March days are new. Each is checked as a whole and once per region.
    reports = check_table(connection, store, table="orders", results_table="morello_results")
    assert sorted({report.batch for report in reports}) == ["2024-03-01", "2024-03-02"]
    assert not any(report.alert for report in reports)
    assert rows_in(connection, "morello_results") == len(reports) == 8

    # Nothing new: nothing is checked and nothing is written.
    assert check_table(connection, store, table="orders", results_table="morello_results") == []
    assert rows_in(connection, "morello_results") == 8

    # A new day where one region's currency no longer matches.
    add_day(connection, n_days=63, since="2024-03-03", break_us_currency=True)
    reports = check_table(connection, store, table="orders", results_table="morello_results")
    by_segment = {report.segment or "all": report for report in reports}
    assert by_segment["all"].alert
    assert by_segment["region=US"].alert and "currency" in by_segment["region=US"].suspect_columns
    assert not by_segment["region=EU"].alert
    status, suspects = connection.execute(
        "SELECT status, suspect_columns FROM morello_results "
        "WHERE batch = '2024-03-03' AND segment = 'region=US'"
    ).fetchall()[0]
    assert status == "alert" and "currency" in suspects

    # The alerting batch was not learned. Accepting it widens the tolerance.
    def alerts_on_the_broken_day() -> int:
        report = check_table(connection, store, table="orders", batch="2024-03-03", update="never")
        return len(report[0].alerts)

    before = alerts_on_the_broken_day()
    accept_batch(connection, store, "2024-03-03", table="orders")
    assert alerts_on_the_broken_day() < before


def test_a_run_that_dies_before_saving_is_repeated_without_double_learning(warehouse, monkeypatch):
    connection, store, _, _ = warehouse
    add_day(connection, n_days=64, since="2024-03-04")
    history_before = len(Monitor.load(store, "orders").history)
    results_before = rows_in(connection, "morello_results")

    def die(*args, **kwargs):
        raise ConnectionError("lost the warehouse while saving")

    with monkeypatch.context() as patch:
        patch.setattr(Monitor, "save", die)
        with pytest.raises(ConnectionError):
            check_table(connection, store, table="orders", results_table="morello_results")
    # The results were written, the history was not, and the batch is still pending.
    first_attempt = rows_in(connection, "morello_results") - results_before
    assert first_attempt == 4
    assert len(Monitor.load(store, "orders").history) == history_before

    reports = check_table(connection, store, table="orders", results_table="morello_results")
    assert {report.batch for report in reports} == {"2024-03-04"}
    assert rows_in(connection, "morello_results") == results_before + 2 * first_attempt
    assert len(Monitor.load(store, "orders").history) == history_before + 1
    assert check_table(connection, store, table="orders") == []


def test_a_backlog_is_worked_through_oldest_first(warehouse):
    connection, store, _, _ = warehouse
    add_day(connection, n_days=67, since="2024-03-05")  # three more days
    first = check_table(connection, store, table="orders", max_batches=2)
    assert sorted({report.batch for report in first}) == ["2024-03-05", "2024-03-06"]
    second = check_table(connection, store, table="orders", max_batches=2)
    assert {report.batch for report in second} == {"2024-03-07"}


def test_a_monitor_fitted_on_files_cannot_run_against_a_table(monitor, tmp_path):
    monitor.save(tmp_path, key="orders")
    with pytest.raises(ValueError, match="not fitted with fit_table"):
        check_table(duckdb.connect(), str(tmp_path), table="orders")
