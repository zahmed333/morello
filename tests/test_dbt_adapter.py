"""The function a dbt Python model calls, driven the way dbt-duckdb drives it."""

import duckdb

from conftest import make_orders
from morello.integrations.dbt import run


def test_fit_then_check_then_keep_reporting_an_unresolved_alert():
    connection = duckdb.connect()
    connection.register("staged", make_orders(n_days=62).query("order_ts < '2024-03-01'"))
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    connection.execute("SELECT setseed(0.5)").fetchall()  # row sampling uses random()

    def build():
        rows = run(
            connection,
            connection.table("orders"),
            name="orders",
            batch_col="order_ts",
            state_table="morello_state",
            sample_rows=1000,
        )
        return list(zip(rows["batch"], rows["status"], strict=True))

    def add_day(day: str, break_us_currency: bool = False) -> None:
        new = make_orders(n_days=63).query("order_ts >= @day and order_ts < @day + ' 23:59:59.999'")
        new = new.copy()
        if break_us_currency:
            new.loc[new["region"] == "US", "currency"] = "EUR"
        connection.register("new_day", new)
        connection.execute("INSERT INTO orders SELECT * FROM new_day")
        connection.unregister("new_day")

    assert build() == [("2024-02-29", "trained")]
    add_day("2024-03-01")
    assert build() == [("2024-03-01", "ok")]
    add_day("2024-03-02", break_us_currency=True)
    assert build() == [("2024-03-02", "alert")]
    assert build() == [("2024-03-02", "unresolved")]  # a repeated build with nothing new
    add_day("2024-03-03")
    assert build() == [("2024-03-03", "ok"), ("2024-03-02", "unresolved")]
