"""Replaying history, profiling a table before fitting, and the sample size."""

import duckdb
import pytest

from conftest import make_orders
from morello import Monitor
from morello.backtest import backtest, backtest_table
from morello.cli import main
from morello.monitor import suggest_sample_rows
from morello.profile import profile_frame, profile_table


def orders_with_a_bad_day(n_days: int = 62, bad_day: str = "2024-02-20"):
    orders = make_orders(n_days=n_days)
    bad = (orders["order_ts"].astype(str).str[:10] == bad_day) & (orders["region"] == "US")
    orders.loc[bad, "currency"] = "EUR"
    return orders


def test_backtest_finds_the_bad_day_and_nothing_else():
    result = backtest(orders_with_a_bad_day(), "order_ts", sample_rows=1000)
    assert result.fit_batches == 31 and len(result.batches) == 31
    assert result.alerting_batches == ["2024-02-20"]
    text = result.text()
    assert "1 of 31 batches would have alerted (3%)" in text
    assert "2024-02-20  currency:" in text


def test_backtest_of_a_table_saves_nothing(tmp_path):
    connection = duckdb.connect()
    connection.register("staged", orders_with_a_bad_day())
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    connection.unregister("staged")
    connection.execute("SELECT setseed(0.5)").fetchall()  # row sampling uses random()
    result = backtest_table(connection, "orders", "order_ts", sample_rows=1000)
    assert result.alerting_batches == ["2024-02-20"]
    tables = connection.execute("SHOW TABLES").fetchall()
    assert tables == [("orders",)]


def test_backtest_needs_history_left_to_replay():
    with pytest.raises(ValueError, match="more than 28 batches"):
        backtest(make_orders(n_days=25), "order_ts")


def test_backtest_and_init_from_the_command_line(tmp_path, capsys):
    path = tmp_path / "orders.parquet"
    orders_with_a_bad_day().to_parquet(path)
    assert main(["backtest", str(path), "--batch-col", "order_ts", "--sample-rows", "1000"]) == 0
    assert "1 of 31 batches would have alerted" in capsys.readouterr().out

    assert main(["init", str(path)]) == 0
    out = capsys.readouterr().out
    assert "batch column: order_ts (62 daily batches, about 1,500 rows each)" in out
    assert f"next: morello fit {path} --batch-col order_ts --model STATE" in out


def test_profile_of_a_table_and_of_a_short_history():
    connection = duckdb.connect()
    connection.register("staged", make_orders())
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    profile = profile_table(connection, "orders")
    assert profile.batch_column.column == "order_ts" and profile.ready
    assert profile.format_only == {"order_id": "identifier or free text"}

    short = profile_frame(make_orders(n_days=8))
    assert not short.ready
    assert "not enough history: 8 batches" in short.text()


def test_sample_size_follows_the_batch_size_and_sets_the_detection_floor():
    assert suggest_sample_rows([800, 900, 1000]) == 5000
    assert suggest_sample_rows([20_000, 24_000, 31_000]) == 24_000
    assert suggest_sample_rows([4_000_000]) == 50_000
    assert Monitor(sample_rows=5000).smallest_detectable_share == pytest.approx(0.0018)
    assert Monitor(sample_rows=50_000).smallest_detectable_share == pytest.approx(0.00018)
