"""The Spark adapter on a local session: SQL sampling, state in a table, scoring on executors."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_orders

pytest.importorskip("pyspark")

from morello.integrations import spark as morello_spark  # noqa: E402


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession

    warehouse = tmp_path_factory.mktemp("spark")
    session = (
        SparkSession.builder.master("local[2]")
        .appName("morello-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_spark_adapter(spark):
    orders = make_orders(n_days=49, rows_per_day=800)
    history = orders[orders["order_ts"] < "2024-02-18"]
    new_day = orders[orders["order_ts"] >= "2024-02-18"].copy()

    spark.createDataFrame(history).createOrReplaceTempView("orders")
    monitor = morello_spark.fit(
        spark, "orders", "order_ts", state_table="morello_state", sample_rows=800, score_rows=2000
    )
    assert monitor.batches_until_ready == 0
    assert monitor.table_config.score_rows == 2000

    # Executors and the driver agree on the statistics of the same rows.
    day = history[history["order_ts"] >= "2024-02-17"]
    on_driver, _ = monitor.statistics(monitor.schema.transform(day))
    on_executors = morello_spark.distributed_statistics(
        monitor, spark.createDataFrame(day).repartition(3)
    )
    for name in on_driver:
        assert np.allclose(on_driver[name], on_executors[name], rtol=1e-4, atol=1e-3), name

    # A clean new day passes.
    spark.createDataFrame(pd.concat([history, new_day])).createOrReplaceTempView("orders")
    reports = morello_spark.check(
        spark, "orders", state_table="morello_state", results_table="morello_results"
    )
    assert [report.batch for report in reports] == ["2024-02-18"]
    assert not reports[0].alert

    # A broken day alerts and names the column.
    broken = new_day.copy()
    broken["order_ts"] += pd.Timedelta(days=1)
    broken.loc[broken["region"] == "US", "currency"] = "EUR"
    everything = pd.concat([history, new_day, broken])
    spark.createDataFrame(everything).createOrReplaceTempView("orders")
    reports = morello_spark.check(
        spark, "orders", state_table="morello_state", results_table="morello_results"
    )
    assert reports[0].batch == "2024-02-19" and reports[0].alert
    assert "currency" in reports[0].suspect_columns
    with pytest.raises(RuntimeError, match="morello alert"):
        morello_spark.fail_on_alert(reports)

    results = spark.sql("SELECT batch, status FROM morello_results ORDER BY batch").toPandas()
    assert list(results["status"]) == ["ok", "alert"]
