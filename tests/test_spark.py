"""Spark adapter on a local session: SQL sampling, state in a table, scoring on executors."""
import tempfile
import numpy as np, pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
from morello import merge_stats
from morello.integrations import spark as tbs
from test_table_duckdb import orders


def test_spark_adapter():
    from pyspark.sql import SparkSession
    with tempfile.TemporaryDirectory() as t:
        spark = (SparkSession.builder.master("local[2]").appName("morello-test")
                 .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4")
                 .config("spark.sql.warehouse.dir", f"{t}/wh").config("spark.driver.memory", "1g").getOrCreate())
        pdf = orders(n_days=49, n=800)
        hist, new = pdf[pdf.order_ts < "2024-02-18"], pdf[pdf.order_ts >= "2024-02-18"].copy()
        spark.createDataFrame(hist).createOrReplaceTempView("orders")
        mon = tbs.fit(spark, "orders", "order_ts", state_table="morello_state", sample_rows=800, score_rows=2000)
        assert mon.batches_until_ready == 0 and mon.table_cfg["score_rows"] == 2000
        # executors and driver agree on the statistics of the same rows
        day = hist[hist.order_ts >= "2024-02-17"]
        local, _ = mon.stats(mon.schema.transform(day))
        dist = tbs.distributed_stats(mon, spark.createDataFrame(day).repartition(3))
        for k in local:
            assert np.allclose(local[k], dist[k], rtol=1e-4, atol=1e-3), k
        # a clean new day passes, a broken one alerts and names the column
        spark.createDataFrame(pd.concat([hist, new])).createOrReplaceTempView("orders")
        reps = tbs.check(spark, "orders", state_table="morello_state", results_table="morello_results")
        assert [r.batch for r in reps] == ["2024-02-18"] and not reps[0].alert
        bad = new.copy(); bad["order_ts"] += pd.Timedelta(days=1)
        us = bad["region"] == "US"; bad.loc[us, "currency"] = "EUR"
        spark.createDataFrame(pd.concat([hist, new, bad])).createOrReplaceTempView("orders")
        reps = tbs.check(spark, "orders", state_table="morello_state", results_table="morello_results")
        assert reps[0].batch == "2024-02-19" and reps[0].alert and "currency" in reps[0].suspects
        with pytest.raises(RuntimeError):
            tbs.fail_on_alert(reps)
        got = spark.sql("select batch, status from morello_results order by batch").toPandas()
        assert list(got["status"]) == ["ok", "alert"]
        spark.stop()


if __name__ == "__main__":
    test_spark_adapter(); print("ok")
