"""Spark and Databricks.

    from morello.integrations import spark as tbs
    tbs.fit(spark, "catalog.schema.orders", "order_ts", store="dbfs:/morello", state_table=None)
    reports = tbs.check(spark, "catalog.schema.orders", store="dbfs:/morello", results_table="ops.morello_results")

Sampling runs as Spark SQL, so only the sampled rows reach the driver. With score_rows set,
scoring itself runs on the executors and only sums and counts come back.
"""
import pandas as pd

from ..monitor import merge_stats
from ..sql import SparkSource
from ..state import dump_model, load_monitor, pack_stats, unpack_stats
from ..store import TableStore
from ..table import fit_table, check_table

_CACHE = {}


def distributed_stats(mon, sdf):
    """Score a Spark DataFrame on the executors. Returns the batch's additive statistics."""
    spark = sdf.sparkSession
    bc = spark.sparkContext.broadcast(dump_model(mon))

    def score(batches):
        key = id(bc)
        if key not in _CACHE:
            _CACHE.clear(); _CACHE[key] = load_monitor(bc.value)
        m = _CACHE[key]
        for pdf in batches:
            if len(pdf):
                st, _ = m.stats(m.schema.transform(pdf))
                yield pd.DataFrame({"s": [pack_stats(st)]})

    rows = sdf.mapInPandas(score, "s binary").collect()
    return merge_stats([unpack_stats(r["s"]) for r in rows])


def _store(spark, store, state_table):
    return TableStore(SparkSource(spark), state_table) if state_table else store


def fit(spark, table, batch_col, store=None, state_table=None, **kw):
    """Fit a monitor on a table or view. Keep state in cloud storage (store=...) or in a table (state_table=...)."""
    return fit_table(SparkSource(spark), table, batch_col, _store(spark, store, state_table), **kw)


def check(spark, table, store=None, state_table=None, **kw):
    return check_table(SparkSource(spark), _store(spark, store, state_table), table=table, **kw)


def fit_dataframe(df, batch_col, store=None, state_table=None, view="_morello_src", **kw):
    """Same as fit, for a DataFrame that is not a table."""
    df.createOrReplaceTempView(view)
    return fit(df.sparkSession, view, batch_col, store, state_table, **kw)


def check_dataframe(df, store=None, state_table=None, view="_morello_src", **kw):
    df.createOrReplaceTempView(view)
    return check(df.sparkSession, view, store, state_table, **kw)


def fail_on_alert(reports):
    """Raise if any report alerts, to stop a job or workflow task."""
    bad = [r for r in reports if r.alert]
    if bad:
        raise RuntimeError("morello alert:\n" + "\n".join(r.text() for r in bad))
    return reports
