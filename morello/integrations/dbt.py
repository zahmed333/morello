"""dbt Python models (Snowflake, Databricks, BigQuery with Spark, and dbt-duckdb).

models/orders_quality.py:

    from morello.integrations.dbt import run

    def model(dbt, session):
        dbt.config(materialized="incremental")
        return run(session, dbt.ref("orders"), name="orders", batch_col="order_ts",
                   state_table=f"{dbt.this.schema}.morello_state")

dbt finds dependencies by reading the model file, so the dbt.ref("...") call has to appear there.

The model returns one row per batch checked in this run. A test on the model then fails the build:

    models:
      - name: orders_quality
        tests:
          - morello_ok          # macro in macros/morello_ok.sql

The first run trains the monitor (needs 20+ batches of history in the upstream model).
"""
import pandas as pd

from ..sql import connect
from ..store import TableStore
from ..table import fit_table, check_table, result_rows, default_key, RESULT_COLUMNS

VIEW = "_morello_src"


def _as_view(rel, session):
    """Expose the upstream relation under one name on every platform."""
    for meth in ("createOrReplaceTempView", "create_or_replace_temp_view"):      # Spark, Snowpark
        if hasattr(rel, meth):
            getattr(rel, meth)(VIEW)
            return VIEW
    if hasattr(rel, "create_view"):                                                # DuckDB relation
        rel.create_view(VIEW, replace=True)
        return VIEW
    session.register(VIEW, rel)                                                    # pandas / Arrow
    return VIEW


def run(session, relation, name, batch_col, state_table, key=None, **fit_kwargs):
    """relation: what dbt.ref(...) returned. name: label used in the results and as the state key."""
    upstream = name
    src = connect(session)
    view = _as_view(relation, session)
    store, key = TableStore(src, state_table), key or default_key(upstream)
    if store.get(f"{key}.model") is None:
        mon = fit_table(src, view, batch_col, store, key=key, **fit_kwargs)
        return pd.DataFrame([dict(zip(RESULT_COLUMNS, [pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S"), upstream,
                                  mon.cursor.get("last_checked", ""), "", "trained", 0, 0, "", "", "", 0.0, "{}"]))],
                            columns=RESULT_COLUMNS)
    reports = check_table(src, store, key=key, table=view)
    return result_rows(upstream, reports)
