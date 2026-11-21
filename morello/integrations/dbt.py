"""dbt Python models. Run with dbt-duckdb; the same model file is meant for Snowflake,
Databricks and BigQuery with Spark, where it has not been run.

models/orders_quality.py:

    from morello.integrations.dbt import run

    def model(dbt, session):
        dbt.config(materialized="table")        # or "incremental" to keep every run's rows
        return run(session, dbt.ref("orders"), name="orders", batch_col="order_ts",
                   state_table=f"{dbt.this.schema}.morello_state")

dbt finds dependencies by reading the model file, so the dbt.ref("...") call has to appear
there. The model returns one row per check made in this run, plus one row with status
"unresolved" for every batch that alerted in an earlier run and has not been accepted. A test
on the model then fails the build while any batch is in either state:

    models:
      - name: orders_quality
        tests:
          - morello_ok          # macro in macros/morello_ok.sql

The first run trains the monitor, which needs 20 or more batches in the upstream model.
To accept a batch, call morello.table.accept_batch with the same state table (see
examples/dbt/README.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..sql import connect
from ..store import TableStore
from ..table import (
    RESULT_COLUMNS,
    TIMESTAMP_FORMAT,
    check_table,
    default_key,
    fit_table,
    result_rows,
    unresolved_alerts,
)

SOURCE_VIEW = "_morello_src"
TRAINED = "trained"  # status of the single row returned by the run that fits the monitor
UNRESOLVED = "unresolved"  # a batch that alerted in an earlier run and is not accepted


def run(
    session: Any,
    relation: Any,
    name: str,
    batch_col: str,
    state_table: str,
    key: str | None = None,
    **fit_options: Any,
) -> pd.DataFrame:
    """Fit on the first run, check on later runs. Returns rows for the results model.

    relation  what dbt.ref(...) returned
    name      label used in the results and, by default, as the name of the saved monitor
    """
    source = connect(session)
    view = _as_view(relation, session)
    key = key or default_key(name)
    store = TableStore(source, state_table)
    if store.get(f"{key}.model") is None:
        monitor = fit_table(source, view, batch_col, store, key=key, **fit_options)
        return _trained_row(name, monitor.last_checked)
    reports = check_table(source, store, key=key, table=view)
    checked = {report.batch for report in reports}
    earlier = [batch for batch in unresolved_alerts(store, key) if batch not in checked]
    return pd.concat([result_rows(name, reports), _unresolved_rows(name, earlier)])


def _as_view(relation: Any, session: Any) -> str:
    """Expose the upstream relation under one name on every platform."""
    for method in ("createOrReplaceTempView", "create_or_replace_temp_view"):  # Spark, Snowpark
        if hasattr(relation, method):
            getattr(relation, method)(SOURCE_VIEW)
            return SOURCE_VIEW
    if hasattr(relation, "create_view"):  # DuckDB relation
        relation.create_view(SOURCE_VIEW, replace=True)
        return SOURCE_VIEW
    session.register(SOURCE_VIEW, relation)  # pandas or Arrow
    return SOURCE_VIEW


def _unresolved_rows(name: str, batches: list[str]) -> pd.DataFrame:
    """One row per batch that alerted in an earlier run and is still not accepted."""
    rows = [_row(name, batch, UNRESOLVED) for batch in batches]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _trained_row(name: str, last_batch: str | None) -> pd.DataFrame:
    return pd.DataFrame([_row(name, last_batch or "", TRAINED)], columns=RESULT_COLUMNS)


def _row(name: str, batch: str, status: str) -> dict[str, Any]:
    """A results row that does not come from a check."""
    row = dict.fromkeys(RESULT_COLUMNS, "")
    row.update(
        checked_at=datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT),
        table_name=name,
        batch=batch,
        status=status,
        rows_checked=0,
        n_alerts=0,
        top_ratio=0.0,
        details="{}",
    )
    return row
