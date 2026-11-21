# Using Morello in a data stack

One engine, several thin adapters. In every case the sample is taken by a SQL query inside the
warehouse, the monitor's state lives in a table or in cloud storage, and results are rows in a
table that any other tool can read.

## What has been run, and what has not

| piece | status |
|---|---|
| In-warehouse sampling, table runner, results table, per-slice checks | tested on DuckDB (unit test, and 13.8M raw taxi rows) |
| State in a warehouse table, state in a folder | tested (DuckDB, Spark) |
| Spark adapter, including scoring on executors | tested on a local Spark session |
| dbt Python model and `morello_ok` test | tested with dbt-duckdb |
| Command line `fit-table`, `check-table`, `accept` | tested on DuckDB |
| Cloud storage state (`s3://`, `abfss://`, `gs://` through fsspec) | written, not run |
| Snowflake stored procedure, Snowflake SQL dialect | written, **not run against Snowflake** |
| Databricks (Unity Catalog tables, DBFS, workflows) | same code as the Spark test, **not run on Databricks** |
| dbt on Snowflake / Databricks / BigQuery | same model file, **not run on those platforms** |
| Airflow operator | tested inside Airflow 3.4.0 (`tests/test_schedulers.py`) |
| Dagster asset check | tested inside Dagster 1.13.31 (`tests/test_schedulers.py`) |
| Scala gate on the results table | compiled with Scala 2.13.18, run with spark-submit on Spark 4.3.0 |
| Native Scala API for the model | not built; scoring runs in Python (see below) |

## Any SQL engine (Python)

```python
from morello.table import fit_table, check_table, accept_batch

fit_table(
    conn, "analytics.orders", batch_col="order_ts", store="s3://bucket/morello", slice_col="country"
)  # once, and again every few months
reports = check_table(
    conn, "s3://bucket/morello", table="analytics.orders", results_table="ops.morello_results"
)  # after every load
```

`conn` can be a DuckDB connection, a SparkSession, a Snowpark session, a SQLAlchemy engine or URL,
or a DB-API connection (pass `dialect="snowflake"` or `"databricks"` for those). `store` can be a
folder, a cloud URL, or `TableStore(source, "ops.morello_state")`.

`check_table` checks every batch that arrived since the last check. A batch that alerts is not
learned from. After someone has looked at it, `accept_batch(conn, store, "2024-05-01", table=...)`
tells the monitor it was fine. A real, lasting change in the source keeps alerting until accepted.

Batches are calendar days of `batch_col` by default. Pass `as_date=False` to use the column's
distinct values (a load id, an hour bucket you computed, a file name).

## Command line

    morello fit-table   --url snowflake://... --table analytics.orders --batch-col order_ts --store s3://bucket/morello
    morello check-table --url snowflake://... --table analytics.orders --store s3://bucket/morello --results-table ops.morello_results
    morello accept      --url snowflake://... --table analytics.orders --store s3://bucket/morello --batch 2024-05-01

`check-table` exits with code 1 on an alert, which is enough for cron, GitHub Actions, Airflow's
BashOperator and most schedulers.

## Alerts stay open until someone deals with them

A scheduler retries failed tasks. On the retry there is nothing new to check, and a gate that
only looked at the current run would pass. Morello therefore records every batch whose latest
check alerted, and every gate fails while that list is not empty:

- `morello check-table` exits with 1 and names the batch.
- The Airflow operator raises, and the Dagster asset check does not pass.
- `morello.table.unresolved_alerts(store, table=...)` returns the batches, for your own gate.
- The dbt model returns a row with status `unresolved`, and the Scala gate reads the latest row
  of every batch in the results table.

A batch leaves the list in one of two ways. A person accepts it:

    morello accept --url ... --table analytics.orders --store ... --batch 2024-05-01 --results-table ops.morello_results

Or the data is repaired and the batch passes a re-check:

    morello check-table --url ... --table analytics.orders --store ... --batch 2024-05-01

## Around the scheduled check

```python
from morello.backtest import backtest_table
from morello.table import fit_table, refit_table
from morello.references import check_reference
from morello.notify import notify
from morello.page import write_page

# Before scheduling: which past batches would have alerted? Nothing is saved.
print(backtest_table(connection, "analytics.orders", "order_ts").text())

# Fit again every 30 checked batches, or on request. Open alerts are left out of training.
fit_table(connection, "analytics.orders", "order_ts", store, refit_every=30)
refit_table(connection, store, table="analytics.orders")

# Keys that have no match in another table. Returns a Report like any other check.
report = check_reference(
    connection, "analytics.orders", "customer_id", "analytics.customers", "id", "order_ts"
)

# Tell people. Returns the channels that were reached; a failure is logged, not raised.
notify("analytics.orders", reports, unresolved, slack_webhook=url, email_to=["oncall@x.org"])

# One HTML file with open alerts, the status of each batch and the findings.
write_page(connection, store, "ops.morello_results", "orders.html", table="analytics.orders")
```

Email is sent through the SMTP server named by `MORELLO_SMTP_HOST`, `MORELLO_SMTP_PORT`,
`MORELLO_SMTP_USER` and `MORELLO_SMTP_PASSWORD`, from `MORELLO_EMAIL_FROM`. The Slack message
was tested against a local HTTP server and the email against a stand-in for the SMTP client.
Neither has been sent through Slack or a real mail server.

## Spark and Databricks

```python
from morello.integrations import spark as morello_spark

morello_spark.fit(spark, "main.sales.orders", "order_ts", state_table="main.ops.morello_state")
reports = morello_spark.check(
    spark,
    "main.sales.orders",
    state_table="main.ops.morello_state",
    results_table="main.ops.morello_results",
)
morello_spark.fail_on_alert(reports)  # raises, which fails the job or workflow task
```

`fit_dataframe` and `check_dataframe` do the same for a DataFrame that is not a table.

Large samples: `morello_spark.fit(..., score_rows=1_000_000)` scores a million rows per batch on the
executors and sends back only sums and counts. Use the same `score_rows` for fitting and checking.

## Snowflake

`morello/integrations/snowflake.py` contains two stored-procedure handlers and prints the
`CREATE PROCEDURE` statements. Data stays inside the account; state and results are tables.
A Snowflake task schedules the check. See the module docstring for the four setup steps.

## dbt

`examples/dbt/` is a complete project: a Python model that checks an upstream model and a generic
test that fails the build on an alert.

```python
# models/orders_quality.py
from morello.integrations.dbt import run


def model(dbt, session):
    dbt.config(materialized="table")
    return run(
        session,
        dbt.ref("orders"),
        name="orders",
        batch_col="order_ts",
        state_table=f"{dbt.this.schema}.morello_state",
    )
```

```yaml
models:
  - name: orders_quality
    tests:
      - morello_ok
```

## Airflow and Dagster

`morello.integrations.airflow.MorelloCheckOperator` and
`morello.integrations.dagster.morello_check_result` wrap `check_table`. Both fail the task or
the asset check on an alert and attach the report.

```python
from morello.integrations.airflow import MorelloCheckOperator

MorelloCheckOperator(
    task_id="orders_quality",
    connect=lambda: my_connection(),  # called on the worker
    table="analytics.orders",
    store="s3://bucket/morello",
    results_table="ops.morello_results",
)
```

```python
from dagster import asset_check

from morello.integrations.dagster import morello_check_result


@asset_check(asset=orders)
def orders_quality():
    return morello_check_result(
        my_connection(), table="analytics.orders", store="s3://bucket/morello"
    )
```

Both were run inside the scheduler against a DuckDB table: a DAG run with `dag.test()` on
Airflow 3.4.0, and `materialize` with the asset check on Dagster 1.13.31. In each, two clean
days passed, a day with a broken currency failed and named the column, a repeated run with
nothing new failed again, and the run after `accept` passed. The operator imports from
`airflow.sdk` and falls back to `airflow.models`, so it should load on Airflow 2, where it has
not been run. Neither adapter has been run against a deployed scheduler with remote workers.

## Scala

There is no Scala port of the model. A Scala Spark job uses Morello in two steps: a Python step
runs the check and writes the results table, and the Scala job calls
`MorelloGate.requireClean(spark, "ops.morello_results", "main.sales.orders")`
(`examples/scala/MorelloGate.scala`) before it continues. The same file can run as a step of
its own with `spark-submit --class MorelloGate`, which exits with a non-zero code on an alert.

The gate was compiled with Scala 2.13.18 and run with spark-submit on Spark 4.3.0 against a
results table written by the Python check through a shared Hive metastore. It exited with 0
after a clean day, with 1 after a broken day and again after a repeated check, and with 0
after the batch was accepted. `examples/scala/README.md` has the commands.

## Very large tables

- Cost does not grow with table size: each check reads one sample of `sample_rows` (default 5,000).
- Failures that touch a small share of rows need a larger sample. On Spark use `score_rows`.
- Failures confined to one country, customer or source need `slice_col`, which keeps a separate
  alert history for each large value of that column and samples each one separately.
- The batch-listing query groups by day over the rows since the last check. Partition or cluster
  the table on the batch column so that query and the sample query prune.

## Results table

| column | meaning |
|---|---|
| checked_at | UTC time of the check |
| table_name, batch, segment | what was checked; segment is empty for all rows or `column=value` for a slice |
| status | ok, alert, or warming_up |
| rows_checked, n_alerts | sample size and number of checks that fired |
| suspect_columns | columns named by the alerts, most suspicious first |
| top_check, top_column, top_ratio | the strongest alert |
| details | JSON with the findings (one sentence per problem), the alerts and example rows |

## State

A monitor is saved as two objects: `<key>.model` (large, written at fit time) and `<key>.history`
(small, rewritten after each check). The format is compressed JSON. It contains only numbers,
text and LightGBM model text, so loading it cannot run code.
