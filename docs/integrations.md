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
| Airflow operator, Dagster asset check | written, **not run inside those schedulers** |
| Native Scala API | not built; Scala jobs gate on the results table (see below) |

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

## Scala

There is no Scala port of the model. A Scala Spark job uses Morello in two steps: a Python step
runs the check and writes the results table, and the Scala job calls
`MorelloGate.requireClean(spark, "ops.morello_results", "main.sales.orders")`
(`examples/scala/MorelloGate.scala`) before it continues.

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
| details | JSON with the alerts and example rows |

## State

A monitor is saved as two objects: `<key>.model` (large, written at fit time) and `<key>.history`
(small, rewritten after each check). The format is compressed JSON. It contains only numbers,
text and LightGBM model text, so loading it cannot run code.
