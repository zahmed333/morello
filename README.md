<p><img src="assets/brand/banner.svg" alt="Morello. Know when your data stops making sense." width="100%"></p>

# Morello

Flags batches of a table whose rows stop making sense together: records joined to the wrong
record, swapped columns, changed units, remapped codes, lost precision. It also runs basic
per-column checks (nulls, unexpected values, distribution shift, duplicates).

It needs no rules. You give it a history of the table that you believe is clean, and then each
new batch.

## Install

    pip install .            # from this folder; needs Python 3.9+, numpy, pandas, lightgbm, pyarrow

## Use from Python

```python
import pandas as pd
from morello import Monitor

history = pd.read_parquet("orders_history.parquet")      # at least 20 batches; about 50 for alerts to be on from day one
mon = Monitor().fit(history, batch_col="order_date")      # daily batches by default
mon.save("orders.monitor")

report = mon.check(pd.read_parquet("orders_today.parquet"))
print(report.text())
if report.alert:
    ...                       # stop the pipeline, page someone
    # mon.accept(report)      # if a person decides the batch was fine
mon.save("orders.monitor")    # history grows with every clean batch
```

## Use on a table in a warehouse

```python
from morello.table import fit_table, check_table

fit_table(conn, "analytics.orders", batch_col="order_ts", store="s3://bucket/morello")
reports = check_table(conn, "s3://bucket/morello", table="analytics.orders",
                      results_table="ops.morello_results")
```

The sample is taken by a query inside the warehouse, so table size does not matter. `conn` can be
DuckDB, Spark, Snowpark, a SQLAlchemy URL or a DB-API connection. Adapters for Spark and Databricks,
Snowflake, dbt, Airflow, Dagster and Scala jobs are described in `INTEGRATIONS.md`, together with
which of them have actually been run.

## Use from the command line

    python -m morello fit   orders_history.parquet --batch-col order_date --model orders.monitor
    python -m morello check orders_today.parquet   --model orders.monitor --json report.json

After `pip install .` the same commands are available as `morello fit`, `morello check` and so on.

`check` exits with code 1 when a batch alerts, so it can gate a pipeline step. CSV files work too.

## What a report looks like

    batch 2024-03-01: ALERT (3000 rows checked, 30 batches of history)
      suspect columns: total
      - surprise | total: 68.03 (usual -3.294), 802.1x the largest change seen before
      - broken relationship | total: 0.03 (usual 0), 10.0x the largest change seen before
      example rows for total:
        total=40.2, country=US, product=P16, category=garden, unit_price=32.95, quantity=3

## How it works

1. **Columns.** Numbers, booleans, text categories and timestamps are handled automatically.
   Timestamps become hour-of-day plus the gap to the first timestamp column. Identifier and
   free-text columns are skipped and listed in `mon.schema.skipped`.
2. **Model.** Each column is predicted from the columns before it (gradient-boosted trees). The
   model gives every value a probability given the rest of its row.
3. **Checks per batch.**
   - *surprise*: how much less predictable a column was than the model expected.
   - *broken relationship*: share of values far less likely given their row than on their own.
   - *extreme values*: share of values the model considered nearly impossible.
   - *digit pattern*: change in the last digits of numeric columns (rounding, unit changes).
   - per-column null rate, unexpected values, distribution shift, mean, spread, duplicates.
4. **Alert rule.** Each check is compared with its median over the last 28 batches (weekdays and
   weekends separately). It alerts when the change is more than `margin` (default 3) times the
   largest change seen on any earlier clean batch. Clean batches are added to the history
   automatically; alerting batches are added only if you call `accept`.

`fit` trains the model on the earliest half of the history and uses the later half to learn how
much each check normally moves. Alerts switch on after 14 calibrated batches
(`mon.batches_until_ready`).

## Settings

| argument | default | meaning |
|---|---|---|
| `margin` | 3.0 | higher means fewer alerts |
| `window` | 28 | batches that define "usual" |
| `sample_rows` | 5000 | rows scored per batch |
| `max_train_rows` | 150000 | rows used to train the model |

## Limits

- Needs volume. With a few hundred rows per batch, or rows that come in clusters from few
  entities, normal variation hides most problems.
- Does not catch duplicated rows through the model (a separate duplicate check is included)
  and is weak on a missing slice of the data.
- Real changes in the source (a holiday, a fee change) alert too. Expect a few alerts a month
  on a busy table; `accept` teaches the monitor the new normal.
- The model should be refitted when the table's meaning changes, or every few months.
- Saved monitors are compressed JSON (no pickle), split into a model part and a small history part.
- Evidence so far is from simulated failures on three public datasets. See `RESULTS.md`.

## Files

    morello/schema.py    column handling
    morello/model.py     the predictive model
    morello/monitor.py   checks, alert rule, reports
    morello/sql.py       sampling inside the warehouse
    morello/table.py     fit and check a table; results rows
    morello/state.py     saved format      morello/store.py   folder, cloud or table storage
    morello/integrations/  spark, snowflake, dbt, airflow, dagster
    examples/dbt/          dbt project       examples/scala/      Scala gate on the results table
    assets/brand/          logo, colours, type (see BRAND.md)
    examples/demo.py       synthetic orders table with a broken join
    tests/test_smoke.py    quick end-to-end test
