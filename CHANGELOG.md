# Changelog

## 0.7.0

Added
- `morello backtest` (`backtest`, `backtest_table`): fit on the first part of a history and
  replay the rest, to see which past batches would have alerted before anything is scheduled.
- `morello init` (`profile_frame`, `profile_table`): which column defines the batches, whether
  there is enough history, which columns are modelled, and the command to run next.
- One finding per problem. A report groups the checks that fired into numbered findings, each
  a sentence such as "total stopped fitting unit_price and quantity in 4.2% of rows (usual
  0%)". Example rows show the value the model expected. `--details` lists every check.
- Where the affected rows are: a finding names the value of another column that most of its
  rows share, when there is one.
- Columns that stopped fitting because another column changed are listed under that column.
- `refit_table` and `morello refit-table` train again with the saved settings and leave out
  batches with an open alert. `fit_table(refit_every=N)` makes `check_table` do it after N
  checked batches.
- New columns in a batch and columns missing from it are named in the report.
- Identifier and free-text columns are no longer skipped. Each contributes its length and,
  for text, whether its format was seen in training.
- `check_reference` and `morello check-reference`: the share of rows whose key has no match in
  another table, with the same alert rule and no saved state.
- `morello check-table --slack-webhook ... --email-to ...` sends the findings and the accept
  command to people. A failed notification never changes the result of a check.
- `morello page` writes a status page for a table as one HTML file.
- `sample_rows="auto"` sizes the sample from the table's batches, between 5,000 and 50,000
  rows. `Monitor.smallest_detectable_share` states the share of rows a fault has to touch.

Changed
- The saved format is now 3. Formats 2 and 3 are read.
- Check numbers are unchanged on tables without identifier or free-text columns (verified on
  fixed inputs). On tables that have them, the model has extra features, so numbers differ
  from 0.6. The benchmark in `docs/results.md` was run on 0.5 and has not been re-run.
- Example rows no longer repeat the batch column.
- Table tests pin the engine's random seed. Sampling in SQL is random, and about 1 clean
  replay in 11 raised a false alarm somewhere in 30 batches, which made the tests flaky.

## 0.6.0

Added
- An alert stays open until it is dealt with. `check_table` records every batch whose latest
  check alerted; `unresolved_alerts` returns them. `morello check-table`, the Airflow operator,
  the Dagster asset check and the dbt test fail while a batch is unresolved, so a task that is
  retried after an alert fails again. Before, the retry found nothing new to check and passed.
- A batch leaves the list when it is accepted or passes a re-check
  (`morello check-table --batch ...`).
- `accept_batch(results_table=...)` and `morello accept --results-table` append a row with
  status `accepted`.
- `morello.integrations.spark.unresolved`, and `fail_on_alert(reports, unresolved)`.
- Tests that run the operator inside Airflow and the asset check inside Dagster when those are
  installed, and a test for the dbt adapter.

Changed
- The Airflow operator imports from `airflow.sdk` first (Airflow 3) and falls back to
  `airflow.models` (Airflow 2).
- The Scala gate and the dbt `morello_ok` test read the latest row of every batch and segment.
  The Scala gate used to read only the newest run. `MorelloGate` has a `main` for spark-submit.
- On Spark, the check for whether a table exists asks the catalog and no longer logs an error.

Verified
- Airflow 3.4.0 and Dagster 1.13.31 against a DuckDB table; the Scala gate compiled with Scala
  2.13.18 and run with spark-submit on Spark 4.3.0.

## 0.5.0

A restructuring for readability and safe re-runs. Check results are unchanged: the same inputs
give the same numbers as 0.4.0.

Changed
- The package is split into one module per job: `coder`, `model`, `schema`, `checks`,
  `history`, `report`, `state`, `store`, `monitor`, `distributed`, `sql`, `table`, `cli`.
  `docs/design.md` describes them.
- Saved format 2. Every class lists the fields it saves, files are valid JSON, and the model
  and its history carry a shared id. Monitors saved by 0.4.0 or earlier must be fitted again.
- Saved files end in `.json.gz` (was `.tbz`).
- `Report` and `Alert` are dataclasses. `Report.n_rows` is now `rows_checked`, `suspects` is
  `suspect_columns`, and alerts are `Alert` objects, not dicts.
- `Monitor.table_cfg` (a dict) is `Monitor.table_config` (a `TableConfig`); `Monitor.cols` is
  `Monitor.columns`; `Monitor.cursor` is `Monitor.last_checked`.
- `check_table` works through a backlog oldest first. It used to take the newest batches and
  skip the rest.
- `TableStore` versions count up from 1. They used to be wall-clock milliseconds.
- The command line exits with 2 when it could not run, and prints the reason.
- The library logs through `logging` and no longer prints. The `verbose` arguments are gone;
  use `morello -v`.

Added
- Checking the same batch twice learns it once.
- Table and column names are validated before they are put into a query.
- `StateError` for saved files that cannot be read or do not belong together.
- Tests for state, stores, schema, query building, the command line and repeated runs.
- `ruff` settings and a CI workflow.

Fixed
- A DuckDB connection was not recognised and went through the generic DB-API path.

## 0.4.0

- Coloured terminal reports, `--color`, `NO_COLOR`.
- README with a real sample report; landing page; slide template.

## 0.3.0

- Renamed to Morello. `morello` console command.

## 0.2.0

- Table runner with in-warehouse sampling, table and cloud stores, results table, slices.
- Adapters for Spark, Snowflake, dbt, Airflow and Dagster.

## 0.1.0

- First version: `Monitor.fit` and `Monitor.check` on DataFrames.
