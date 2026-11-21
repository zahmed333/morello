# Changelog

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
