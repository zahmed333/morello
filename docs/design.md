# Design

This page explains how Morello is put together and why. The design goals are the three that
*Designing Data-Intensive Applications* (Kleppmann) puts first: reliability, scalability and
maintainability. Each section below says which of the book's ideas was applied and where in
the code.

## What happens to a batch

```
table in a warehouse ──sample──▶ rows ──encode──▶ numeric columns ──score──▶ statistics
       (sql.py)                        (schema.py)              (model.py, checks.py)
                                                                        │
results table ◀──one row per check── report ◀──alert rule── metrics ◀───┘
   (table.py)                       (report.py)  (history.py)
```

1. **Sample.** A query inside the warehouse returns a few thousand random rows of the batch.
2. **Encode.** The schema learned at fit time turns every column into numbers.
3. **Score.** The table model says how many bits each value costs given the rest of its row.
4. **Statistics.** Sums and counts over the rows. These add up across chunks of a batch.
5. **Metrics.** One number per check, computed from the totals.
6. **Alert rule.** Each metric is compared with what it usually reads on clean batches.
7. **Report.** Status, the checks that fired, suspect columns and example rows.

## Modules

Dependencies point one way, from the bottom of this list to the top.

| module | responsibility |
|---|---|
| `term.py` | colour for terminal output |
| `coder.py` | one column's values as classes plus a cost in bits |
| `model.py` | predicts each column from the columns before it; scores rows |
| `schema.py` | turns a table into numeric columns; decides what to skip |
| `checks.py` | what is measured on a batch, and how chunks of a batch combine |
| `history.py` | the alert rule and the record of clean batches |
| `report.py` | the result of one check, as data and as text |
| `state.py` | the saved format |
| `sql.py` | connections and the queries that count and sample batches |
| `store.py` | where saved state lives: folder, object store or table |
| `monitor.py` | ties the above together: fit, check, accept, save, load |
| `distributed.py` | scores a batch on Spark executors |
| `table.py` | runs a monitor against a table on a schedule |
| `cli.py` | command line |
| `integrations/` | thin adapters: Spark, Snowflake, dbt, Airflow, Dagster |

## Reliability

*The system keeps working correctly when things go wrong, including operator mistakes.*

**A run can be repeated.** Scheduled jobs get retried, and a retry must not change the outcome.

- The history keeps one entry per batch key. Checking a batch a second time replaces its entry
  and does not count it twice (`SegmentHistory.record`).
- `check_table` saves nothing until the results are written. A run that dies starts again from
  the same batch, so no batch is skipped.
- The results table is an append-only log. A retried run may append a batch twice. Rows are
  identified by `(table_name, batch, segment)` and readers take the latest `checked_at`. This
  is at-least-once delivery with a deduplication key, chosen over losing a result.
- A backlog is worked through oldest first, a bounded number of batches per run.

**State is never half-written.**

- `LocalStore` writes a temporary file, syncs it and renames it over the old one.
- `TableStore` only appends. Each save is a new version, and a reader takes the highest
  complete one. Nothing is updated in place, so there is no moment when a reader sees a mix.
- Version numbers count up from the highest one present. They do not come from a clock, so
  clock differences between machines cannot reorder saves.

**Parts that belong together are checked.** A saved monitor is a model document and a history
document. Both carry the id of the fit they came from, and loading refuses a history that
belongs to a different model. Both carry a format version, and loading refuses a version it
does not know. See `state.py`.

**Bad data cannot teach the monitor.** A batch that alerts is not added to the history, so one
bad day does not widen the tolerance for the next. A person adds it with `accept` after
looking at it. `update="never"` checks a batch without changing anything, for dry runs.

**Failures are loud.** A missing or mismatched monitor raises `StateError` or
`FileNotFoundError` with what to do. The command line exits 0 for no alert, 1 for an alert and
2 when it could not run. Table and column names are validated before they go into a query.

## Scalability

*The load parameter is rows per batch. Cost should not grow with it.*

**Work is bounded by the sample, not the table.** The warehouse does the scan and returns a
fixed-size random sample. A batch of ten thousand rows and a batch of a billion cost the same
to check once the sample query returns. Day ranges filter on the raw batch column, so the
engine can prune partitions.

**Statistics are additive.** Everything the model checks need is a sum or a count, so chunks of
a batch can be scored on different machines and added in any order (`merge_statistics`). This
is the same property that makes a combiner valid in MapReduce: the merge is commutative and
associative. `distributed.py` uses it to score on executors and return a few kilobytes
per partition. The two measurements that are not additive, medians and the duplicate-row
share, are isolated in `sample_statistics` and computed on the driver's sample.

**State stays small.** The history keeps a fixed window of recent batches plus one running
maximum per check, so it does not grow with the age of the monitor. The model is written once
at fit time and never rewritten by a check.

## Maintainability

**Operability.** The results table is the interface for operators and downstream jobs: one row
per check, with status, suspect columns and the report as JSON. The library logs through the
standard `logging` module and prints nothing; the command line prints reports to stdout and
progress to stderr with `-v`.

**Simplicity.** Each module has one job and the dependency graph has no cycles. The three
steps of measuring a batch (additive statistics, sample statistics, metrics) are separate
functions, so the distributed path and the local path share all of their arithmetic. Tuning
values are named constants at the top of the module that uses them.

**Evolvability.**

- The saved format is JSON with explicit fields per class (`to_state` / `from_state`). It does
  not depend on Python object layout, and loading builds only numbers, arrays and LightGBM
  models, so a saved file cannot run code. Pickle was rejected for both reasons.
- The column schema is decided once at fit time and applied unchanged to every batch. A column
  that disappears from the table becomes nulls and shows up as a null-rate alert. It does not
  silently change what the model sees.
- A new check is one statistic in `BatchChecks.statistics`, one line in `metrics` and one
  `CheckSpec`. The alert rule needs no change.
- A new engine is a `Source` subclass with `query` and `append`.

## Time

Batch keys come from the data (a date column or a batch id), never from when the job ran. A
late or repeated run therefore checks the same batch under the same key. The alert rule is
causal: a batch is compared only with batches recorded before it. Weekdays and weekends get
separate baselines when the key is a date.

## Known limits

- **One writer per monitor.** Stores do not lock. Two jobs checking the same table at the same
  time would each save a history and the later save wins. Schedule one job per table.
- **Sampling is not repeatable.** The sample uses the engine's random function, so two checks
  of the same batch read slightly different rows and can give slightly different numbers.
- **No migration between format versions.** A monitor saved by an older format has to be
  fitted again.
- **The model does not update itself.** When the meaning of the table changes, refit.
- Snowflake, Databricks, Airflow and Dagster adapters have not been run on those platforms.
  See [integrations.md](integrations.md).
