# Evidence so far

All failures are simulated. Data: NYC yellow taxi trips and NYC EMS dispatch records (real,
public) and Synthea lab records (synthetic). Code for the experiments is in `experiments/`.

## 1. Controlled experiment on an untouched month (May 2024)

Design and alert margin were fixed on April data. Ten failure types x four severities
(1%, 5%, 25%, 100% of rows), every day of May. Share of failure cells detected, with
false-alarm days on the clean copies in brackets.

| data | standard per-column checks | this detector alone | both (what this package runs) |
|---|---|---|---|
| taxi | 0.51 (2/31) | 0.64 (3/31) | 0.77 (3/31) |
| EMS | 0.69 (2/31) | 0.58 (2/31) | 0.78 (3/31) |
| lab records (development data only) | 0.47 (2/62) | 0.30 (2/62) | 0.48 (2/62) |

Records joined to the wrong record: taxi 1.00 at every severity vs 0.06 for standard checks;
EMS 0.06 / 0.19 / 0.65 / 1.00 by severity vs 0.06.

## 2. Other multivariate methods on the same May batches

| method | taxi | EMS |
|---|---|---|
| NannyML 0.13.1 PCA reconstruction, default threshold | 0.19 detected, 1/31 false-alarm days | 0.34 detected, 7/31 false-alarm days |
| domain classifier (own implementation) | 0.16 | 0.00 |

NannyML's PCA method caught wrong-record joins on taxi only when every row was affected
(0.03 / 0.03 / 0.06 / 1.00 by severity). On EMS its alerts on corrupted data were almost
all on the same days it alerted on clean data. NannyML's own domain classifier and commercial
products (Anomalo, Monte Carlo and others) were not run.

## 3. This package on raw files

`Monitor().fit(...)` on February-April with the original columns (timestamps, text codes),
then every day of May checked clean and with eight injected failures.

| | taxi | EMS |
|---|---|---|
| columns modelled | 20 | 28 |
| fit time, 2 cores | 217 s | 220 s |
| clean May days that alerted | 4/31 | 3/31 |
| injected failures detected | 8 of 8 types on 31/31 days | 8 of 8 types on 31/31 days |

Every clean-day alert traced to a visible change in the source data (a surcharge change on
May 1, a flag change on May 20, Memorial Day, an extreme trip distance, a shift in EMS severity
codes on May 13). The injected failures here are mostly severe (25-100% of rows), and the EMS
wrong-record test is easier than in section 1 because timestamps moved with the mismatched
record.

## 4. Table runner on 13.8 million rows

`fit_table` and `check_table` on DuckDB over the four raw taxi files (Feb-May 2024, 13.8M rows),
sampling inside the engine, with a separate alert history per vendor (`slice_col="VendorID"`).

- Fit on Feb-Apr: about 6 minutes on 2 cores. Saved state: 17 MB model, 0.2 MB history.
- May, checked day by day with each alert reviewed and accepted: 3 of 31 days alerted on all rows
  (May 1 surcharge change, May 20 flag change, May 27 Memorial Day), plus 6 slice-level alerts.
- Without accepting, the May 1 surcharge change alerted every following day, which is the intended
  behaviour for a lasting change nobody has acknowledged.
- Corrupted copies of May 15: a wrong-record join on 1% and on 5% of rows both alerted. A unit
  change applied only to vendor 1 alerted for all rows and for vendor 1, and vendor 2 stayed quiet.

The Spark adapter (SQL sampling, state in a table, scoring on executors) and the dbt model were
tested on small synthetic tables only. Snowflake, Databricks, Airflow and Dagster adapters have
not been run. See `INTEGRATIONS.md`.

## Not yet shown

- Performance on real incidents.
- Tables with few rows per batch, or rows clustered by entity (lab records were weak).
- Comparison with commercial tools.
