# Experiments behind RESULTS.md

These scripts are research code. They are kept so the numbers can be reproduced; they are not
part of the installed package.

## Data (public, not stored in this repo)

- NYC yellow taxi trips, Jan-May 2024: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-MM.parquet`
  saved as `yellow_2024-MM.parquet`
- NYC EMS incident dispatch data (NYC Open Data, dataset `76xm-jjuj`), Jan-May 2024, one CSV per
  month saved as `ems_2024-MM.csv`
- Synthea 10k synthetic patients (COVID-19 sample), CSV export

## Controlled experiment (sections 1 and 2 of RESULTS.md)

    python3 dump.py  ds_taxi <raw dir> <work dir> holdout   # scores clean and corrupted days, resumable
    python3 dump0.py ds_taxi <raw dir> <work dir> holdout   # no-context code lengths
    python3 final_eval.py <work dir> 3                      # causal protocol, alert margin 3

`ds_ems.py` and `ds_labs.py` define the other two datasets. `results/` holds the per-failure
detection tables. `competitors/` exports the same batches and runs NannyML on them
(needs Python 3.11 or 3.12).

## Package test on raw files (section 3)

    pip install .
    MORELLO_DATA=<dir with taxi/ and ems/> python3 experiments/package_test/real_test.py taxi out/

## Protocol in brief

Each check is compared with its rolling median over the previous 28 clean days and alerts when
the change exceeds 3x the largest change on any earlier clean day. Design and margin were fixed
on April data; May was used once. Ten failure types x four severities are injected into every
test day.
