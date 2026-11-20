"""Run NannyML's two multivariate drift calculators, with their default thresholds, on the
exported holdout batches. One chunk per day. Results are saved after every run (resumable).
usage: run_nannyml.py <data dir> <out json> <pca|dc> <dataset>"""
import sys, os, json, glob, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
import nannyml as nml
data, outp, method, dsname = sys.argv[1:5]
CAT = {"taxi": ["VendorID", "PULocationID", "DOLocationID", "RatecodeID", "store_and_fwd_flag", "payment_type"],
       "ems": ["initial_call_type", "initial_severity", "final_call_type", "final_severity", "borough", "dispatch_area",
               "zipcode", "precinct", "held", "valid_dispatch", "valid_incident", "disposition", "reopen",
               "special_event", "standby", "transfer"]}[dsname]


def prep(path):
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["batch"])
    for c in CAT:
        df[c] = df[c].astype("Int64").astype(str).astype("category")
    return df.drop(columns=["batch"])


ref = prep(f"{data}/reference.parquet")
feats = [c for c in ref.columns if c != "ts"]
if method == "pca":
    calc = nml.DataReconstructionDriftCalculator(column_names=feats, timestamp_column_name="ts", chunk_period="D")
    key = "reconstruction_error"
else:
    ref = ref.sample(30000, random_state=0).sort_values("ts").reset_index(drop=True)   # full reference is too slow here
    calc = nml.DomainClassifierCalculator(feature_column_names=feats, timestamp_column_name="ts", chunk_period="D")
    key = "domain_classifier_auroc"
calc.fit(ref)
res = json.load(open(outp)) if os.path.exists(outp) else {}
files = sorted(glob.glob(f"{data}/*.parquet"))
order = [f for f in files if "clean" in f] + [f for f in files if "join" in f] + \
        [f for s in ("1.0", "0.25", "0.05", "0.01") for f in files if f.endswith(f"__{s}.parquet") and "join" not in f]
if method == "dc":
    order = order[:5]          # clean + the four join severities
for f in order:
    name = os.path.basename(f)[:-8]
    if name in res or name == "reference":
        continue
    out = calc.calculate(prep(f)).filter(period="analysis").to_df()
    alerts = out[(key, "alert")].astype(bool).to_numpy()
    res[name] = {"alert_days": int(alerts.sum()), "days": int(len(alerts)), "values": [float(v) for v in out[(key, "value")]]}
    json.dump(res, open(outp, "w"))
    print(name, res[name]["alert_days"], "/", res[name]["days"], flush=True)
print("DONE")
