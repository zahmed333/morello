"""End-to-end test of the packaged tool on raw files with their original columns.
Fit on Feb-Apr 2024, then walk through May day by day: check the clean day, and check corrupted
copies of it. Resumable. usage: real_test.py <taxi|ems> <out dir>   (install the package first: pip install .)"""
import sys, os, json, glob, pickle, time
import numpy as np, pandas as pd
from morello import Monitor

name, out = sys.argv[1], sys.argv[2]
DATA = os.environ.get("MORELLO_DATA", "data")       # expects DATA/taxi/*.parquet and DATA/ems/*.csv
os.makedirs(out, exist_ok=True)
t0 = time.time()
if name == "taxi":
    TS = "tpep_pickup_datetime"
    def month(mo):
        df = pd.read_parquet(f"{DATA}/taxi/yellow_2024-{mo:02d}.parquet")
        df = df[(df[TS].dt.year == 2024) & (df[TS].dt.month == mo)]
        return pd.concat([g.sample(min(10_000, len(g)), random_state=1) for _, g in df.groupby(df[TS].dt.date)])
    GROUP = ["payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
             "improvement_surcharge", "congestion_surcharge", "Airport_fee", "total_amount"]
    def unit(d): d = d.copy(); d["trip_distance"] = (d["trip_distance"] * 1.609).round(2); return d
    def swap(d): d = d.copy(); d["PULocationID"], d["DOLocationID"] = d["DOLocationID"].values, d["PULocationID"].values; return d
    def enum(d): d = d.copy(); d["payment_type"] = d["payment_type"].map(lambda v: {1: 2, 2: 1}.get(v, v)); return d
    def cast(d): d = d.copy(); d["trip_distance"] = d["trip_distance"].round(0); return d
else:
    TS = "incident_datetime"
    def month(mo):
        return pd.read_csv(f"{DATA}/ems/ems_2024-{mo:02d}.csv", low_memory=False)
    GROUP = ["held_indicator", "valid_dispatch_rspns_time_indc", "dispatch_response_seconds_qy", "first_on_scene_datetime",
             "valid_incident_rspns_time_indc", "incident_response_seconds_qy", "incident_travel_tm_seconds_qy",
             "first_to_hosp_datetime", "first_hosp_arrival_datetime", "incident_close_datetime",
             "incident_disposition_code", "reopen_indicator", "special_event_indicator", "standby_indicator", "transfer_indicator"]
    def unit(d): d = d.copy(); d["incident_travel_tm_seconds_qy"] = (d["incident_travel_tm_seconds_qy"] / 60).round(2); return d
    def swap(d): d = d.copy(); d["dispatch_response_seconds_qy"], d["incident_travel_tm_seconds_qy"] = d["incident_travel_tm_seconds_qy"].values, d["dispatch_response_seconds_qy"].values; return d
    def enum(d): d = d.copy(); d["borough"] = d["borough"].map(lambda v: {"BROOKLYN": "MANHATTAN", "MANHATTAN": "BROOKLYN"}.get(v, v)); return d
    def cast(d): d = d.copy(); d["incident_travel_tm_seconds_qy"] = (d["incident_travel_tm_seconds_qy"] / 60).round(0) * 60; return d


def partial(df, fn, f, rng):
    cor = fn(df); idx = df.index[rng.choice(len(df), max(2, int(f * len(df))), replace=False)]
    changed = [c for c in df.columns if not cor[c].equals(df[c])]
    d = df.copy(); d.loc[idx, changed] = cor.loc[idx, changed].values
    return d


def join(df, f, rng):
    idx = df.index[rng.choice(len(df), max(2, int(f * len(df))), replace=False)]
    d = df.copy(); d.loc[idx, GROUP] = df.loc[np.roll(idx, 1), GROUP].values
    return d


def dup(df, f, rng):
    k = max(2, int(f * len(df))); p = rng.permutation(len(df))
    return pd.concat([df.iloc[p[k:]], df.iloc[rng.choice(p[k:], k)]], ignore_index=True)


VARIANTS = {"broken join 1%": lambda d, r: join(d, 0.01, r), "broken join 5%": lambda d, r: join(d, 0.05, r),
            "broken join 25%": lambda d, r: join(d, 0.25, r), "unit change 25%": lambda d, r: partial(d, unit, 0.25, r),
            "swapped columns 25%": lambda d, r: partial(d, swap, 0.25, r), "category remap 100%": lambda d, r: enum(d),
            "lost precision 100%": lambda d, r: cast(d), "duplicate rows 5%": lambda d, r: dup(d, 0.05, r)}

mpath, rpath = f"{out}/monitor.pkl", f"{out}/results.json"
if os.path.exists(mpath):
    mon = Monitor.load(mpath); res = json.load(open(rpath))
else:
    hist = pd.concat([month(m) for m in (2, 3, 4)], ignore_index=True)
    mon = Monitor().fit(hist, batch_col=TS)
    res = {"fit_seconds": round(time.time() - t0), "columns": mon.cols, "skipped": mon.schema.skipped,
           "history": len(mon.history), "days": {}, "sample_report": None}
    mon.save(mpath); json.dump(res, open(rpath, "w"), default=str)
    print("fit done", res["fit_seconds"], "s;", len(mon.cols), "columns; skipped", mon.schema.skipped, flush=True)
may = month(5)
days = may[TS].astype(str).str[:10] if name == "ems" else may[TS].dt.strftime("%Y-%m-%d")
for d in sorted(days.unique()):
    if d in res["days"] or not d.startswith("2024-05"):
        continue
    df = may[days == d]
    if len(df) > 5000:
        df = df.sample(5000, random_state=2)
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(int(d.replace("-", "")))
    row = {}
    for vname, fn in VARIANTS.items():
        rep = mon.check(fn(df, rng), batch=d, update="never")
        row[vname] = dict(alert=rep.alert, suspects=rep.suspects[:3])
        if vname == "broken join 5%" and rep.alert and res["sample_report"] is None:
            res["sample_report"] = rep.text()
    rep = mon.check(df, batch=d, update="auto")
    row["clean"] = dict(alert=rep.alert, top=[(a["check"], a["column"], round(a["ratio"], 1)) for a in rep.alerts[:3]])
    if rep.alert:
        mon.accept(rep)                 # a person reviewed it and said it was fine
    res["days"][d] = row
    mon.save(mpath); json.dump(res, open(rpath, "w"), default=str)
    print(d, "clean alert" if rep.alert else "clean ok", {k: int(v["alert"]) for k, v in row.items() if k != "clean"}, round(time.time() - t0), "s", flush=True)
print("TEST DONE")
