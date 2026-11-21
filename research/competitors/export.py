"""Write the exact clean and corrupted batches used in the holdout experiment to parquet,
so an external library can be run on identical data.
usage: export.py <dataset module> <raw dir> <work dir> <out dir>"""
import sys, os, pickle, importlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ds = importlib.import_module(sys.argv[1]); raw, work, out = sys.argv[2:5]
os.makedirs(out, exist_ok=True)
COLS = ds.COLS; ci = {c: i for i, c in enumerate(COLS)}
splits = ds.load(raw, "holdout"); D = pickle.load(open(f"{work}/dump.pkl", "rb"))
frames = {}
for k in ("ref", "calib", "test"):
    for d, g in splits[k].groupby("batch"):
        frames[d] = g.sample(min(ds.BATCH, len(g)), random_state=2).reset_index(drop=True)
days = D["days"]; test_days = [d for d in days if D["where"][d] == "test"]
keep_cols = ["batch"] + COLS
pd.concat([frames[d][keep_cols] for d in days if D["where"][d] != "test"]).to_parquet(f"{out}/reference.parquet", index=False)
pd.concat([frames[d][keep_cols] for d in test_days]).to_parquet(f"{out}/clean.parquet", index=False)
for name, spec in ds.scenarios(ci).items():
    for f in D["sev"]:
        parts = []
        for d in test_days:
            df = frames[d]; e = D["scen"][name]["days"][d]; n = len(df)
            idx = e["perm"][: min(max(int(round(f * n)), 2), n)]
            if spec[0] == "rowwise":
                cor, touched = spec[1](df); dff = df.copy(); dff.loc[idx, touched] = cor.loc[idx, touched].values
            elif spec[0] == "join":
                dff = df.copy(); dff.loc[idx, spec[1]] = df.loc[np.roll(idx, 1), spec[1]].values
            elif spec[0] == "dup":
                dff = df.copy(); dff.loc[idx, COLS] = df.loc[e[("src", f)], COLS].values
            else:
                dff = df.iloc[e[("keep", f)]]
            parts.append(dff[keep_cols])
        pd.concat(parts).to_parquet(f"{out}/{name}__{f}.parquet", index=False)
print("exported", len(os.listdir(out)), "files")
