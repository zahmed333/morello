"""Second, cheap pass: code length of every value under a no-context model (its plain frequency
in the training data). Together with the main dump this gives, per row and column, a log
likelihood ratio: how much less likely the value is given the rest of the row than on its own.
usage: dump0.py <dataset module> <raw dir> <work dir> <dev|holdout>
"""
import sys, os, pickle, time, importlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from model3 import P_FLOOR, ESCAPE_BITS
ds = importlib.import_module(sys.argv[1]); raw_dir, work, phase = sys.argv[2], sys.argv[3], sys.argv[4]
COLS = ds.COLS; C = len(COLS); ci = {c: i for i, c in enumerate(COLS)}
t0 = time.time()
splits = ds.load(raw_dir, phase); train = splits["train"]
m = pickle.load(open(f"{work}/model.pkl", "rb")); D = pickle.load(open(f"{work}/dump.pkl", "rb"))
L0 = []
for j in range(C):
    coder = m.coders[j]
    idx = coder.remap[coder.transform(m._target(train, j))[0]]
    p = np.bincount(idx[idx >= 0], minlength=coder.n_present) + 0.5
    L0.append(-np.log2(p / p.sum()))


def b0(df):
    out = np.zeros((len(df), C), np.float32)
    for j in range(C):
        coder = m.coders[j]
        cls, extra = coder.transform(m._target(df, j))
        idx = coder.remap[cls]
        s0 = np.full(len(df), -np.log2(P_FLOOR) + ESCAPE_BITS)
        s0[idx >= 0] = L0[j][idx[idx >= 0]]
        out[:, j] = s0 + extra
    return out


h16 = lambda a: np.clip(a, -6e4, 6e4).astype(np.float16)
frames = {}
for k in ("ref", "calib", "test"):
    for d, g in splits[k].groupby("batch"):
        frames[d] = g.sample(min(ds.BATCH, len(g)), random_state=2).reset_index(drop=True)
clean = {d: h16(b0(frames[d])) for d in D["days"]}
scen = {}
for name, spec in ds.scenarios(ci).items():
    scen[name] = {}
    for d, e in D["scen"][name]["days"].items():
        df = frames[d]; n = len(df); o = {}
        if spec[0] == "rowwise":
            o["full"] = h16(b0(spec[1](df)[0]))
        elif spec[0] == "join":
            for f in D["sev"]:
                idx = e["perm"][: min(max(int(round(f * n)), 2), n)]
                dff = df.copy(); dff.loc[idx, spec[1]] = df.loc[np.roll(idx, 1), spec[1]].values
                o[("join", f)] = h16(b0(dff.iloc[idx]))
        scen[name][d] = o
pickle.dump({"clean": clean, "scen": scen}, open(f"{work}/dump0.pkl", "wb"), protocol=4)
print("DUMP0 DONE", round(time.time() - t0), "s")
