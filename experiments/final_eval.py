"""Frozen comparison under the causal protocol.
usage: final_eval.py <work dir> <false-alarm budget in days> [label]"""
import sys, os, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online2 import run
STD = ["rules", "drift"]
V2 = ["z", "tail10", "digits", "lr6", "lr10", "lrrow"]          # frozen "bits v2"
UP = {b: "up" for b in ["z", "tail10", "lr6", "lr10", "lrrow", "v1"]}
DET = {"standard checks": (STD, None), "PCA reconstruction": (["pca"], None), "domain classifier": (["dc"], None),
       "standard + multivariate": (STD + ["pca", "dc"], None),
       "bits v1": (["v1"], None), "bits v2": (V2, UP), "standard + bits v2": (STD + V2, UP)}
from online2 import ratios, MARGINS


def hits(M, blocks, margin, sided):
    clean, cells = ratios(M, blocks, "runmax", sided)
    return margin, clean.max(axis=1) > margin, {k: r.max(axis=1) > margin for k, r in cells.items()}


work, margin = sys.argv[1], float(sys.argv[2])
from analyze import build
M = pickle.load(open(f"{work}/metrics.pkl", "rb")) if os.path.exists(f"{work}/metrics.pkl") else build(work)
T = len(M["test_idx"])
H = {name: hits(M, blocks, margin, sided) for name, (blocks, sided) in DET.items() if name != "standard + bits v2"}
s1, v2 = H["standard checks"], H["bits v2"]
H["standard + bits v2"] = (margin, s1[1] | v2[1], {k: s1[2][k] | v2[2][k] for k in s1[2]})
print(f"{work}: {T} test days, every metric alerts at {margin:g}x its largest earlier clean deviation")
for name, (mg, cl, det) in H.items():
    print(f"  {name:26s} false-alarm days {int(cl.sum()):2d}/{T}  detection {np.mean([d.mean() for d in det.values()]):.3f}")
df = pd.DataFrame({n: pd.Series({k: d.mean() for k, d in H[n][2].items()}) for n in H})
df.index = pd.MultiIndex.from_tuples(df.index, names=["scenario", "sev"])
cols = ["standard checks", "PCA reconstruction", "domain classifier", "bits v1", "bits v2", "standard + bits v2"]
short = {"standard checks": "standard", "PCA reconstruction": "PCA", "domain classifier": "dom.clf", "bits v1": "bits v1", "bits v2": "bits v2", "standard + bits v2": "std+v2"}
d2 = df[cols].rename(columns=short)
if "--brief" not in sys.argv:
    print(d2.groupby("scenario", sort=False).mean().round(2).to_string())
    print(d2.groupby("sev").mean().round(2).to_string())
    print("join failures by share of rows hit"); print(d2[d2.index.get_level_values(0).str.contains("join")].round(2).to_string())
d2.to_csv(f"{work}/final_online_m{margin:g}.csv")
