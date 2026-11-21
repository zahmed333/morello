"""Causal protocol with options for how alert bars are scaled and how metrics are combined.
scale:   runmax = largest earlier clean deviation | mad = robust spread of earlier clean deviations
sided:   two = |deviation| | up = only increases count (for surprise statistics)
combine: max = any metric over its bar | top3 = root-sum-square of the three largest ratios
"""
import sys, os, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import reference, build

MARGINS = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50]


def ratios(M, blocks, scale="runmax", sided=None):
    Mc = np.concatenate([M["clean"][b] for b in blocks], axis=1)
    up = np.concatenate([np.full(M["clean"][b].shape[1], (sided or {}).get(b, "two") == "up") for b in blocks])
    ref = reference(Mc, M["we"], M["where"], "rolling")
    sign = lambda D: np.where(up, np.maximum(D, 0), np.abs(D))
    dev = np.nan_to_num(sign(Mc - ref), nan=0.0)
    valid = ~np.isnan(ref).all(axis=1); ti = M["test_idx"]
    base = np.zeros((len(ti), Mc.shape[1]))
    for n, i in enumerate(ti):
        hist = dev[:i][valid[:i]]
        mx = hist.max(axis=0)
        base[n] = mx if scale == "runmax" else np.maximum(1.4826 * np.median(hist, axis=0) * 3, 0.25 * mx)
    # rate metrics never alert on a change of fewer than 3 rows in a batch
    RATE = {"tail10", "tail99", "tail999", "lr6", "lr10", "lrrow"}
    is_rate = np.concatenate([
        np.array(["rate" in k for k in M["names"][b]]) if b in ("rules", "drift")
        else np.full(M["clean"][b].shape[1], b in RATE) for b in blocks])
    base = np.maximum(base, np.where(is_rate, 3.0 / M["n"], 0.0)) + 1e-12
    cells = {k: np.nan_to_num(sign(np.concatenate([c[b] for b in blocks], axis=1) - ref[ti]), nan=0.0) / base
             for k, c in M["cor"].items()}
    return dev[ti] / base, cells


def score(R, combine):
    if combine == "max":
        return R.max(axis=1)
    top = np.sort(R, axis=1)[:, -3:]
    return np.sqrt((top ** 2).sum(axis=1))


def run(M, blocks, budget, scale="runmax", sided=None, combine="max"):
    clean, cells = ratios(M, blocks, scale, sided)
    sc = score(clean, combine)
    for mg in MARGINS:
        fa = (sc > mg).mean()
        if fa <= budget + 1e-9:
            break
    det = {k: (score(r, combine) > mg).mean() for k, r in cells.items()}
    return mg, fa, det


if __name__ == "__main__":
    work = sys.argv[1]; budget_days = float(sys.argv[2])
    M = pickle.load(open(f"{work}/metrics.pkl", "rb"))
    T = len(M["test_idx"]); budget = budget_days / T
    STD = ["rules", "drift"]
    UP = {b: "up" for b in ["z", "tail99", "tail999", "tail10", "xmean", "chi", "v1", "lr6", "lr10", "lrrow", "lrmean"]}
    fam = {"standard": STD, "bits v1": ["v1"], "z+digits": ["z", "digits"], "z+tail10+digits": ["z", "tail10", "digits"],
           "z+tails+digits": ["z", "tail99", "tail999", "digits"], "z+tail10+tail999+digits": ["z", "tail10", "tail999", "digits"],
           "std + z+tail10+digits": STD + ["z", "tail10", "digits"],
           "lr6": ["lr6"], "lr10": ["lr10"], "lrrow": ["lrrow"], "lrmean": ["lrmean"],
           "z+digits+lr6+lrrow": ["z", "digits", "lr6", "lrrow"], "z+digits+lr10+lrrow": ["z", "digits", "lr10", "lrrow"],
           "z+tail10+digits+lr6+lrrow": ["z", "tail10", "digits", "lr6", "lrrow"],
           "z+tail10+digits+lr6+lr10+lrrow": ["z", "tail10", "digits", "lr6", "lr10", "lrrow"],
           "std + z+tail10+digits+lr6+lrrow": STD + ["z", "tail10", "digits", "lr6", "lrrow"]}
    if "lr6" not in M["clean"]:
        fam = {k: v for k, v in fam.items() if not any(b.startswith("lr") for b in v)}
    print(f"{work}: budget {budget_days:g} false-alarm day(s) in {T}")
    print(f"{'detector':26s}" + "".join(f"{c:>16s}" for c in ["max/runmax/two", "max/runmax/up", "max/mad/two", "max/mad/up", "top3/runmax/up", "top3/mad/up"]))
    for name, blocks in fam.items():
        line = f"{name:26s}"
        for combine, scale, sd in [("max", "runmax", None), ("max", "runmax", UP), ("max", "mad", None), ("max", "mad", UP), ("top3", "runmax", UP), ("top3", "mad", UP)]:
            mg, fa, det = run(M, blocks, budget, scale, sd, combine)
            line += f"{np.mean(list(det.values())):10.3f}@{mg:<5g}"
        print(line)
