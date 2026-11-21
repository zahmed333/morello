"""Offline comparison of detector designs from a dump.

usage: analyze.py <work dir> [fixed|rolling]

Two alert protocols, both with per-metric bars = 1.25 x the largest clean deviation:
  G (generous): bars use every clean day, test month included -> no false alarms by construction.
  S (strict):   bars use only clean days before the test month. Reported: share of clean test
                days with a false alarm, and detection after muting metrics that false-alarmed.
"""
import sys, pickle
import numpy as np, pandas as pd

MARGIN = 1.25


def load(work):
    D = pickle.load(open(f"{work}/dump.pkl", "rb"))
    D["clean32"] = {d: tuple(a.astype(np.float32) for a in t) for d, t in D["clean"].items()}
    import os
    if os.path.exists(f"{work}/dump0.pkl"):
        D0 = pickle.load(open(f"{work}/dump0.pkl", "rb"))
        for d in D["clean32"]:                    # 4th array: no-context code length
            D["clean32"][d] = D["clean32"][d] + (D0["clean"][d].astype(np.float32),)
        for sc in D["scen"]:
            for d, e in D["scen"][sc]["days"].items():
                for k, v in D0["scen"][sc][d].items():
                    e[("b0", k)] = v
        D["has0"] = True
    return D


def composed(D, sc, d, f):
    e = D["scen"][sc]["days"][d]; kind = D["scen"][sc]["kind"]
    S = D["clean32"][d]; n = len(S[0])
    k = min(max(int(round(f * n)), 2), n); idx = e["perm"][:k]
    if kind == "missing":
        return tuple(a[e[("keep", f)]] for a in S)
    out = tuple(a.copy() for a in S)
    for a, i in zip(out, range(len(S))):
        if kind == "rowwise":
            a[idx] = e["full"][i][idx] if i < 3 else e[("b0", "full")][idx]
        elif kind == "join":
            a[idx] = e[("join", f)][i] if i < 3 else e[("b0", ("join", f))]
        elif kind == "dup":
            a[idx] = S[i][e[("src", f)]]
    return out


def thresholds(D):
    ref_days = [d for d in D["days"] if D["where"][d] == "ref"]
    B = np.concatenate([D["clean32"][d][0] for d in ref_days]); X = np.concatenate([D["clean32"][d][1] for d in ref_days])
    Cm = D["n_model_cols"]
    return {"row995": np.quantile(B[:, :Cm].sum(1), 0.995),
            "x99": np.quantile(X, 0.99, axis=0), "x999": np.quantile(X, 0.999, axis=0),
            "rx99": np.quantile(X[:, :Cm].sum(1), 0.99), "rx999": np.quantile(X[:, :Cm].sum(1), 0.999)}


def superset(D, S, thr):
    """All candidate bits statistics for one batch, as named blocks."""
    B, X, V = S[:3]; Cm = D["n_model_cols"]
    row = B[:, :Cm].sum(1); rx = X[:, :Cm].sum(1)
    z = X[:, :Cm].sum(0) / np.sqrt(V[:, :Cm].sum(0) + 1e-6)
    lr = {}
    if len(S) > 3:        # log likelihood ratio per value: bits given the row minus bits on its own
        LLR = B[:, :Cm] - S[3]; T = LLR.sum(1)
        lr = {"lr6": np.r_[(LLR.max(1) > 6).mean(), (LLR > 6).mean(0)],
              "lr10": np.r_[(LLR.max(1) > 10).mean(), (LLR > 10).mean(0)],
              "lrrow": np.array([(T > 0).mean(), (T > 5).mean(), (T > 10).mean(), (T > 20).mean()]),
              "lrmean": np.r_[T.mean(), LLR.mean(0)]}
    return {
        **lr,
        "chi": np.r_[(z ** 2).sum(), (np.maximum(z, 0) ** 2).sum()],
        "tail10": np.r_[(X[:, :Cm].max(1) > 10).mean(), (X[:, :Cm] > 10).mean(0)],
        "v1": np.r_[row.mean(), (row > thr["row995"]).mean(), B[:, :Cm].mean(0)],
        "xmean": np.r_[rx.mean(), X[:, :Cm].mean(0)],
        "z": X[:, :Cm].sum(0) / np.sqrt(V[:, :Cm].sum(0) + 1e-6),
        "tail99": np.r_[(rx > thr["rx99"]).mean(), (X[:, :Cm] > thr["x99"][:Cm]).mean(0)],
        "tail999": np.r_[(rx > thr["rx999"]).mean(), (X[:, :Cm] > thr["x999"][:Cm]).mean(0)],
        "clip": np.clip(X[:, :Cm], -8, 8).mean(0),
        "digits": X[:, Cm:].mean(0),
    }


def build(work):
    D = load(work); thr = thresholds(D)
    days = D["days"]; test_days = [d for d in days if D["where"][d] == "test"]
    M = {"days": days, "where": np.array([D["where"][d] for d in days]),
         "we": np.array([pd.Timestamp(d).dayofweek >= 5 for d in days]),
         "test_idx": np.array([days.index(d) for d in test_days]), "touched": {s: D["scen"][s]["touched"] for s in D["scen"]},
         "cols": D["cols"], "n_model_cols": D["n_model_cols"], "sev": D["sev"], "clean": {}, "cor": {},
         "n": float(np.median([len(D["clean32"][d][0]) for d in days])), "names": D["names"]}
    sup = [superset(D, D["clean32"][d], thr) for d in days]
    for k in sup[0]:
        M["clean"][k] = np.array([s[k] for s in sup])
    for k in ("rules", "drift"):
        M["clean"][k] = np.array([D["base"][d][k] for d in days])
    for k in ("pca", "dc"):
        M["clean"][k] = np.array([[D["comp"][d][k]] for d in days])
    for sc in D["scen"]:
        for f in D["sev"]:
            sup = [superset(D, composed(D, sc, d, f), thr) for d in test_days]
            cell = {k: np.array([s[k] for s in sup]) for k in sup[0]}
            for k in ("rules", "drift"):
                cell[k] = np.array([D["scen"][sc]["days"][d][("base", f)][k] for d in test_days])
            for k in ("pca", "dc"):
                cell[k] = np.array([[D["scen"][sc]["days"][d][("comp", f)][k]] for d in test_days])
            M["cor"][(sc, f)] = cell
    pickle.dump(M, open(f"{work}/metrics.pkl", "wb"), protocol=4)
    return M


def reference(Mc, we, where, mode, W=28, minh=10):
    D = len(Mc); ref = np.full(Mc.shape, np.nan)
    if mode == "fixed":
        for g in (False, True):
            sel = (where == "ref") & (we == g)
            med = np.nanmedian(Mc[sel] if sel.any() else Mc[where == "ref"], axis=0)
            ref[(where != "ref") & (we == g)] = med
        return ref
    for i in range(D):
        sel = np.arange(max(0, i - W), i)
        if len(sel) < minh:
            continue
        same = sel[we[sel] == we[i]]
        ref[i] = np.nanmedian(Mc[same if len(same) >= 3 else sel], axis=0)
    return ref


def evaluate(M, blocks, mode="rolling"):
    Mc = np.concatenate([M["clean"][b] for b in blocks], axis=1)
    ref = reference(Mc, M["we"], M["where"], mode)
    dev = np.nan_to_num(np.abs(Mc - ref), nan=0.0)
    valid = ~np.isnan(ref).all(axis=1)
    ti = M["test_idx"]; pre = valid & (M["where"] != "test")
    gbar = MARGIN * dev[valid].max(axis=0) + 1e-12
    sbar = MARGIN * dev[pre].max(axis=0) + 1e-12
    fired_clean = dev[ti] > sbar
    fa = fired_clean.any(axis=1).mean(); mute = fired_clean.any(axis=0)
    rows = []
    for (sc, f), cell in M["cor"].items():
        X = np.concatenate([cell[b] for b in blocks], axis=1)
        dc = np.nan_to_num(np.abs(X - ref[ti]), nan=0.0)
        rows.append(dict(scenario=sc, sev=f, G=(dc > gbar).any(axis=1).mean(),
                         S=(dc[:, ~mute] > sbar[~mute]).any(axis=1).mean() if (~mute).any() else 0.0,
                         hitG=(dc > gbar).any(axis=1), ratio=dc / gbar))
    return pd.DataFrame(rows), fa, int(mute.sum()), Mc.shape[1]


if __name__ == "__main__":
    import os
    work = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "rolling"
    M = pickle.load(open(f"{work}/metrics.pkl", "rb")) if os.path.exists(f"{work}/metrics.pkl") else build(work)
    STD = ["rules", "drift"]
    DET = {
        "standard checks": STD,
        "PCA reconstruction": ["pca"],
        "domain classifier": ["dc"],
        "standard + both multivariate": STD + ["pca", "dc"],
        "bits v1 (mean bits)": ["v1"],
        "A excess mean": ["xmean"],
        "B excess z": ["z"],
        "C excess mean + tails": ["xmean", "tail99", "tail999"],
        "D z + tails": ["z", "tail99", "tail999"],
        "E z + tails + digits": ["z", "tail99", "tail999", "digits"],
        "F z + tails + digits + clip": ["z", "tail99", "tail999", "digits", "clip"],
        "G E + tail10": ["z", "tail99", "tail999", "digits", "tail10"],
        "H E + chi": ["z", "tail99", "tail999", "digits", "chi"],
        "I E + tail10 + chi": ["z", "tail99", "tail999", "digits", "tail10", "chi"],
        "J z + tail10 + digits": ["z", "tail10", "digits"],
        "standard + E": STD + ["z", "tail99", "tail999", "digits"],
        "standard + both mv + E": STD + ["pca", "dc", "z", "tail99", "tail999", "digits"],
    }
    res = {}
    print(f"{work}  reference={mode}  test days={len(M['test_idx'])}")
    print(f"{'detector':30s} {'metrics':>7s} {'G det':>6s} {'S det':>6s} {'S FA':>6s} {'muted':>5s}")
    for name, blocks in DET.items():
        df, fa, muted, k = evaluate(M, blocks, mode)
        res[name] = df
        print(f"{name:30s} {k:7d} {df.G.mean():6.3f} {df.S.mean():6.3f} {fa:6.2f} {muted:5d}")
    show = [s for s in sys.argv[3:]] or ["standard checks", "PCA reconstruction", "domain classifier", "bits v1 (mean bits)", "E z + tails + digits", "standard + E"]
    tab = pd.DataFrame({n: res[n].groupby("scenario", sort=False).G.mean() for n in show}).round(2)
    print("\nG detection by failure type (mean over severities)"); print(tab.to_string())
    sev = pd.DataFrame({n: res[n].groupby("sev").G.mean() for n in show}).round(2)
    print("\nG detection by share of rows hit"); print(sev.to_string())
