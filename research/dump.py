"""Score everything once and cache it, so detector designs can be compared offline.

usage: dump.py <dataset module> <raw dir> <work dir> <dev|holdout>

Cached per clean day and per corrupted day:
  - per-row, per-column code length (B), excess over expected code length (X), variance (V)
  - rule and drift metric vectors (standard checks)
  - two multivariate competitors: PCA reconstruction error, and a domain classifier's AUC
"""
import sys, os, pickle, time, zlib, importlib
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model3 import TableModel, ColumnCoder

ds = importlib.import_module(sys.argv[1])
raw_dir, work, phase = sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(work, exist_ok=True)
COLS, CONTEXT, RES = ds.COLS, ds.CONTEXT, ds.RES
C = len(COLS)
ci = {c: i for i, c in enumerate(COLS)}
SEV = [0.01, 0.05, 0.25, 1.0]
t0 = time.time()

splits = ds.load(raw_dir, phase)
train = splits["train"]
print(phase, {k: (len(v), v["batch"].nunique()) for k, v in splits.items()}, flush=True)
mpath = f"{work}/model.pkl"
if os.path.exists(mpath):
    m = pickle.load(open(mpath, "rb"))
else:
    m = TableModel(COLS, RES, CONTEXT).fit(train, rounds=40, verbose=False)
    pickle.dump(m, open(mpath, "wb"))
print("model ready", round(time.time() - t0), "s", flush=True)

# ---------- standard checks (same metric definitions as before)
coders = {c: ColumnCoder(c, RES.get(c, 0.01)).fit(train[c].to_numpy(dtype=float)) for c in COLS}
train_hist = {}
for c in COLS:
    cls, _ = coders[c].transform(train[c].to_numpy(dtype=float))
    train_hist[c] = np.bincount(cls, minlength=coders[c].n_classes) / len(cls)
rng_lo = {c: np.nanquantile(train[c], 0.001) for c in COLS}
rng_hi = {c: np.nanquantile(train[c], 0.999) for c in COLS}
seen_vals = {c: np.unique(train[c].dropna()) for c in COLS}
SEG = getattr(ds, "SEGMENT", None)
if SEG:
    seg_keys = train[SEG[0]].value_counts().index[: SEG[2]].tolist()


def drift_metrics(df):
    out = {}
    for c in COLS:
        x = df[c].to_numpy(dtype=float); coder = coders[c]
        out[f"null_rate|{c}"] = np.isnan(x).mean()
        cls, _ = coder.transform(x)
        h = np.bincount(cls, minlength=coder.n_classes) / len(cls)
        p0 = train_hist[c]
        out[f"psi|{c}"] = float(np.sum((h - p0) * np.log((h + 1e-4) / (p0 + 1e-4))))
        if coder.categorical:
            for k in np.argsort(-p0)[:8]:
                out[f"share_rate|{c}|{k}"] = h[k]
        else:
            v = x[~np.isnan(x)]
            if len(v) == 0:
                v = np.array([np.nan])
            out[f"mean|{c}"] = v.mean(); out[f"std|{c}"] = v.std()
            q = np.quantile(v, [0.1, 0.5, 0.9])
            out[f"p10|{c}"], out[f"p50|{c}"], out[f"p90|{c}"] = q
            out[f"zero_rate|{c}"] = (v == 0).mean()
    if SEG:
        seg, val = df[SEG[0]].to_numpy(dtype=float), df[SEG[1]].to_numpy(dtype=float)
        for k in seg_keys:
            v = val[seg == k]
            out[f"segmean|{SEG[1]}|{k}"] = np.nanmean(v) if len(v) and not np.all(np.isnan(v)) else np.nan
    return out


def rule_metrics(df):
    out = {}
    for c in COLS:
        x = df[c].to_numpy(dtype=float); ok = ~np.isnan(x)
        out[f"null_rate|{c}"] = (~ok).mean()
        if ok.sum() == 0:
            out[f"bad_rate|{c}"] = 0.0
        elif coders[c].categorical:
            out[f"bad_rate|{c}"] = (~np.isin(x[ok], seen_vals[c])).mean()
        else:
            out[f"bad_rate|{c}"] = ((x[ok] < rng_lo[c]) | (x[ok] > rng_hi[c])).mean()
    out["dup_rate"] = df[COLS].duplicated().mean()
    out.update(ds.extra_rules(df))
    return out


names = {}


def base_vecs(df):
    r, d = rule_metrics(df), drift_metrics(df)
    names.setdefault("rules", list(r)); names.setdefault("drift", list(d))
    return {"rules": np.array([r[k] for k in names["rules"]], dtype=float),
            "drift": np.array([d[k] for k in names["drift"]], dtype=float)}


# ---------- digit-pattern pseudo-columns (last two digits of each numeric column)
NUM = [c for c in COLS if not coders[c].categorical]
dig = {}
for c in NUM:
    res = RES.get(c, 0.01)
    v = train[c].to_numpy(float); v = v[np.isfinite(v)]
    p = np.bincount(np.mod(np.round(v / res), 100).astype(int), minlength=100) + 0.5
    p = p / p.sum(); L = -np.log2(p); H = float((p * L).sum())
    dig[c] = (res, L, H, float((p * L * L).sum() - H * H))
ALLCOLS = COLS + [f"digits|{c}" for c in NUM]


def digit_cols(df):
    n = len(df)
    B, X, V = (np.zeros((n, len(NUM)), np.float32) for _ in range(3))
    for j, c in enumerate(NUM):
        res, L, H, Vv = dig[c]
        v = df[c].to_numpy(float); ok = np.isfinite(v)
        d = np.mod(np.round(v[ok] / res), 100).astype(int)
        B[ok, j], X[ok, j], V[ok, j] = L[d], L[d] - H, Vv
    return B, X, V


def score(df, start=0, prev=None):
    pm = None if prev is None else tuple(np.asarray(a[:, :C], dtype=np.float32) for a in prev)
    B, X, V = m.score(df, start, pm)
    dB, dX, dV = digit_cols(df)
    return np.hstack([B, dB]), np.hstack([X, dX]), np.hstack([V, dV])


h16 = lambda t: tuple(np.clip(a, -6e4, 6e4).astype(np.float16) for a in t)

# ---------- multivariate competitors
is_cat = {c: coders[c].categorical for c in COLS}
freq = {c: train[c].value_counts(normalize=True).to_dict() for c in COLS if is_cat[c]}
med = {c: float(np.nanmedian(train[c])) for c in COLS if not is_cat[c]}


def encode(df):
    M = np.empty((len(df), C))
    for j, c in enumerate(COLS):
        M[:, j] = df[c].map(freq[c]).fillna(0.0) if is_cat[c] else df[c].fillna(med[c])
    return M


Tr = encode(train.sample(min(100_000, len(train)), random_state=0))
mu, sd = Tr.mean(0), Tr.std(0) + 1e-9
pca = PCA(n_components=0.65, svd_solver="full").fit((Tr - mu) / sd)


def pca_err(df):
    Z = (encode(df) - mu) / sd
    return float(np.sqrt(((Z - pca.inverse_transform(pca.transform(Z))) ** 2).sum(1)).mean())


def dc_auc(ref_rows, df, seed):
    if ref_rows is None:
        return np.nan
    Xm = np.vstack([ref_rows, df[COLS].to_numpy(float)])
    y = np.r_[np.zeros(len(ref_rows)), np.ones(len(df))]
    p = np.random.default_rng(seed).permutation(len(y)); h = len(y) // 2
    clf = lgb.train(dict(objective="binary", learning_rate=0.2, num_leaves=15, min_data_in_leaf=20,
                         verbose=-1, num_threads=2, seed=0), lgb.Dataset(Xm[p[:h]], y[p[:h]]), num_boost_round=30)
    return float(roc_auc_score(y[p[h:]], clf.predict(Xm[p[h:]])))


# ---------- clean days
frames, where = {}, {}
for k in ("ref", "calib", "test"):
    for d, g in splits[k].groupby("batch"):
        frames[d] = g.sample(min(ds.BATCH, len(g)), random_state=2).reset_index(drop=True); where[d] = k
days = sorted(frames)
refrows = {}
for i, d in enumerate(days):
    prev_days = days[max(0, i - 14): i]
    if len(prev_days) >= 7:
        pool = pd.concat([frames[x][COLS] for x in prev_days])
        refrows[d] = pool.sample(min(5000, len(pool)), random_state=3).to_numpy(float)
    else:
        refrows[d] = None
clean, out_clean, base, comp = {}, {}, {}, {}
cpath = f"{work}/part_clean.pkl"
if os.path.exists(cpath):                       # resume after an interruption
    out_clean, base, comp, names = pickle.load(open(cpath, "rb"))
    clean = {d: tuple(a.astype(np.float32) for a in t) for d, t in out_clean.items()}
else:
    for d in days:
        clean[d] = score(frames[d])
        out_clean[d] = h16(clean[d])
        base[d] = base_vecs(frames[d])
        comp[d] = {"pca": pca_err(frames[d]), "dc": dc_auc(refrows[d], frames[d], 11)}
    pickle.dump((out_clean, base, comp, names), open(cpath, "wb"), protocol=4)
print("clean days done", round(time.time() - t0), "s", flush=True)

# ---------- corrupted test days
test_days = [d for d in days if where[d] == "test"]
scen_out = {}
for name, spec in ds.scenarios(ci).items():
    spath = f"{work}/part_scen_{name}.pkl"
    if os.path.exists(spath):
        scen_out[name] = pickle.load(open(spath, "rb"))
        print(name, "loaded", flush=True)
        continue
    per_day = {}
    for d in test_days:
        df, S = frames[d], clean[d]
        n = len(df)
        rng = np.random.default_rng(zlib.crc32(f"{name}|{d}".encode()))
        perm = rng.permutation(n)
        e = {"perm": perm.astype(np.int32)}
        if spec[0] == "rowwise":
            cor, touched = spec[1](df)
            e["full"] = h16(score(cor, min(ci[c] for c in touched), S))
        for f in SEV:
            k = min(max(int(round(f * n)), 2), n)
            idx = perm[:k]
            if spec[0] == "rowwise":
                dff = df.copy(); dff.loc[idx, touched] = cor.loc[idx, touched].values
            elif spec[0] == "join":
                touched = spec[1]
                dff = df.copy(); dff.loc[idx, touched] = df.loc[np.roll(idx, 1), touched].values
                e[("join", f)] = h16(score(dff.iloc[idx], min(ci[c] for c in touched), tuple(a[idx] for a in S)))
            elif spec[0] == "dup":
                touched = []
                pool = perm[k:] if k < n else perm
                src = rng.choice(pool, size=k, replace=True)
                dff = df.copy(); dff.loc[idx, COLS] = df.loc[src, COLS].values
                e[("src", f)] = src.astype(np.int32)
            elif spec[0] == "missing":
                mask, touched = spec[1](df)
                cand = perm[mask[perm]]
                keep = np.setdiff1d(np.arange(n), cand[: int(round(f * len(cand)))])
                dff = df.iloc[keep].reset_index(drop=True)
                e[("keep", f)] = keep.astype(np.int32)
            e[("base", f)] = base_vecs(dff)
            e[("comp", f)] = {"pca": pca_err(dff), "dc": dc_auc(refrows[d], dff, 11)}
        per_day[d] = e
    scen_out[name] = {"kind": spec[0], "touched": touched, "days": per_day}
    pickle.dump(scen_out[name], open(spath, "wb"), protocol=4)
    print(name, "done", round(time.time() - t0), "s", flush=True)

pickle.dump(dict(cols=ALLCOLS, n_model_cols=C, days=days, where=where, clean=out_clean, base=base, comp=comp,
                 names=names, scen=scen_out, sev=SEV), open(f"{work}/dump.pkl", "wb"), protocol=4)
print("DUMP DONE", round(time.time() - t0), "s", flush=True)
