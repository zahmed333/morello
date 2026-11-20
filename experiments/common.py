import numpy as np, pandas as pd


def encode_categoricals(splits, cols):
    """Replace string categories by float codes ranked by frequency in the train split."""
    maps = {}
    for c in cols:
        order = splits["train"][c].dropna().astype(str).value_counts().index
        mp = {k: float(i) for i, k in enumerate(order)}
        maps[c] = mp
        for df in splits.values():
            s = df[c]
            df[c] = np.where(s.isna(), np.nan, s.astype(str).map(mp).fillna(-1.0))
    return maps


def split_by_date(df, ranges):
    out = {}
    for k, (lo, hi) in ranges.items():
        out[k] = df[(df["batch"] >= lo) & (df["batch"] <= hi)].reset_index(drop=True).copy()
    return out
