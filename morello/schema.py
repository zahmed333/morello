"""Turn an arbitrary table into numeric columns the model can read."""
import re
import numpy as np
import pandas as pd
from pandas.api import types as T

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?$")


def as_datetime(s):
    """Return the column as datetimes if it is one (or is text that looks like one), else None."""
    if T.is_datetime64_any_dtype(s):
        return s
    if T.is_bool_dtype(s) or T.is_numeric_dtype(s):
        return None
    sample = s.dropna().astype(str).head(200)
    if len(sample) and sample.map(lambda v: bool(_DATE_RE.match(v))).mean() > 0.95:
        out = pd.to_datetime(s, errors="coerce", format="ISO8601")
        if out.notna().sum() >= 0.95 * s.notna().sum():
            return out
    return None


class Schema:
    def __init__(self, max_categories=300):
        self.max_categories = max_categories

    def fit(self, df, exclude=()):
        self.plan, self.skipped, self.features, self.origin = [], {}, [], {}
        first_dt = None
        for c in df.columns:
            if c in exclude:
                continue
            s = df[c]
            if s.notna().sum() == 0:
                self.skipped[c] = "empty"
                continue
            if T.is_bool_dtype(s):
                self._add(c, "bool", [c])
            elif T.is_numeric_dtype(s):
                v = s.dropna()
                if len(v) > 100 and v.nunique() > 0.9 * len(v) and np.all(v == np.round(v)):
                    self.skipped[c] = "identifier"
                else:
                    self._add(c, "num", [c])
            elif as_datetime(s) is not None:
                feats = [f"{c}__hour"] + ([f"{c}__minus__{first_dt}"] if first_dt else [])
                self._add(c, "datetime", feats, ref=first_dt)
                first_dt = first_dt or c
            else:
                sv = s.dropna().astype(str)
                if pd.to_numeric(sv, errors="coerce").notna().mean() > 0.99:
                    self._add(c, "numstr", [c])
                    continue
                vc = sv.value_counts()
                if len(vc) > self.max_categories and len(vc) > 0.5 * len(sv):
                    self.skipped[c] = "identifier or free text"
                    continue
                keep = vc.index[: self.max_categories]
                self._add(c, "cat", [c], mapping={k: float(i) for i, k in enumerate(keep)},
                          other=float(len(keep)) if len(vc) > len(keep) else -1.0)
        return self

    def _add(self, col, kind, feats, **kw):
        self.plan.append(dict(col=col, kind=kind, feats=feats, **kw))
        for f in feats:
            self.features.append(f); self.origin[f] = col

    def transform(self, df):
        out, dts = {}, {}
        for p in self.plan:
            c, kind = p["col"], p["kind"]
            if c not in df.columns:
                for f in p["feats"]:
                    out[f] = np.full(len(df), np.nan)
                continue
            s = df[c]
            if kind == "bool":
                out[c] = s.astype("float64").to_numpy()
            elif kind == "num":
                out[c] = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
            elif kind == "numstr":
                out[c] = pd.to_numeric(s.astype(str).where(s.notna()), errors="coerce").to_numpy(dtype="float64")
            elif kind == "datetime":
                dt = s if T.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce", format="ISO8601"); dts[c] = dt
                out[f"{c}__hour"] = dt.dt.hour.to_numpy(dtype="float64")
                if p.get("ref"):
                    out[f"{c}__minus__{p['ref']}"] = (dt - dts[p["ref"]]).dt.total_seconds().to_numpy(dtype="float64")
            else:
                m = s.astype(str).map(p["mapping"])
                out[c] = np.where(s.isna().to_numpy(), np.nan, m.fillna(p["other"]).to_numpy(dtype="float64"))
        return pd.DataFrame(out, index=df.index)[self.features]
