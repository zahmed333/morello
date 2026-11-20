"""Monitor: learn what normal batches of a table look like, then check new batches."""
import numpy as np
import pandas as pd

from .model import TableModel, ColumnCoder
from .schema import Schema, as_datetime

WHOLE_ROW = "<whole row>"
TROW = (0, 5, 10, 20)


def _batch_keys(s, freq="D"):
    """Batch key per row: calendar period for datetime-like columns, the value itself otherwise."""
    dt = as_datetime(s)
    if dt is not None:
        return dt.dt.to_period(freq).astype(str).where(dt.notna())
    return s.astype(str).where(s.notna())


def _is_weekend(key):
    try:
        return bool(pd.Timestamp(key).dayofweek >= 5)
    except Exception:
        return None


def merge_stats(parts):
    """Add up statistics computed on separate chunks of one batch."""
    parts = [p for p in parts if p]
    if not parts:
        return {}
    return {k: sum(np.asarray(p[k], dtype=float) for p in parts) for k in parts[0]}


class Report:
    def __init__(self, batch, status, n_rows, alerts, suspects, examples, skipped, history_len, segment=""):
        self.batch, self.status, self.n_rows = batch, status, n_rows
        self.alerts, self.suspects, self.examples = alerts, suspects, examples
        self.skipped, self.history_len, self.segment = skipped, history_len, segment
        self._entry = None

    @property
    def alert(self):
        return self.status == "alert"

    def to_dict(self):
        return dict(batch=self.batch, segment=self.segment, status=self.status, rows_checked=self.n_rows,
                    alerts=self.alerts, suspect_columns=self.suspects, examples=self.examples,
                    history_batches=self.history_len)

    def text(self, color=False):
        """The report as text. color=True adds terminal colour codes (see morello.term)."""
        from .term import paint
        c = lambda t, style: paint(t, style, color)
        seg = f" [{self.segment}]" if self.segment else ""
        word = c(self.status.upper().replace("_", " "), self.status)
        head = f"batch {self.batch}{seg}: {word} ({self.n_rows} rows checked, {self.history_len} batches of history)"
        if self.status == "warming_up":
            return head + "\n  not enough history yet to alert reliably"
        if not self.alerts:
            return head
        lines = [head, "  suspect columns: " + ", ".join(c(s, "bold") for s in self.suspects[:5])]
        for a in self.alerts[:8]:
            lines.append(f"  - {a['check']} | {c(a['column'], 'bold')}: {a['value']:.4g} (usual {a['usual']:.4g}), "
                         f"{c('%.1fx' % a['ratio'], 'bold')} the largest change seen before")
        for col, rows in self.examples.items():
            lines.append(f"  example rows for {c(col, 'bold')}:")
            for r in rows:
                lines.append(c("    " + ", ".join(f"{k}={v}" for k, v in r.items()), "dim"))
        return "\n".join(lines)


class Monitor:
    """
    margin       a check alerts when its change exceeds margin x the largest change seen on earlier clean batches
    window       number of recent batches that define "usual"
    sample_rows  rows scored per batch (larger batches are sampled). Keep it the same for
                 calibration and checking: how much a check normally moves depends on it.
    """

    def __init__(self, margin=3.0, window=28, min_history=14, sample_rows=5000, max_train_rows=150_000,
                 max_categories=300, threads=0, seed=0):
        self.margin, self.window, self.min_history = margin, window, min_history
        self.sample_rows, self.max_train_rows = sample_rows, max_train_rows
        self.max_categories, self.threads, self.seed = max_categories, threads, seed
        self.histories, self.runmax, self.n_valid = {}, {}, {}
        self.batch_col, self.freq, self.table_cfg, self.cursor = None, "D", {}, {}

    @property
    def history(self):
        return self.histories.get("", [])

    # ------------------------------------------------------------------ fitting
    def fit(self, df, batch_col, freq="D", train_fraction=0.5, verbose=False):
        """Learn from a history table that is believed to be clean.

        The earliest `train_fraction` of batches trains the model; the remaining batches are
        scored out-of-sample to learn how much every check normally moves."""
        keys = _batch_keys(df[batch_col], freq)
        df = df[keys.notna()]; keys = keys[keys.notna()]
        order = sorted(keys.unique())
        if len(order) < 20:
            raise ValueError(f"need at least 20 batches of history to calibrate alerts, got {len(order)}")
        n_train = max(1, int(len(order) * train_fraction))
        self.fit_model(df[keys.isin(set(order[:n_train]))], batch_col, freq, verbose)
        for k in order[n_train:]:
            self.calibrate(df[keys == k], batch=k)
        return self

    def fit_model(self, train, batch_col=None, freq="D", verbose=False):
        """Train the model on rows from the training batches. Follow with calibrate() per later batch."""
        self.batch_col, self.freq = batch_col, freq
        if len(train) > self.max_train_rows:
            train = train.sample(self.max_train_rows, random_state=self.seed)
        train = train.sample(frac=1.0, random_state=self.seed)
        exclude = []
        if batch_col is not None and batch_col in train.columns:
            dt = as_datetime(train[batch_col])
            has_time = dt is not None and bool((dt.dt.hour != 0).any())      # keep time-of-day as a column
            exclude = [] if has_time else [batch_col]
        self.schema = Schema(self.max_categories).fit(train, exclude=exclude)
        enc = self.schema.transform(train)
        self.cols = list(self.schema.features)
        self.model = TableModel(self.cols, threads=self.threads).fit(enc, verbose=verbose).fit_no_context(enc)
        self._fit_standard(enc)
        self._build_meta()
        self.histories, self.runmax, self.n_valid = {}, {}, {}
        return self

    def calibrate(self, df, batch, segment=""):
        """Record a clean batch so the monitor learns how much every check normally moves."""
        return self.check(df, batch=batch, update="always", segment=segment)

    @property
    def batches_until_ready(self):
        """Clean batches still needed before alerts are switched on."""
        return max(0, self.min_history - self.n_valid.get("", 0))

    def _fit_standard(self, enc):
        self.raw, self.hist0, self.lo, self.hi, self.seen, self.digits = {}, {}, {}, {}, {}, {}
        for c in self.cols:
            x = enc[c].to_numpy(dtype=float)
            coder = ColumnCoder(c, 0.01).fit(x)
            cls, _ = coder.transform(x)
            self.raw[c] = coder
            self.hist0[c] = np.bincount(cls, minlength=coder.n_classes) / len(cls)
            v = x[np.isfinite(x)]
            self.lo[c], self.hi[c] = (float(np.quantile(v, 0.001)), float(np.quantile(v, 0.999))) if len(v) else (0.0, 0.0)
            self.seen[c] = np.unique(v)
            if not coder.categorical:
                p = np.bincount(np.mod(np.round(v / 0.01), 100).astype(int), minlength=100) + 0.5
                p = p / p.sum(); L = -np.log2(p)
                self.digits[c] = (L, float((p * L).sum()))

    def _build_meta(self):
        """One entry per check: (name, column, counts only increases, is a rate)."""
        m = []
        for c in self.cols:
            m.append(("surprise", c, True, False))
        m.append(("extreme values", WHOLE_ROW, True, True))
        for c in self.cols:
            m.append(("extreme values", c, True, True))
        for c in self.digits:
            m.append(("digit pattern", c, False, False))
        for label in ("broken relationship", "broken relationship (strong)"):
            m.append((label, WHOLE_ROW, True, True))
            for c in self.cols:
                m.append((label, c, True, True))
        for t in TROW:
            m.append((f"rows contradicting their own columns (> {t} bits)", WHOLE_ROW, True, True))
        for c in self.cols:
            m.append(("null rate", c, False, True))
            m.append(("unexpected values", c, True, True))
            m.append(("distribution shift", c, True, False))
            if self.raw[c].categorical:
                for k in np.argsort(-self.hist0[c])[:5]:
                    m.append((f"category share #{k}", c, False, True))
            else:
                m += [("mean", c, False, False), ("spread", c, False, False), ("median", c, False, False),
                      ("zero rate", c, False, True)]
        m.append(("duplicate rows", WHOLE_ROW, True, True))
        self.meta = [tuple(x) for x in m]
        self.up = np.array([x[2] for x in m]); self.rate = np.array([x[3] for x in m])

    # ------------------------------------------------------------------ statistics
    def stats(self, enc):
        """Additive statistics of a chunk of encoded rows (sums and counts only), so chunks scored
        on different machines can be added together. Also returns per-row detail for examples."""
        B, X, V, B0 = self.model.score(enc)
        LLR = B - B0; Trow = LLR.sum(1)
        st = {"n": np.array([len(enc)], dtype=float),
              "sx": X.sum(0, dtype=np.float64), "sv": V.sum(0, dtype=np.float64),
              "t10": np.r_[(X.max(1) > 10).sum(), (X > 10).sum(0)].astype(float),
              "lr6": np.r_[(LLR.max(1) > 6).sum(), (LLR > 6).sum(0)].astype(float),
              "lr10": np.r_[(LLR.max(1) > 10).sum(), (LLR > 10).sum(0)].astype(float),
              "trow": np.array([(Trow > t).sum() for t in TROW], dtype=float)}
        dsum, dn = [], []
        for c, (L, H) in self.digits.items():
            v = enc[c].to_numpy(dtype=float); ok = np.isfinite(v)
            dsum.append(float((L[np.mod(np.round(v[ok] / 0.01), 100).astype(int)] - H).sum())); dn.append(ok.sum())
        st["dsum"], st["dn"] = np.array(dsum, dtype=float), np.array(dn, dtype=float)
        nok, nbad, s1, s2, nzero = [], [], [], [], []
        for c in self.cols:
            x = enc[c].to_numpy(dtype=float); ok = np.isfinite(x); coder = self.raw[c]
            cls, _ = coder.transform(x)
            st[f"h|{c}"] = np.bincount(cls, minlength=coder.n_classes).astype(float)
            nok.append(ok.sum())
            if coder.categorical:
                nbad.append((~np.isin(x[ok], self.seen[c])).sum()); s1.append(0.0); s2.append(0.0); nzero.append(0.0)
            else:
                v = x[ok]
                nbad.append(((v < self.lo[c]) | (v > self.hi[c])).sum())
                s1.append(v.sum()); s2.append((v * v).sum()); nzero.append((v == 0).sum())
        for k, a in (("nok", nok), ("nbad", nbad), ("s1", s1), ("s2", s2), ("nzero", nzero)):
            st[k] = np.array(a, dtype=float)
        return st, (X, LLR)

    def extras(self, enc):
        """The two statistics that cannot be added across chunks; computed on one sample."""
        med = [float(np.nanmedian(enc[c].to_numpy(dtype=float))) if enc[c].notna().any() else np.nan
               for c in self.cols if not self.raw[c].categorical]
        return {"median": np.array(med, dtype=float), "dup": float(enc.duplicated().mean()) if len(enc) else 0.0}

    def metrics_from(self, st, ex):
        n = max(float(st["n"][0]), 1.0)
        out = list(st["sx"] / np.sqrt(st["sv"] + 1e-6))
        out += list(st["t10"] / n)
        out += list(np.where(st["dn"] > 0, st["dsum"] / np.maximum(st["dn"], 1), 0.0))
        out += list(st["lr6"] / n) + list(st["lr10"] / n) + list(st["trow"] / n)
        im = 0
        for j, c in enumerate(self.cols):
            h = st[f"h|{c}"] / n; p0 = self.hist0[c]; nok = st["nok"][j]
            out += [1 - nok / n, st["nbad"][j] / nok if nok else 0.0,
                    float(np.sum((h - p0) * np.log((h + 1e-4) / (p0 + 1e-4))))]
            if self.raw[c].categorical:
                out += [h[k] for k in np.argsort(-p0)[:5]]
            else:
                mean = st["s1"][j] / nok if nok else np.nan
                var = st["s2"][j] / nok - mean * mean if nok else np.nan
                out += [mean, float(np.sqrt(max(var, 0.0))) if nok else np.nan, ex["median"][im],
                        st["nzero"][j] / nok if nok else np.nan]
                im += 1
        out.append(ex["dup"])
        return np.array(out, dtype=float)

    def _usual(self, weekend, segment):
        hist = self.histories.get(segment, [])[-self.window:]
        if len(hist) < 10:
            return None
        same = [h for h in hist if h["weekend"] == weekend]
        use = same if (weekend is not None and len(same) >= 3) else hist
        return np.nanmedian(np.array([h["metrics"] for h in use]), axis=0)

    # ------------------------------------------------------------------ checking
    def check(self, df, batch=None, update="auto", segment=""):
        """Check one batch. update: 'auto' adds the batch to history when it does not alert,
        'always' adds it regardless, 'never' leaves history unchanged."""
        if batch is None and self.batch_col in df.columns:
            k = _batch_keys(df[self.batch_col], self.freq).dropna()
            batch = k.iloc[0] if len(k) else "?"
        if len(df) > self.sample_rows:
            df = df.sample(self.sample_rows, random_state=self.seed)
        enc = self.schema.transform(df)
        st, detail = self.stats(enc)
        return self.evaluate(st, self.extras(enc), batch, update, segment, df=df, detail=detail)

    def evaluate(self, st, ex, batch, update="auto", segment="", df=None, detail=None):
        """Turn statistics into a report. Use directly when statistics were computed elsewhere
        (for example on a Spark cluster); df and detail, if given, supply example rows."""
        batch = str(batch); n_rows = int(st["n"][0])
        m = self.metrics_from(st, ex)
        weekend = _is_weekend(batch)
        usual = self._usual(weekend, segment)
        entry = dict(batch=batch, weekend=weekend, metrics=m, dev=None, segment=segment)
        alerts, suspects, examples, status = [], [], {}, "warming_up"
        if usual is not None:
            d = m - usual
            dev = np.nan_to_num(np.where(self.up, np.maximum(d, 0), np.abs(d)), nan=0.0)
            entry["dev"] = dev
            if self.n_valid.get(segment, 0) >= self.min_history:
                base = np.maximum(self.runmax[segment], np.where(self.rate, 3.0 / max(n_rows, 1), 0.0)) + 1e-12
                ratio = dev / base
                hit = np.where(ratio > self.margin)[0]
                status = "alert" if len(hit) else "ok"
                for i in hit[np.argsort(-ratio[hit])]:
                    name, col = self.meta[i][0], self.meta[i][1]
                    alerts.append(dict(check=name, column=self._source(col), value=float(m[i]), usual=float(usual[i]),
                                       ratio=float(min(ratio[i], 1e6))))
                for a in alerts:
                    if a["column"] != WHOLE_ROW and a["column"] not in suspects:
                        suspects.append(a["column"])
                if df is not None and detail is not None and len(hit):
                    examples = self._examples(df, hit, ratio, *detail)
        rep = Report(batch, status, n_rows, alerts, suspects, examples, dict(self.schema.skipped),
                     len(self.histories.get(segment, [])), segment)
        rep._entry = entry
        if update == "always" or (update == "auto" and status != "alert"):
            self._append(entry)
        return rep

    def _append(self, entry):
        seg = entry["segment"]
        self.histories.setdefault(seg, []).append(entry)
        if entry["dev"] is not None:
            self.runmax[seg] = np.maximum(self.runmax[seg], entry["dev"]) if seg in self.runmax else entry["dev"].copy()
            self.n_valid[seg] = self.n_valid.get(seg, 0) + 1
        keep = max(self.window, 60)                       # older batches only matter through runmax
        if len(self.histories[seg]) > keep:
            self.histories[seg] = self.histories[seg][-keep:]

    def accept(self, report):
        """Tell the monitor an alerting batch was fine after all, so it widens its tolerance."""
        e = report._entry
        if e is not None and all(h is not e for h in self.histories.get(e["segment"], [])):
            self._append(e)

    @property
    def last_batch(self):
        h = self.history
        return h[-1]["batch"] if h else None

    def _source(self, feature):
        return self.schema.origin.get(feature, feature)

    def _examples(self, df, hit, ratio, X, LLR, per_col=3, max_cols=2):
        out = {}
        for i in hit[np.argsort(-ratio[hit])]:
            name, col = self.meta[i][0], self.meta[i][1]
            if col == WHOLE_ROW or col not in self.cols or len(out) >= max_cols:
                continue
            src = self._source(col)
            if src in out:
                continue
            j = self.cols.index(col)
            score = LLR[:, j] if "relationship" in name else X[:, j]
            rows = df.iloc[np.argsort(-score)[:per_col]]
            show = [src] + [c for c in df.columns if c != src and c not in self.schema.skipped][:7]
            out[src] = [{c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else str(r[c]))) for c in show}
                        for _, r in rows.iterrows()]
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, target, key="monitor", history_only=False):
        """Save to a folder path, a cloud URL (needs fsspec), or a store object.
        The model is written once; the small history part is rewritten after every check."""
        from .state import dump_model, dump_history
        from .store import open_store
        store = open_store(target)
        if not history_only:
            store.put(f"{key}.model", dump_model(self))
        store.put(f"{key}.history", dump_history(self))

    @staticmethod
    def load(target, key="monitor"):
        from .state import load_monitor
        from .store import open_store
        store = open_store(target)
        model, history = store.get(f"{key}.model"), store.get(f"{key}.history")
        if model is None:
            raise FileNotFoundError(f"no saved monitor named {key!r} in {target!r}")
        return load_monitor(model, history)
