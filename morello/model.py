"""Autoregressive table model that assigns a code length (bits) to every value.

Each column is predicted from the columns before it. The code length of a
value is -log2 of the probability the model gave it, which is the size an
arithmetic coder driven by this model would produce.
"""
import os
import numpy as np
import lightgbm as lgb


def default_threads():
    """CPU cores actually available to this process (LightGBM's own default can oversubscribe in containers)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)

ESCAPE_BITS = 32.0   # cost of a value the model has no code for
P_FLOOR = 1e-6
TAIL_CAP = 64.0


class ColumnCoder:
    """Maps a column's values to classes plus a within-class cost in bits."""

    def __init__(self, name, res):
        self.name, self.res = name, res

    def fit(self, x):
        v = x[~np.isnan(x)]
        uniq, cnt = np.unique(v, return_counts=True)
        self.categorical = len(uniq) <= 20 or (
            np.all(uniq == np.round(uniq)) and len(uniq) <= 300
        )
        if self.categorical:
            order = np.argsort(-cnt)
            top = uniq[order][:40]
            self.atoms = {float(a): i + 1 for i, a in enumerate(top)}
            rest_vals, rest_cnt = uniq[order][40:], cnt[order][40:]
            self.other_class = len(top) + 1
            tot = rest_cnt.sum()
            self.other_cost = {
                float(a): -np.log2(c / tot) for a, c in zip(rest_vals, rest_cnt)
            }
            self.n_classes = len(top) + 2
            return self
        # numeric: frequent exact values become their own classes ("atoms")
        freq = cnt / len(v)
        atom_vals = uniq[freq >= 0.01][:20]
        self.atoms = {float(a): i + 1 for i, a in enumerate(atom_vals)}
        rest = v[~np.isin(v, atom_vals)]
        if len(np.unique(rest)) < 30:              # too little left to bin: treat as categorical
            self.categorical = True
            order = np.argsort(-cnt)
            top = uniq[order][:40]
            self.atoms = {float(a): i + 1 for i, a in enumerate(top)}
            rest_vals, rest_cnt = uniq[order][40:], cnt[order][40:]
            self.other_class = len(top) + 1
            tot = max(rest_cnt.sum(), 1)
            self.other_cost = {float(a): -np.log2(c / tot) for a, c in zip(rest_vals, rest_cnt)}
            self.n_classes = len(top) + 2
            return self
        edges = np.unique(np.quantile(rest, np.linspace(0.005, 0.995, 25)))
        self.edges = edges
        base = len(atom_vals) + 1
        self.low_class = base
        self.bin0 = base + 1
        self.high_class = self.bin0 + len(edges) - 1
        self.n_classes = self.high_class + 1
        widths = np.diff(edges)
        self.bin_cost = np.log2(np.maximum(widths / self.res, 1.0))
        lo, hi = rest[rest < edges[0]], rest[rest > edges[-1]]
        self.lo_scale = max(float(np.mean(edges[0] - lo)) if len(lo) else self.res, self.res)
        self.hi_scale = max(float(np.mean(hi - edges[-1])) if len(hi) else self.res, self.res)
        return self

    def transform(self, x):
        """Returns (class index, within-class bits)."""
        n = len(x)
        cls = np.zeros(n, dtype=np.int32)       # class 0 = null
        extra = np.zeros(n)
        isnan = np.isnan(x)
        atom_keys = np.array(list(self.atoms.keys()))
        atom_ids = np.array(list(self.atoms.values()))
        if len(atom_keys):
            sorter = np.argsort(atom_keys)
            pos = np.searchsorted(atom_keys[sorter], x)
            pos = np.clip(pos, 0, len(atom_keys) - 1)
            is_atom = (~isnan) & (atom_keys[sorter][pos] == x)
            cls[is_atom] = atom_ids[sorter][pos[is_atom]]
        else:
            is_atom = np.zeros(n, dtype=bool)
        todo = (~isnan) & (~is_atom)
        if self.categorical:
            cls[todo] = self.other_class
            extra[todo] = [self.other_cost.get(float(a), ESCAPE_BITS) for a in x[todo]]
            return cls, extra
        xv = x[todo]
        c = np.empty(len(xv), dtype=np.int32)
        e = np.empty(len(xv))
        low, high = xv < self.edges[0], xv > self.edges[-1]
        mid = ~(low | high)
        b = np.clip(np.searchsorted(self.edges, xv[mid], side="right") - 1, 0, len(self.edges) - 2)
        c[mid], e[mid] = self.bin0 + b, self.bin_cost[b]
        ln2 = np.log(2)
        c[low] = self.low_class
        e[low] = np.log2(self.lo_scale / self.res) + (self.edges[0] - xv[low]) / (self.lo_scale * ln2)
        c[high] = self.high_class
        e[high] = np.log2(self.hi_scale / self.res) + (xv[high] - self.edges[-1]) / (self.hi_scale * ln2)
        cls[todo], extra[todo] = c, np.clip(e, 0, TAIL_CAP)
        return cls, extra


class TableModel:
    def __init__(self, cols, res=None, context=(), threads=0):
        self.cols, self.res, self.context = list(cols), dict(res or {}), list(context)
        self.threads = threads or default_threads()

    def _features(self, df, j):
        X = df[self.context + self.cols[:j]].to_numpy(dtype=np.float64)
        lin = self.lin[j] if j < len(self.lin) else None
        if lin is not None:
            mu, sd, coef = lin
            Z = (np.nan_to_num(X, nan=0.0) - mu) / sd
            pred = np.c_[Z, np.ones(len(Z))] @ coef
            X = np.c_[X, pred]
        return X

    def _fit_linear(self, df, j):
        """Least-squares prediction of a numeric column from earlier columns,
        given to the trees as one extra feature (captures sums like totals)."""
        X = np.nan_to_num(df[self.context + self.cols[:j]].to_numpy(dtype=np.float64), nan=0.0)
        y = np.nan_to_num(df[self.cols[j]].to_numpy(dtype=np.float64), nan=0.0)
        lo, hi = np.quantile(y, [0.001, 0.999])
        keep = (y >= lo) & (y <= hi)
        mu, sd = X[keep].mean(0), X[keep].std(0) + 1e-9
        A = np.c_[(X[keep] - mu) / sd, np.ones(keep.sum())]
        coef = np.linalg.solve(A.T @ A + 1e-8 * np.eye(A.shape[1]), A.T @ y[keep])
        return mu, sd, coef

    def _linpred(self, df, j):
        mu, sd, coef = self.lin[j]
        X = np.nan_to_num(df[self.context + self.cols[:j]].to_numpy(dtype=np.float64), nan=0.0)
        return np.c_[(X - mu) / sd, np.ones(len(X))] @ coef

    def _fit_group(self, df, j):
        y = df[self.cols[j]].to_numpy(dtype=np.float64)
        ok = ~np.isnan(y)
        best = None
        for i in range(j):
            g = df[self.cols[i]].to_numpy(dtype=np.float64)
            keys = np.unique(g[~np.isnan(g)])
            if not (5 <= len(keys) <= 300):
                continue
            m = ok & ~np.isnan(g)
            if m.sum() < 100:
                continue
            idx = np.searchsorted(keys, g[m])
            cnt = np.bincount(idx, minlength=len(keys)).astype(float)
            mean = np.bincount(idx, weights=y[m], minlength=len(keys)) / np.maximum(cnt, 1)
            eta2 = float(np.sum(cnt * (mean - y[m].mean()) ** 2) / max(np.sum((y[m] - y[m].mean()) ** 2), 1e-12))
            if eta2 > 0.5 and (best is None or eta2 > best[0]):
                best = (eta2, i, keys)
        if best is None:
            return None
        _, i, keys = best
        g = df[self.cols[i]].to_numpy(dtype=np.float64)
        table = {}
        for k in keys:
            v = y[ok & (g == k)]
            if len(v) >= 5:
                q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
                table[float(k)] = (med, max((q3 - q1) / 1.349, 1e-9))
        gm = float(np.median(y[ok])); q1, q3 = np.quantile(y[ok], [0.25, 0.75])
        return i, table, (gm, max((q3 - q1) / 1.349, 1e-9))

    def _group_z(self, df, j):
        i, table, glob = self.group[j]
        y = df[self.cols[j]].to_numpy(dtype=np.float64)
        g = df[self.cols[i]].to_numpy(dtype=np.float64)
        med = np.full(len(y), glob[0]); sc = np.full(len(y), glob[1])
        for k, (m_, s_) in table.items():
            sel = g == k
            med[sel], sc[sel] = m_, s_
        return np.round((y - med) / sc, 2) + 0.0

    def _target(self, df, j):
        """Values to code: the raw column, or its residual against the linear fit."""
        if getattr(self, "group", None) and self.group[j] is not None:
            return self._group_z(df, j)
        y = df[self.cols[j]].to_numpy(dtype=np.float64)
        if self.residual[j]:
            res = self.res.get(self.cols[j], 0.01)
            y = np.round(np.round((y - self._linpred(df, j)) / res) * res, 6) + 0.0
        return y

    @staticmethod
    def _marginal_bits(coder, y):
        cls, extra = coder.transform(y)
        freq = np.bincount(cls) / len(cls)
        return float(np.mean(-np.log2(freq[cls]) + extra))

    def fit(self, df, rounds=40, verbose=False):
        import time
        self.coders, self.models, self.priors, self.lin, self.residual, self.group = [], [], [], [], [], []
        for j, c in enumerate(self.cols):
            t0 = time.time()
            raw = df[c].to_numpy(dtype=np.float64)
            coder = ColumnCoder(c, self.res.get(c, 0.01)).fit(raw)
            self.lin.append(self._fit_linear(df, j) if (not coder.categorical and j > 0) else None)
            self.residual.append(False)
            self.group.append(self._fit_group(df, j) if (not coder.categorical and j > 0) else None)
            if self.group[j] is not None:
                coder = ColumnCoder(c, 0.01).fit(self._group_z(df, j))
            elif self.lin[j] is not None:
                self.residual[j] = True
                r = self._target(df, j)
                rcoder = ColumnCoder(c, self.res.get(c, 0.01)).fit(r)
                if self._marginal_bits(rcoder, r) < self._marginal_bits(coder, raw):
                    coder = rcoder
                else:
                    self.residual[j] = False
            y, ex = coder.transform(self._target(df, j))
            present = np.unique(y)
            remap = -np.ones(coder.n_classes, dtype=np.int32)
            remap[present] = np.arange(len(present))
            coder.remap, coder.n_present = remap, len(present)
            yy_ = remap[y]
            coder.m_extra = np.array([ex[yy_ == k].mean() for k in range(len(present))])
            coder.v_extra = np.array([ex[yy_ == k].var() for k in range(len(present))])
            self.coders.append(coder)
            X = self._features(df, j)
            if len(present) == 1 or X.shape[1] == 0:
                counts = np.bincount(remap[y], minlength=len(present)) + 0.5
                self.models.append(None)
                self.priors.append(counts / counts.sum())
            else:
                params = dict(
                    objective="multiclass", num_class=len(present),
                    learning_rate=0.12, num_leaves=31, min_data_in_leaf=50,
                    lambda_l2=5.0, min_sum_hessian_in_leaf=1.0,
                    feature_fraction=0.8, max_bin=63, verbose=-1, num_threads=self.threads,
                    seed=0,
                )
                yy = remap[y]
                nva = max(len(yy) // 8, 1)
                dtr = lgb.Dataset(X[:-nva], yy[:-nva])
                dva = lgb.Dataset(X[-nva:], yy[-nva:], reference=dtr)
                m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(4, verbose=False)])
                self.best_iter = getattr(self, "best_iter", {})
                self.best_iter[c] = m.best_iteration
                self.models.append(m)
                self.priors.append(None)
            if verbose:
                print(f"{c:24s} classes={len(present):3d} "
                      f"{'cat' if coder.categorical else 'num'} "
                      f"{'group-z' if self.group[j] is not None else 'residual' if self.residual[j] else 'raw':8s} "
                      f"rounds={getattr(self, 'best_iter', {}).get(c, 0):3d} {time.time()-t0:6.1f}s", flush=True)
        return self

    def fit_no_context(self, df):
        """Frequency of every coded value on its own, ignoring the rest of the row."""
        self.L0 = []
        for j in range(len(self.cols)):
            coder = self.coders[j]
            idx = coder.remap[coder.transform(self._target(df, j))[0]]
            p = np.bincount(idx[idx >= 0], minlength=coder.n_present) + 0.5
            self.L0.append(-np.log2(p / p.sum()))
        return self

    def score(self, df):
        """Per (row, column):
        B  code length of the value given the columns before it (bits)
        X  B minus the code length the model expected for that row
        V  the model's variance of that code length
        B0 code length of the value from its plain frequency, ignoring the row
        """
        n, C = len(df), len(self.cols)
        B, X, V, B0 = (np.zeros((n, C), np.float32) for _ in range(4))
        ar = np.arange(n)
        for j in range(C):
            coder = self.coders[j]
            cls, extra = coder.transform(self._target(df, j))
            idx = coder.remap[cls]
            if self.models[j] is None:
                P = np.tile(self.priors[j], (n, 1))
            else:
                P = self.models[j].predict(self._features(df, j), num_threads=self.threads).reshape(n, -1)
            P = np.maximum(P, P_FLOOR)
            cost = -np.log2(P) + coder.m_extra[None, :]
            ecost = (P * cost).sum(1)
            var = (P * (cost ** 2 + coder.v_extra[None, :])).sum(1) - ecost ** 2
            seen = idx >= 0
            s_ = np.full(n, -np.log2(P_FLOOR) + ESCAPE_BITS)
            s_[seen] = -np.log2(P[ar[seen], idx[seen]])
            s0 = np.full(n, -np.log2(P_FLOOR) + ESCAPE_BITS)
            s0[seen] = self.L0[j][idx[seen]]
            B[:, j], X[:, j], V[:, j], B0[:, j] = s_ + extra, s_ + extra - ecost, np.maximum(var, 0), s0 + extra
        return B, X, V, B0
