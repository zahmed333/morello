"""Fit and check a monitor directly against a table in a warehouse."""
import json, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from .monitor import Monitor
from .sql import connect, TableSampler
from .store import open_store

RESULT_COLUMNS = ["checked_at", "table_name", "batch", "segment", "status", "rows_checked", "n_alerts",
                  "suspect_columns", "top_check", "top_column", "top_ratio", "details"]


def default_key(table):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(table)).strip("_").lower()


def result_rows(table, reports):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for r in reports:
        top = r.alerts[0] if r.alerts else {}
        rows.append(dict(checked_at=now, table_name=str(table), batch=r.batch, segment=r.segment, status=r.status,
                         rows_checked=int(r.n_rows), n_alerts=len(r.alerts), suspect_columns=", ".join(r.suspects[:8]),
                         top_check=top.get("check", ""), top_column=top.get("column", ""),
                         top_ratio=float(top.get("ratio", 0.0)),
                         details=json.dumps({"alerts": r.alerts[:10], "examples": r.examples}, default=str)))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _real_name(df, col):
    return next((c for c in df.columns if c.lower() == col.lower()), col)


def _slice_counts(sm, cfg, since=None):
    if not cfg.get("slices"):
        return {}
    sc = sm.batches(cfg["slice_col"], since=since)
    out = {}
    for b, v, n in zip(sc["b"], sc["s"], sc["n"]):
        if v in cfg["slices"]:
            out.setdefault(b, {})[v] = int(n)
    return out


def _check_batch(mon, src, sm, key, n_rows, update, slice_counts=None):
    """Global check of one batch, then one check per configured slice."""
    cfg = mon.table_cfg
    small = sm.sample(key, key, n_rows, mon.sample_rows, mon.seed)
    if len(small) == 0:
        return []
    if cfg.get("score_rows") and hasattr(src, "frame"):        # large sample scored on the cluster
        from .integrations.spark import distributed_stats
        st = distributed_stats(mon, src.frame(sm.sample_sql(key, key, n_rows, cfg["score_rows"])))
        enc = mon.schema.transform(small)
        _, detail = mon.stats(enc)
        reports = [mon.evaluate(st, mon.extras(enc), key, update, df=small, detail=detail)]
    else:
        reports = [mon.check(small, batch=key, update=update)]
    if cfg.get("slices"):
        have = (slice_counts if slice_counts is not None else _slice_counts(sm, cfg, key)).get(key, {})
        for v, g in sm.sample_slices(key, cfg["slice_col"], have, cfg["slice_sample_rows"], mon.seed).items():
            if len(g) >= 50:
                reports.append(mon.check(g, batch=key, update=update, segment=f"{cfg['slice_col']}={v}"))
    return reports


def fit_table(source, table, batch_col, store, key=None, as_date=True, where=None, until=None,
              slice_col=None, max_slices=8, min_slice_rows=500, slice_sample_rows=2000,
              train_fraction=0.5, score_rows=None, dialect=None, verbose=False, **monitor_kwargs):
    """Train and calibrate a monitor on a table's history and save it to `store`.

    until        only use batches up to and including this key (default: all of them)
    slice_col    also keep a separate alert history for each large value of this column
    score_rows   Spark only: score this many rows per batch on the cluster instead of sample_rows
    """
    src = connect(source, dialect); sm = TableSampler(src, table, batch_col, as_date, where)
    counts = sm.batches()
    if until is not None:
        counts = counts[counts["b"] <= str(until)]
    order, n = list(counts["b"]), dict(zip(counts["b"], counts["n"]))
    if len(order) < 20:
        raise ValueError(f"need at least 20 batches of history to calibrate alerts, got {len(order)}")
    n_train = max(1, int(len(order) * train_fraction))
    mon = Monitor(**monitor_kwargs)
    train = sm.sample(order[0], order[n_train - 1], sum(n[b] for b in order[:n_train]), mon.max_train_rows, mon.seed)
    mon.fit_model(train, _real_name(train, batch_col), verbose=verbose)
    slices = []
    if slice_col:
        sc = sm.batches(slice_col)
        sc = sc[sc["b"].isin(order[:n_train]) & sc["s"].notna()]
        per_batch = sc.groupby("s")["n"].sum() / n_train
        slices = [str(v) for v in per_batch[per_batch >= min_slice_rows].sort_values(ascending=False).index[:max_slices]]
    mon.table_cfg = dict(table=str(table), batch_col=batch_col, as_date=as_date, where=where, slice_col=slice_col,
                         slices=slices, slice_sample_rows=slice_sample_rows, score_rows=score_rows)
    sl = _slice_counts(sm, mon.table_cfg, order[n_train])
    for k in order[n_train:]:
        _check_batch(mon, src, sm, k, n[k], "always", sl)
    mon.cursor = {"last_checked": order[-1]}
    mon.save(open_store(store), key or default_key(table))
    return mon


def check_table(source, store, key=None, table=None, batch=None, results_table=None, update="auto",
                max_batches=31, dialect=None):
    """Check the batches that arrived since the last check (or one named batch).
    Returns the list of reports; appends one row per report to results_table when given."""
    src = connect(source, dialect); store = open_store(store)
    if key is None:
        if table is None:
            raise ValueError("give key or table")
        key = default_key(table)
    mon = Monitor.load(store, key); cfg = mon.table_cfg
    table = table or cfg["table"]
    sm = TableSampler(src, table, cfg["batch_col"], cfg["as_date"], cfg["where"])
    last = mon.cursor.get("last_checked")
    since = str(batch) if batch is not None else last
    counts = sm.batches(since=since); n = dict(zip(counts["b"], counts["n"]))
    if batch is not None:
        todo = [str(batch)]
    else:
        todo = [b for b in counts["b"] if last is None or b > last][-max_batches:]
    sl = _slice_counts(sm, cfg, todo[0]) if todo else {}
    reports = []
    for k in todo:
        if k not in n:
            continue
        reports += _check_batch(mon, src, sm, k, n[k], update, sl)
        if batch is None:
            mon.cursor["last_checked"] = k
    if reports and results_table:
        src.append(results_table, result_rows(table, reports))
    if update != "never":
        mon.save(store, key, history_only=True)
    return reports


def accept_batch(source, store, batch, key=None, table=None, dialect=None):
    """Record that an alerting batch was reviewed and is fine, so tolerances widen to include it."""
    return check_table(source, store, key=key, table=table, batch=batch, update="always", dialect=dialect)
