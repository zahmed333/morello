"""Command line.

Files:
  python -m morello fit   history.parquet --batch-col date --model state_dir
  python -m morello check today.parquet   --model state_dir [--batch 2024-05-03] [--json report.json]

Tables (any SQLAlchemy URL, or duckdb:path/to/file.db):
  python -m morello fit-table   --url URL --table analytics.orders --batch-col order_ts --store STATE
  python -m morello check-table --url URL --table analytics.orders --store STATE --results-table ops.morello_results
  python -m morello accept      --url URL --table analytics.orders --store STATE --batch 2024-05-03

STATE is a folder or a cloud URL such as s3://bucket/morello.
`check` and `check-table` exit with code 1 when a batch alerts, so they can gate a pipeline step.
"""
import argparse, json, sys
import pandas as pd
from .monitor import Monitor, _batch_keys


def read(path):
    return pd.read_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pd.read_csv(path, low_memory=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="morello")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="learn from a clean history file")
    f.add_argument("file"); f.add_argument("--batch-col", required=True); f.add_argument("--model", required=True)
    f.add_argument("--freq", default="D", help="period for datetime batch columns: D, h, W ...")
    f.add_argument("--margin", type=float, default=3.0); f.add_argument("--sample-rows", type=int, default=5000)
    c = sub.add_parser("check", help="check one file; every batch in it is checked in order")
    c.add_argument("file"); c.add_argument("--model", required=True); c.add_argument("--batch", default=None)
    c.add_argument("--json", default=None); c.add_argument("--no-update", action="store_true")
    for name, helptext in (("fit-table", "learn from a table's history"), ("check-table", "check new batches of a table"),
                           ("accept", "mark a reviewed batch as fine")):
        t = sub.add_parser(name, help=helptext)
        t.add_argument("--url", required=True); t.add_argument("--table", required=True); t.add_argument("--store", required=True)
        t.add_argument("--key", default=None); t.add_argument("--dialect", default=None)
        if name == "fit-table":
            t.add_argument("--batch-col", required=True); t.add_argument("--not-dates", action="store_true")
            t.add_argument("--where", default=None); t.add_argument("--until", default=None)
            t.add_argument("--slice-col", default=None); t.add_argument("--margin", type=float, default=3.0)
            t.add_argument("--sample-rows", type=int, default=5000)
        elif name == "check-table":
            t.add_argument("--batch", default=None); t.add_argument("--results-table", default=None)
            t.add_argument("--json", default=None); t.add_argument("--no-update", action="store_true")
        else:
            t.add_argument("--batch", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "fit":
        mon = Monitor(margin=a.margin, sample_rows=a.sample_rows).fit(read(a.file), a.batch_col, freq=a.freq)
        mon.save(a.model)
        print(f"trained on {len(mon.cols)} columns, {len(mon.history)} calibration batches")
        if mon.batches_until_ready:
            print(f"warming up: alerts switch on after {mon.batches_until_ready} more clean batches")
        if mon.schema.skipped:
            print("skipped columns:", ", ".join(f"{k} ({v})" for k, v in mon.schema.skipped.items()))
        return 0
    if a.cmd == "check":
        mon = Monitor.load(a.model); df = read(a.file)
        update = "never" if a.no_update else "auto"
        if a.batch is not None or mon.batch_col not in df.columns:
            reports = [mon.check(df, batch=a.batch or "batch", update=update)]
        else:
            keys = _batch_keys(df[mon.batch_col], mon.freq)
            reports = [mon.check(df[keys == k], batch=k, update=update) for k in sorted(keys.dropna().unique())]
        if not a.no_update:
            mon.save(a.model, history_only=True)
    else:
        from .table import fit_table, check_table, accept_batch
        if a.cmd == "fit-table":
            mon = fit_table(a.url, a.table, a.batch_col, a.store, key=a.key, as_date=not a.not_dates, where=a.where,
                            until=a.until, slice_col=a.slice_col, dialect=a.dialect, margin=a.margin, sample_rows=a.sample_rows)
            print(f"trained on {len(mon.cols)} columns, {len(mon.history)} calibration batches"
                  + (f", slices {mon.table_cfg['slices']}" if mon.table_cfg["slices"] else ""))
            if mon.batches_until_ready:
                print(f"warming up: alerts switch on after {mon.batches_until_ready} more clean batches")
            return 0
        if a.cmd == "accept":
            accept_batch(a.url, a.store, a.batch, key=a.key, table=a.table, dialect=a.dialect)
            print(f"batch {a.batch} accepted")
            return 0
        reports = check_table(a.url, a.store, key=a.key, table=a.table, batch=a.batch, results_table=a.results_table,
                              update="never" if a.no_update else "auto", dialect=a.dialect)
        if not reports:
            print("no new batches")
    for r in reports:
        print(r.text())
    if getattr(a, "json", None):
        json.dump([r.to_dict() for r in reports], open(a.json, "w"), indent=1, default=str)
    return 1 if any(r.alert for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
