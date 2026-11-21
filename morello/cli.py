"""Command line.

Files:
  morello fit   history.parquet --batch-col date --model STATE
  morello check today.parquet   --model STATE [--batch 2024-05-03] [--json report.json]

Tables (any SQLAlchemy URL, or duckdb:path/to/file.db):
  morello fit-table   --url URL --table analytics.orders --batch-col order_ts --store STATE
  morello check-table --url URL --table analytics.orders --store STATE \\
                      --results-table ops.morello_results
  morello accept      --url URL --table analytics.orders --store STATE --batch 2024-05-03

STATE is a folder or a cloud URL such as s3://bucket/morello.

Exit codes: 0 no alert, 1 at least one batch alerted, 2 the command could not run. `check` and
`check-table` can therefore gate a pipeline step.

Reports are coloured when printed to a terminal. --color always|never overrides, and NO_COLOR
is respected. -v prints progress to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from . import term
from .monitor import Monitor, batch_keys
from .report import Report
from .state import StateError
from .table import accept_batch, check_table, fit_table

EXIT_OK, EXIT_ALERT, EXIT_ERROR = 0, 1, 2
COLOR_CHOICES = ("auto", "always", "never")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return args.run(args)
    except (ValueError, FileNotFoundError, StateError) as error:
        if args.verbose:
            raise
        print(f"morello: {error}", file=sys.stderr)
        return EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="morello", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--color", choices=COLOR_CHOICES, default="auto", help="colour in reports")
    parser.add_argument("-v", "--verbose", action="store_true", help="print progress to stderr")
    # Options that may also follow the subcommand: `morello check ... --color never`.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--color", choices=COLOR_CHOICES, default=argparse.SUPPRESS)
    shared.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", parents=[shared], help="learn from a clean history file")
    fit.add_argument("file")
    fit.add_argument("--batch-col", required=True)
    fit.add_argument("--model", required=True, help="where to save the monitor")
    fit.add_argument("--freq", default="D", help="period of a datetime batch column: D, h, W ...")
    _add_monitor_settings(fit)
    fit.set_defaults(run=run_fit)

    check = commands.add_parser(
        "check", parents=[shared], help="check a file; every batch in it is checked in order"
    )
    check.add_argument("file")
    check.add_argument("--model", required=True)
    check.add_argument("--batch", help="treat the whole file as this one batch")
    _add_check_options(check)
    check.set_defaults(run=run_check)

    fit_table_ = commands.add_parser(
        "fit-table", parents=[shared], help="learn from a table's history"
    )
    _add_table_options(fit_table_)
    fit_table_.add_argument("--batch-col", required=True)
    fit_table_.add_argument(
        "--not-dates", action="store_true", help="batches are the distinct values of batch-col"
    )
    fit_table_.add_argument("--where", help="SQL filter, inserted as written")
    fit_table_.add_argument("--until", help="only use batches up to this key")
    fit_table_.add_argument("--slice-col", help="also check each large value of this column")
    _add_monitor_settings(fit_table_)
    fit_table_.set_defaults(run=run_fit_table)

    check_table_ = commands.add_parser(
        "check-table", parents=[shared], help="check new batches of a table"
    )
    _add_table_options(check_table_)
    check_table_.add_argument("--batch", help="check this one batch")
    check_table_.add_argument("--results-table", help="append one row per check to this table")
    _add_check_options(check_table_)
    check_table_.set_defaults(run=run_check_table)

    accept = commands.add_parser("accept", parents=[shared], help="mark a reviewed batch as fine")
    _add_table_options(accept)
    accept.add_argument("--batch", required=True)
    accept.set_defaults(run=run_accept)
    return parser


def _add_monitor_settings(command: argparse.ArgumentParser) -> None:
    command.add_argument("--margin", type=float, default=3.0)
    command.add_argument("--sample-rows", type=int, default=5000)


def _add_check_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--json", help="also write the reports to this file")
    command.add_argument("--no-update", action="store_true", help="do not learn from this run")


def _add_table_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--url", required=True)
    command.add_argument("--table", required=True)
    command.add_argument("--store", required=True, help="where the monitor is saved")
    command.add_argument("--key", help="name of the saved monitor (default: from the table name)")
    command.add_argument("--dialect", help="SQL dialect, when it cannot be told from the URL")


def run_fit(args: argparse.Namespace) -> int:
    monitor = Monitor(margin=args.margin, sample_rows=args.sample_rows)
    monitor.fit(read_table(args.file), args.batch_col, freq=args.freq)
    monitor.save(args.model)
    _print_fit_summary(monitor)
    if monitor.schema.skipped:
        skipped = ", ".join(f"{name} ({why})" for name, why in monitor.schema.skipped.items())
        print("skipped columns:", skipped)
    return EXIT_OK


def run_check(args: argparse.Namespace) -> int:
    monitor = Monitor.load(args.model)
    table = read_table(args.file)
    update = "never" if args.no_update else "auto"
    if args.batch is not None or monitor.batch_col not in table.columns:
        reports = [monitor.check(table, batch=args.batch, update=update)]
    else:
        keys = batch_keys(table[monitor.batch_col], monitor.freq)
        reports = [
            monitor.check(table[keys == key], batch=key, update=update)
            for key in sorted(keys.dropna().unique())
        ]
    if not args.no_update:
        monitor.save(args.model, history_only=True)
    return _print_reports(reports, args)


def run_fit_table(args: argparse.Namespace) -> int:
    monitor = fit_table(
        args.url,
        args.table,
        args.batch_col,
        args.store,
        key=args.key,
        as_date=not args.not_dates,
        where=args.where,
        until=args.until,
        slice_col=args.slice_col,
        dialect=args.dialect,
        margin=args.margin,
        sample_rows=args.sample_rows,
    )
    slices = monitor.table_config.slices
    _print_fit_summary(monitor, suffix=f", slices {slices}" if slices else "")
    return EXIT_OK


def run_check_table(args: argparse.Namespace) -> int:
    reports = check_table(
        args.url,
        args.store,
        key=args.key,
        table=args.table,
        batch=args.batch,
        results_table=args.results_table,
        update="never" if args.no_update else "auto",
        dialect=args.dialect,
    )
    if not reports:
        print("no new batches")
    return _print_reports(reports, args)


def run_accept(args: argparse.Namespace) -> int:
    reports = accept_batch(
        args.url, args.store, args.batch, key=args.key, table=args.table, dialect=args.dialect
    )
    if not reports:
        raise ValueError(f"batch {args.batch!r} was not found in {args.table}")
    print(f"batch {args.batch} accepted")
    return EXIT_OK


def read_table(path: str) -> pd.DataFrame:
    """Read a parquet file, or anything else as CSV."""
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _print_fit_summary(monitor: Monitor, suffix: str = "") -> None:
    print(
        f"trained on {len(monitor.columns)} columns, "
        f"{len(monitor.history)} calibration batches{suffix}"
    )
    if monitor.batches_until_ready:
        print(
            f"warming up: alerts switch on after {monitor.batches_until_ready} more clean batches"
        )


def _print_reports(reports: list[Report], args: argparse.Namespace) -> int:
    color = term.enabled(args.color)
    for report in reports:
        print(report.text(color=color))
    if args.json:
        with open(args.json, "w") as file:
            json.dump([report.to_dict() for report in reports], file, indent=1, default=str)
    return EXIT_ALERT if any(report.alert for report in reports) else EXIT_OK
