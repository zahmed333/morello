"""Command line.

Files:
  morello fit   history.parquet --batch-col date --model STATE
  morello check today.parquet   --model STATE [--batch 2024-05-03] [--json report.json]

Before fitting:
  morello init     history.parquet                 which column defines the batches
  morello backtest history.parquet --batch-col date    which past batches would have alerted
Both also take --url URL --table NAME in place of a file.

Tables (any SQLAlchemy URL, or duckdb:path/to/file.db):
  morello fit-table   --url URL --table analytics.orders --batch-col order_ts --store STATE
  morello check-table --url URL --table analytics.orders --store STATE \\
                      --results-table ops.morello_results
  morello accept      --url URL --table analytics.orders --store STATE --batch 2024-05-03
  morello refit-table --url URL --table analytics.orders --store STATE
  morello check-reference --url URL --child orders.customer_id --parent customers.id \\
                      --batch-col order_ts
  morello page        --url URL --table analytics.orders --store STATE \\
                      --results-table ops.morello_results --out orders.html

STATE is a folder or a cloud URL such as s3://bucket/morello.

Exit codes: 0 no alert, 1 at least one batch alerted, 2 the command could not run. `check` and
`check-table` can therefore gate a pipeline step. `check-table` keeps exiting with 1 after an
alert, also when there is nothing new to check, until the batch is accepted or passes a re-check
(`check-table --batch ...`).

`check-table --slack-webhook URL` and `--email-to a@x.org,b@x.org` send the alerts to people.
The webhook can also come from MORELLO_SLACK_WEBHOOK. Email uses MORELLO_SMTP_HOST, _PORT, _USER,
_PASSWORD and MORELLO_EMAIL_FROM.

--details lists every check behind the findings.

Reports are coloured when printed to a terminal. --color always|never overrides, and NO_COLOR
is respected. -v prints progress to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

from . import term
from .backtest import backtest, backtest_table
from .monitor import Monitor, batch_keys
from .notify import notify
from .page import write_page
from .profile import profile_frame, profile_table
from .references import check_reference
from .report import Report
from .state import StateError
from .table import (
    accept_batch,
    alert_summary,
    check_table,
    fit_table,
    refit_table,
    unresolved_alerts,
)

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
    fit_table_.add_argument(
        "--refit-every", type=int, help="fit again after this many checked batches"
    )
    _add_monitor_settings(fit_table_)
    fit_table_.set_defaults(run=run_fit_table)

    check_table_ = commands.add_parser(
        "check-table", parents=[shared], help="check new batches of a table"
    )
    _add_table_options(check_table_)
    check_table_.add_argument("--batch", help="check this one batch")
    check_table_.add_argument("--results-table", help="append one row per check to this table")
    check_table_.add_argument("--slack-webhook", help="post alerts to this Slack webhook")
    check_table_.add_argument("--email-to", help="email alerts to these addresses, comma separated")
    _add_check_options(check_table_)
    check_table_.set_defaults(run=run_check_table)

    accept = commands.add_parser("accept", parents=[shared], help="mark a reviewed batch as fine")
    _add_table_options(accept)
    accept.add_argument("--batch", required=True)
    accept.add_argument("--results-table", help="also record the decision in this table")
    accept.set_defaults(run=run_accept)
    refit = commands.add_parser(
        "refit-table", parents=[shared], help="train again on the table as it is now"
    )
    _add_table_options(refit)
    refit.set_defaults(run=run_refit_table)

    init = commands.add_parser("init", parents=[shared], help="look at a table before fitting")
    _add_file_or_table(init)
    init.set_defaults(run=run_init)

    backtest_ = commands.add_parser(
        "backtest", parents=[shared], help="replay history: which past batches would have alerted"
    )
    _add_file_or_table(backtest_)
    backtest_.add_argument("--batch-col", required=True)
    backtest_.add_argument("--fit-batches", type=int, help="batches used to fit (default: half)")
    backtest_.add_argument("--slice-col", help="tables only: also check each large slice")
    backtest_.add_argument("--json", help="also write the result to this file")
    _add_monitor_settings(backtest_)
    backtest_.set_defaults(run=run_backtest)

    reference = commands.add_parser(
        "check-reference", parents=[shared], help="check that keys exist in another table"
    )
    reference.add_argument("--url", required=True)
    reference.add_argument("--child", required=True, help="table.column holding the reference")
    reference.add_argument("--parent", required=True, help="table.column it should match")
    reference.add_argument("--batch-col", required=True, help="batch column of the child table")
    reference.add_argument("--batch", help="check this batch (default: the newest)")
    reference.add_argument("--dialect")
    reference.add_argument("--json", help="also write the report to this file")
    reference.set_defaults(run=run_check_reference, details=False)

    page = commands.add_parser("page", parents=[shared], help="write a status page for a table")
    _add_table_options(page)
    page.add_argument("--results-table", required=True)
    page.add_argument("--out", required=True, help="the HTML file to write")
    page.set_defaults(run=run_page)
    return parser


def _add_file_or_table(command: argparse.ArgumentParser) -> None:
    command.add_argument("file", nargs="?", help="a parquet or CSV file")
    command.add_argument("--url", help="a database, in place of a file")
    command.add_argument("--table", help="the table in that database")
    command.add_argument("--dialect")


def _add_monitor_settings(command: argparse.ArgumentParser) -> None:
    command.add_argument("--margin", type=float, default=3.0)
    command.add_argument(
        "--sample-rows",
        type=lambda text: text if text == "auto" else int(text),
        default=5000,
        help="rows scored per batch, or auto to size it from the table",
    )


def _add_check_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--json", help="also write the reports to this file")
    command.add_argument("--no-update", action="store_true", help="do not learn from this run")
    command.add_argument("--details", action="store_true", help="list every check that alerted")


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
        refit_every=args.refit_every,
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
    exit_code = _print_reports(reports, args)
    still_unresolved = _still_unresolved(args, reports)
    earlier = alert_summary([], still_unresolved)
    if earlier:
        print(earlier)
        exit_code = EXIT_ALERT
    slack = args.slack_webhook or os.environ.get("MORELLO_SLACK_WEBHOOK")
    email_to = [address.strip() for address in (args.email_to or "").split(",") if address.strip()]
    if exit_code == EXIT_ALERT and (slack or email_to):
        accept = (
            f"morello accept --url <URL> --table {args.table} --store {args.store} --batch <BATCH>"
        )
        notify(args.table, reports, still_unresolved, slack, email_to, accept)
    return exit_code


def _still_unresolved(args: argparse.Namespace, reports: list[Report]) -> list[str]:
    """Batches that alerted in an earlier run, are not accepted, and are not in `reports`."""
    reported = {report.batch for report in reports}
    unresolved = unresolved_alerts(args.store, args.key, args.table)
    return [batch for batch in unresolved if batch not in reported]


def run_accept(args: argparse.Namespace) -> int:
    reports = accept_batch(
        args.url,
        args.store,
        args.batch,
        key=args.key,
        table=args.table,
        results_table=args.results_table,
        dialect=args.dialect,
    )
    if not reports:
        raise ValueError(f"batch {args.batch!r} was not found in {args.table}")
    print(f"batch {args.batch} accepted")
    return EXIT_OK


def run_refit_table(args: argparse.Namespace) -> int:
    monitor = refit_table(args.url, args.store, args.key, args.table, args.dialect)
    _print_fit_summary(monitor)
    return EXIT_OK


def run_init(args: argparse.Namespace) -> int:
    if args.url:
        profile = profile_table(args.url, _required(args.table, "--table"), args.dialect)
        target = f"fit-table --url {args.url} --table {args.table}"
        store = "--store STATE"
    else:
        profile = profile_frame(read_table(_required(args.file, "a file or --url")))
        target, store = f"fit {args.file}", "--model STATE"
    print(profile.text())
    best = profile.batch_column
    if best is not None and profile.ready:
        not_dates = "" if best.as_date or not args.url else " --not-dates"
        sample = f" --sample-rows {profile.sample_rows}" if profile.sample_rows != 5000 else ""
        print(f"next: morello {target} --batch-col {best.column}{not_dates}{sample} {store}")
    return EXIT_OK if profile.ready else EXIT_ERROR


def run_backtest(args: argparse.Namespace) -> int:
    settings = {"margin": args.margin, "sample_rows": args.sample_rows}
    if args.url:
        result = backtest_table(
            args.url,
            _required(args.table, "--table"),
            args.batch_col,
            fit_batches=args.fit_batches,
            slice_col=args.slice_col,
            dialect=args.dialect,
            **settings,
        )
    else:
        table = read_table(_required(args.file, "a file or --url"))
        result = backtest(table, args.batch_col, fit_batches=args.fit_batches, **settings)
    print(result.text(color=term.enabled(args.color)))
    if args.json:
        with open(args.json, "w") as file:
            json.dump(result.to_dict(), file, indent=1, default=str)
    return EXIT_OK


def run_check_reference(args: argparse.Namespace) -> int:
    child, child_col = _table_and_column(args.child)
    parent, parent_col = _table_and_column(args.parent)
    report = check_reference(
        args.url,
        child,
        child_col,
        parent,
        parent_col,
        args.batch_col,
        batch=args.batch,
        dialect=args.dialect,
    )
    return _print_reports([report], args)


def run_page(args: argparse.Namespace) -> int:
    write_page(args.url, args.store, args.results_table, args.out, args.key, args.table)
    print(f"wrote {args.out}")
    return EXIT_OK


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"give {name}")
    return value


def _table_and_column(text: str) -> tuple[str, str]:
    table, _, column = text.rpartition(".")
    if not table or not column:
        raise ValueError(f"expected table.column, got {text!r}")
    return table, column


def read_table(path: str) -> pd.DataFrame:
    """Read a parquet file, or anything else as CSV."""
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _print_fit_summary(monitor: Monitor, suffix: str = "") -> None:
    print(
        f"trained on {len(monitor.schema.plans)} columns, "
        f"{len(monitor.history)} calibration batches{suffix}"
    )
    print(
        f"each batch is checked on up to {monitor.sample_rows} rows; a fault has to touch about "
        f"{monitor.smallest_detectable_share:.2%} of them to be noticed"
    )
    for label, columns in (
        ("format and length checks only", monitor.schema.format_only),
        ("skipped columns", monitor.schema.skipped),
    ):
        if columns:
            print(f"{label}:", ", ".join(f"{name} ({why})" for name, why in columns.items()))
    if monitor.batches_until_ready:
        print(
            f"warming up: alerts switch on after {monitor.batches_until_ready} more clean batches"
        )


def _print_reports(reports: list[Report], args: argparse.Namespace) -> int:
    color = term.enabled(args.color)
    for report in reports:
        print(report.text(color=color, details=args.details))
    if args.json:
        with open(args.json, "w") as file:
            json.dump([report.to_dict() for report in reports], file, indent=1, default=str)
    return EXIT_ALERT if any(report.alert for report in reports) else EXIT_OK
