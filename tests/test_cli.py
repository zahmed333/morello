"""The command line: output, exit codes, colour."""

import io
import json

import duckdb
import pytest

from conftest import LAST_DAY, break_relationship, make_orders
from morello import term
from morello.cli import EXIT_ALERT, EXIT_ERROR, EXIT_OK, main
from morello.report import Alert, Report


@pytest.fixture
def files(monitor, today, tmp_path):
    """A saved monitor, a clean batch and a broken batch on disk."""
    model = tmp_path / "state"
    monitor.save(model)
    clean, broken = tmp_path / "clean.parquet", tmp_path / "broken.csv"
    today.to_parquet(clean)
    break_relationship(today).to_csv(broken, index=False)
    return str(model), str(clean), str(broken)


def test_check_exits_0_on_a_clean_file_and_1_on_an_alert(files, capsys, tmp_path):
    model, clean, broken = files
    assert main(["check", clean, "--model", model, "--no-update"]) == EXIT_OK
    assert f"batch {LAST_DAY}: OK" in capsys.readouterr().out

    report_file = tmp_path / "report.json"
    assert main(["check", broken, "--model", model, "--json", str(report_file)]) == EXIT_ALERT
    assert "ALERT" in capsys.readouterr().out
    saved = json.loads(report_file.read_text())
    assert saved[0]["status"] == "alert" and "amount" in saved[0]["suspect_columns"]


def test_a_command_that_cannot_run_exits_2_with_a_message(tmp_path, capsys, files):
    _, clean, _ = files
    assert main(["check", clean, "--model", str(tmp_path / "missing")]) == EXIT_ERROR
    assert "no saved monitor" in capsys.readouterr().err


def test_fit_and_check_a_table(tmp_path, capsys):
    database = tmp_path / "wh.duckdb"
    connection = duckdb.connect(str(database))
    connection.register("staged", make_orders(n_days=55, rows_per_day=600))
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    connection.close()
    common = ["--url", f"duckdb:{database}", "--table", "orders", "--store", str(tmp_path / "s")]

    fit = ["fit-table", *common, "--batch-col", "order_ts", "--until", "2024-02-23"]
    assert main([*fit, "--sample-rows", "600"]) == EXIT_OK
    assert "calibration batches" in capsys.readouterr().out
    assert main(["check-table", *common]) == EXIT_OK
    assert "batch 2024-02-24: OK" in capsys.readouterr().out
    assert main(["check-table", *common]) == EXIT_OK
    assert "no new batches" in capsys.readouterr().out
    assert main(["accept", *common, "--batch", "1999-01-01"]) == EXIT_ERROR


def report(status: str) -> Report:
    alerts = [Alert("surprise", "total", 68.03, -3.294, 802.1)] if status == "alert" else []
    return Report(
        batch="2024-03-01",
        status=status,
        rows_checked=3000,
        history_batches=30,
        alerts=alerts,
        suspect_columns=["total"] if alerts else [],
        examples={"total": [{"total": 40.2, "country": "US"}]} if alerts else {},
    )


@pytest.mark.parametrize("status", ["alert", "ok", "warming_up"])
def test_colour_adds_codes_and_nothing_else(status):
    plain, coloured = report(status).text(), report(status).text(color=True)
    assert "\x1b[" not in plain and "\x1b[" in coloured
    assert term.strip(coloured) == plain


def test_status_words():
    assert "\x1b[1;31mALERT\x1b[0m" in report("alert").text(color=True)
    assert "WARMING UP" in report("warming_up").text()


def test_when_colour_is_on(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert term.enabled("auto", Terminal()) and not term.enabled("auto", io.StringIO())
    assert term.enabled("always", io.StringIO()) and not term.enabled("never", Terminal())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not term.enabled("auto", Terminal()) and term.enabled("always", Terminal())
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert term.enabled("auto", io.StringIO())
