import io
from morello import term
from morello.monitor import Report


def report(status="alert"):
    alerts = [dict(check="surprise", column="total", value=68.03, usual=-3.294, ratio=802.1)] if status == "alert" else []
    examples = {"total": [{"total": 40.2, "country": "US"}]} if alerts else {}
    return Report("2024-03-01", status, 3000, alerts, ["total"] if alerts else [], examples, {}, 30)


def test_colour_only_adds_codes():
    for status in ("alert", "ok", "warming_up"):
        r = report(status)
        plain, coloured = r.text(), r.text(color=True)
        assert "\x1b[" not in plain and "\x1b[" in coloured
        assert term.strip(coloured) == plain
    assert "\x1b[1;31mALERT\x1b[0m" in report("alert").text(color=True)
    assert "WARMING UP" in report("warming_up").text()


def test_when_colour_is_on(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self): return True
    monkeypatch.delenv("NO_COLOR", raising=False); monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert term.enabled("auto", Tty()) and not term.enabled("auto", io.StringIO())
    assert term.enabled("always", io.StringIO()) and not term.enabled("never", Tty())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not term.enabled("auto", Tty()) and term.enabled("always", Tty())
    monkeypatch.delenv("NO_COLOR"); monkeypatch.setenv("FORCE_COLOR", "1")
    assert term.enabled("auto", io.StringIO())
