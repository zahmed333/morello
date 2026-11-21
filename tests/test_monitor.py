"""The monitor's behavior on files: what alerts, what is learned, what is rejected."""

import pytest

from conftest import break_relationship, make_events
from morello import Monitor


def test_clean_batch_passes(monitor, today):
    report = monitor.check(today, update="never")
    assert report.status == "ok"
    assert report.rows_checked == 800
    assert report.batch == "2024-02-29"


def test_rows_that_stop_fitting_alert_and_name_the_column(monitor, today):
    report = monitor.check(break_relationship(today), update="never")
    assert report.alert
    assert "amount" in report.suspect_columns
    assert report.alerts == sorted(report.alerts, key=lambda alert: -alert.ratio)
    assert "amount" in report.examples
    assert "ALERT" in report.text()


def test_identifiers_get_format_checks_and_empty_columns_are_skipped(monitor):
    assert monitor.schema.format_only == {"id": "identifier or free text"}
    assert monitor.schema.skipped == {"note": "empty"}
    assert monitor.columns == ["id__length", "id__known_format", "kind", "amount", "flag"]


def test_a_clean_batch_is_learned_and_an_alerting_one_is_not(monitor, today):
    before = len(monitor.history)
    monitor.check(break_relationship(today))
    assert len(monitor.history) == before
    monitor.check(today)
    assert len(monitor.history) == before + 1


def test_checking_the_same_batch_twice_learns_it_once(monitor, today):
    monitor.check(today)
    calibrated = monitor.histories[""].calibrated_batches
    entries = len(monitor.history)
    monitor.check(today)
    assert monitor.histories[""].calibrated_batches == calibrated
    assert len(monitor.history) == entries


def test_batches_without_a_key_are_all_learned(monitor, today):
    unnamed = today.drop(columns="date")
    before = len(monitor.history)
    assert monitor.check(unnamed).batch is None
    monitor.check(unnamed)
    assert len(monitor.history) == before + 2


def test_update_never_leaves_the_monitor_unchanged(monitor, today):
    before = monitor.history_bytes()
    monitor.check(today, update="never")
    assert monitor.history_bytes() == before


def test_accepting_an_alert_widens_the_tolerance(monitor, today):
    broken = break_relationship(today)
    first = monitor.check(broken)
    monitor.accept(first)
    second = monitor.check(broken, update="never")
    assert len(second.alerts) < len(first.alerts)


def test_alerts_stay_off_until_there_is_enough_history():
    events = make_events(n_days=30, rows_per_day=300)
    monitor = Monitor(sample_rows=300, min_history=14).fit(
        events[events["date"] < "2024-01-30"], "date"
    )
    assert monitor.batches_until_ready > 0
    report = monitor.check(
        break_relationship(events[events["date"] == "2024-01-30"]), update="never"
    )
    assert report.status == "warming_up"
    assert "not enough history" in report.text()


def test_too_little_history_is_rejected():
    events = make_events(n_days=10, rows_per_day=50)
    with pytest.raises(ValueError, match="at least 20 batches"):
        Monitor().fit(events, "date")


def test_unknown_update_mode_is_rejected(monitor, today):
    with pytest.raises(ValueError, match="update must be one of"):
        monitor.check(today, update="sometimes")
