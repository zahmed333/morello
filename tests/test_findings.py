"""Alerts are grouped into one finding per problem, with a sentence, examples and a place."""

from conftest import break_relationship, make_orders
from morello import Monitor
from morello.findings import summarize
from morello.report import Alert


def alert(check, column, ratio, value=0.05, usual=0.0, detail=""):
    feature = "" if column == "<whole row>" else column
    return Alert(check, column, value, usual, ratio, feature=feature, detail=detail)


def test_checks_on_one_column_become_one_finding():
    alerts = [
        alert("surprise", "total", 900.0, value=170.0),
        alert("extreme values", "total", 45.0),
        alert("broken relationship", "<whole row>", 36.0),
        alert("broken relationship", "total", 36.0, value=0.036),
    ]
    findings = summarize(alerts, related=lambda feature: ["unit_price", "quantity"])
    assert len(findings) == 1
    assert findings[0].summary == (
        "total stopped fitting unit_price and quantity in 3.6% of rows (usual 0%)"
    )
    assert findings[0].checks == ["surprise", "extreme values", "broken relationship"]


def test_the_sentence_comes_from_the_most_specific_check():
    alerts = [
        alert("surprise", "amount", 500.0),
        alert("distribution shift", "amount", 80.0),
        alert("mean", "amount", 60.0, value=4100.0, usual=41.0),
        alert("null rate", "city", 9.0, value=0.2, usual=0.01),
        alert("duplicate rows", "<whole row>", 5.0, value=0.1),
    ]
    summaries = [finding.summary for finding in summarize(alerts, related=lambda feature: [])]
    assert summaries == [
        "amount: values shifted, mean 4100 (usual 41)",
        "city: 20% of values are null (usual 1.0%)",
        "10% of rows are exact duplicates (usual 0%)",
    ]


def test_a_column_that_only_stopped_fitting_is_folded_into_its_cause():
    alerts = [
        alert("surprise", "currency", 400.0),
        alert("null rate", "region", 90.0, value=0.4),
        alert("surprise", "items", 12.0),
    ]
    related = {"currency": ["region"], "region": [], "items": ["amount"]}
    findings = summarize(alerts, related=lambda feature: related[feature])
    assert [finding.column for finding in findings] == ["region", "items"]
    assert findings[0].affects == ["currency"] and findings[0].ratio == 400.0


def test_report_names_the_value_the_model_expected_and_where_the_rows_are():
    orders = make_orders(n_days=61)
    day = orders["order_ts"].astype(str).str[:10]
    monitor = Monitor(sample_rows=1500).fit(orders[day < "2024-03-01"], "order_ts")
    broken = orders[day == "2024-03-01"].copy()
    broken.loc[broken["region"] == "US", "currency"] = "EUR"

    report = monitor.check(broken, batch="2024-03-01")
    assert report.alert
    finding = report.findings[0]
    assert finding.column == "currency"
    assert "have the value EUR" in finding.summary
    assert finding.where.startswith("100% of the affected rows have region=US")
    example = report.examples["currency"][0]
    assert example["currency"] == "EUR" and example["(model expects)"] == "USD"
    assert "model expects USD" in report.text()
    assert "category share" in report.text(details=True)


def test_a_new_format_in_an_identifier_column_alerts(history, today, monitor):
    changed = today.copy()
    changed["id"] = "legacy/" + changed["id"].astype(str)
    report = monitor.check(changed, update="never")
    assert report.alert
    assert "id" in [finding.column for finding in report.findings]


def test_new_and_missing_columns_are_reported(today, monitor):
    changed = today.drop(columns=["flag"]).assign(channel="web")
    report = monitor.check(changed, update="never")
    assert report.new_columns == ["channel"] and report.missing_columns == ["flag"]
    assert "columns missing from this batch: flag" in report.text()
    assert any("flag: 100% of values are null" in f.summary for f in report.findings)


def test_broken_relationship_still_found(history, today, monitor):
    report = monitor.check(break_relationship(today), update="never")
    assert report.alert and report.findings
