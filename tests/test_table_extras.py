"""Around the table runner: refitting, references to other tables, notifications, the page."""

import http.server
import json
import pathlib
import threading

import duckdb
import pytest

from conftest import make_orders
from morello import Monitor
from morello.cli import main
from morello.notify import email_message, message_text, notify
from morello.page import render_page
from morello.references import check_reference
from morello.table import check_table, fit_table, refit_table, unresolved_alerts


def add_days(connection, n_days: int, since: str, **changes) -> None:
    days = make_orders(n_days=n_days).query("order_ts >= @since").assign(**changes)
    connection.register("new_days", days)
    connection.execute("INSERT INTO orders BY NAME SELECT * FROM new_days")
    connection.unregister("new_days")


@pytest.fixture
def warehouse(tmp_path):
    connection = duckdb.connect()
    connection.register("staged", make_orders())
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    connection.unregister("staged")
    connection.execute("SELECT setseed(0.5)").fetchall()  # row sampling uses random()
    store = str(tmp_path / "state")
    fit_table(connection, "orders", "order_ts", store, until="2024-02-29", sample_rows=1000)
    return connection, store


def test_refit_learns_a_new_column_and_leaves_out_unresolved_batches(warehouse):
    connection, store = warehouse
    connection.execute("ALTER TABLE orders ADD COLUMN channel VARCHAR")
    connection.execute("UPDATE orders SET channel = CASE WHEN items > 3 THEN 'app' ELSE 'web' END")
    connection.execute("UPDATE orders SET currency = 'EUR' WHERE order_ts >= '2024-03-02'")

    reports = check_table(connection, store, table="orders")
    assert reports[0].new_columns == ["channel"]
    assert unresolved_alerts(store, table="orders") == ["2024-03-02"]
    before = Monitor.load(store, "orders")

    after = refit_table(connection, store, table="orders")
    assert after.model_id != before.model_id
    assert "channel" in after.columns
    assert after.sample_rows == 1000 and after.last_checked == "2024-03-02"
    assert after.unresolved_alerts == ["2024-03-02"]
    newest_learned = max(entry.batch for entry in after.history)
    assert newest_learned == "2024-03-01"  # the unresolved batch was not learned


def test_check_table_refits_on_schedule(tmp_path):
    connection = duckdb.connect()
    connection.register("staged", make_orders(n_days=66))
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    connection.execute("SELECT setseed(0.5)").fetchall()
    store = str(tmp_path / "state")
    first = fit_table(
        connection, "orders", "order_ts", store, until="2024-02-29", sample_rows=1000, refit_every=3
    )
    assert len(check_table(connection, store, table="orders", max_batches=3)) == 3
    assert Monitor.load(store, "orders").model_id == first.model_id
    check_table(connection, store, table="orders")  # three batches checked: fits again first
    again = Monitor.load(store, "orders")
    assert again.model_id != first.model_id and again.checked_since_fit == 3


def test_missing_references_alert_only_when_they_rise(warehouse):
    connection, _ = warehouse
    connection.execute("ALTER TABLE orders ADD COLUMN customer_id INTEGER")
    connection.execute("UPDATE orders SET customer_id = items * 7 + CAST(amount AS INTEGER) % 50")
    connection.execute("CREATE TABLE customers AS SELECT DISTINCT customer_id AS id FROM orders")
    arguments = (connection, "orders", "customer_id", "customers", "id", "order_ts")

    assert check_reference(*arguments).status == "ok"
    connection.execute(
        "UPDATE orders SET customer_id = customer_id + 100000 "
        "WHERE order_ts >= '2024-03-02' AND items = 2"
    )
    report = check_reference(*arguments)
    assert report.alert and report.batch == "2024-03-02"
    assert "point at a customers.id that does not exist" in report.findings[0].summary
    assert report.examples["customer_id"][0]["customer_id"] > 100000
    assert check_reference(*arguments, batch="2024-03-01").status == "ok"


def test_alerts_reach_slack_and_email(warehouse, monkeypatch):
    connection, store = warehouse
    connection.execute("UPDATE orders SET currency = 'EUR' WHERE order_ts >= '2024-03-02'")
    reports = check_table(connection, store, table="orders")

    received = []

    class Slack(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Slack)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sent = []
    monkeypatch.setattr("morello.notify.send_email", lambda message: sent.append(message))

    webhook = f"http://127.0.0.1:{server.server_port}/hook"
    reached = notify("orders", reports, [], webhook, ["oncall@example.org"], "morello accept ...")
    server.shutdown()
    assert reached == ["slack", "email"]
    assert received[0]["text"] == "Morello: 1 batch of orders need a look"
    assert "currency:" in received[0]["blocks"][1]["text"]["text"]
    assert sent[0]["To"] == "oncall@example.org" and "morello accept" in sent[0].get_content()

    # Nothing wrong, nothing sent. A dead webhook is logged and does not raise.
    assert notify("orders", [], [], webhook) == []
    assert notify("orders", reports, [], "http://127.0.0.1:9/hook") == []
    assert message_text("orders", [], ["2024-03-02"])[1].startswith("Still open from earlier")
    assert email_message("s", "b", "a@x.org", ["b@x.org"])["From"] == "a@x.org"


def test_status_page_and_reference_command(tmp_path, capsys):
    path = str(tmp_path / "wh.duckdb")
    connection = duckdb.connect(path)
    connection.register("staged", make_orders(n_days=63))
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    connection.execute("CREATE TABLE regions AS SELECT DISTINCT region AS code FROM orders")
    connection.execute("SELECT setseed(0.5)").fetchall()
    connection.execute("UPDATE orders SET currency = 'EUR' WHERE order_ts >= '2024-03-03'")
    store, url = str(tmp_path / "state"), f"duckdb:{path}"
    fit_table(connection, "orders", "order_ts", store, until="2024-02-29", sample_rows=1000)
    check_table(connection, store, table="orders", results_table="morello_results")
    results = connection.execute("SELECT * FROM morello_results").df()
    connection.close()

    html = render_page("orders", results, Monitor.load(store, "orders"))
    assert html.count('class="cell ok"') == 2 and html.count('class="cell alert"') == 1
    assert "currency: " in html and "<h1>orders</h1>" in html

    out = str(tmp_path / "orders.html")
    table = ["--url", url, "--table", "orders", "--store", store]
    assert main(["page", *table, "--results-table", "morello_results", "--out", out]) == 0
    assert "Open alerts" in pathlib.Path(out).read_text()
    reference = ["--child", "orders.region", "--parent", "regions.code", "--batch-col", "order_ts"]
    assert main(["check-reference", "--url", url, *reference]) == 0
    assert "batch 2024-03-03 [region -> regions.code]: OK" in capsys.readouterr().out
    assert main(["refit-table", *table]) == 0
