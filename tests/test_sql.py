"""Query building: names that are refused, and what the sampler returns."""

import duckdb
import pytest

from conftest import make_orders
from morello.sql import DuckDBSource, TableSampler, connect, identifier


@pytest.mark.parametrize(
    "name", ["orders", "analytics.orders", "_morello_src", "DB.SCHEMA.T$1", "`my-project`.ds.t"]
)
def test_ordinary_names_are_accepted(name):
    assert identifier(name) == name


@pytest.mark.parametrize("name", ["orders; DROP TABLE x", "a b", "o' OR 1=1", "", "a..b", "1abc"])
def test_names_that_could_change_the_query_are_refused(name):
    with pytest.raises(ValueError, match="not a table or column name"):
        identifier(name)


@pytest.fixture(scope="module")
def source():
    connection = duckdb.connect()
    connection.register("staged", make_orders(n_days=5, rows_per_day=400))
    connection.execute("CREATE TABLE orders AS SELECT * FROM staged")
    return connect(connection)


def test_connect_wraps_a_duckdb_connection(source):
    assert isinstance(source, DuckDBSource)
    assert source.table_exists("orders")
    assert not source.table_exists("no_such_table")


def test_batch_counts_are_per_day_and_oldest_first(source):
    counts = TableSampler(source, "orders", "order_ts").batch_counts()
    assert list(counts["batch"]) == [f"2024-01-0{day}" for day in range(1, 6)]
    assert list(counts["n_rows"]) == [400] * 5
    later = TableSampler(source, "orders", "order_ts").batch_counts(since="2024-01-04")
    assert list(later["batch"]) == ["2024-01-04", "2024-01-05"]


def test_sample_returns_the_rows_asked_for_from_the_batches_asked_for(source):
    sampler = TableSampler(source, "orders", "order_ts")
    rows = sampler.sample("2024-01-02", "2024-01-03", total_rows=800, want_rows=100)
    assert len(rows) == 100
    assert set(rows["order_ts"].dt.strftime("%Y-%m-%d")) <= {"2024-01-02", "2024-01-03"}
    everything = sampler.sample("2024-01-02", "2024-01-02", total_rows=400, want_rows=5000)
    assert len(everything) == 400


def test_where_filter_and_slices(source):
    sampler = TableSampler(source, "orders", "order_ts", where="region = 'EU'")
    assert set(sampler.sample("2024-01-01", "2024-01-01", 400, 400)["region"]) == {"EU"}

    sampler = TableSampler(source, "orders", "order_ts")
    counts = sampler.batch_counts("region")
    per_region = dict(zip(counts["slice"], counts["n_rows"]))
    samples = sampler.sample_slices("2024-01-01", "region", {"US": per_region["US"]}, want_rows=20)
    assert set(samples) == {"US"}
    assert len(samples["US"]) == 20 and set(samples["US"]["region"]) == {"US"}
