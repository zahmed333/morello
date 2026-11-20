"""Table runner on DuckDB: in-engine sampling, state kept in a table, results table, slices, CLI."""
import os, tempfile
import duckdb, numpy as np, pandas as pd
from morello import TableStore
from morello.sql import connect
from morello.table import fit_table, check_table, accept_batch
from morello.__main__ import main


def orders(n_days=62, n=1500, seed=0):
    rng = np.random.default_rng(seed); parts = []
    for d in pd.date_range("2024-01-01", periods=n_days):
        region = rng.choice(["EU", "US", "APAC"], n, p=[.5, .35, .15])
        ts = d + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
        parts.append(pd.DataFrame({"order_ts": ts, "order_id": [f"{d:%Y%m%d}-{i}" for i in range(n)], "region": region,
                                   "currency": pd.Series(region).map({"EU": "EUR", "US": "USD", "APAC": "JPY"}).values,
                                   "amount": (np.where(region == "US", 40, 25) + rng.normal(0, 3, n)).round(2),
                                   "items": rng.integers(1, 6, n)}))
    return pd.concat(parts, ignore_index=True)


def test_table_runner():
    with tempfile.TemporaryDirectory() as t:
        con = duckdb.connect(f"{t}/wh.duckdb")
        df = orders(); con.register("o", df); con.execute("CREATE TABLE orders AS SELECT * FROM o")
        src = connect(con); store = TableStore(src, "morello_state")
        mon = fit_table(con, "orders", "order_ts", store, until="2024-02-29", slice_col="region",
                        min_slice_rows=200, slice_sample_rows=500, sample_rows=1000)
        assert mon.table_cfg["slices"] == ["EU", "US", "APAC"] and "order_id" in mon.schema.skipped
        assert mon.batches_until_ready == 0
        reps = check_table(con, store, table="orders", results_table="morello_results")       # 1 and 2 March
        assert sorted({r.batch for r in reps}) == ["2024-03-01", "2024-03-02"] and not any(r.alert for r in reps)
        assert check_table(con, store, table="orders") == []                                    # nothing new
        assert con.execute("select count(*) from morello_results").fetchall()[0][0] == len(reps) == 8
        # break one region on a new day: its currency no longer matches
        day = orders(n_days=63, seed=0).query("order_ts >= '2024-03-03'").copy()
        us = day["region"] == "US"; day.loc[us, "currency"] = "EUR"
        con.register("d", day); con.execute("INSERT INTO orders SELECT * FROM d")
        reps = check_table(con, store, table="orders", results_table="morello_results")
        by = {r.segment or "all": r for r in reps}
        assert by["all"].alert and by["region=US"].alert and not by["region=EU"].alert
        assert "currency" in by["region=US"].suspects
        row = con.execute("select status, suspect_columns from morello_results where batch = '2024-03-03' and segment = 'region=US'").fetchall()[0]
        assert row[0] == "alert" and "currency" in row[1]
        # the alerting batch was not learned; accepting it widens the tolerance
        n0 = len(check_table(con, store, table="orders", batch="2024-03-03", update="never")[0].alerts)
        accept_batch(con, store, "2024-03-03", table="orders")
        assert len(check_table(con, store, table="orders", batch="2024-03-03", update="never")[0].alerts) < n0
        con.close()
        # command line against the same database, state in a folder
        url = f"duckdb:{t}/wh.duckdb"
        assert main(["fit-table", "--url", url, "--table", "orders", "--batch-col", "order_ts", "--store", f"{t}/state",
                     "--until", "2024-02-29", "--sample-rows", "1000"]) == 0
        assert main(["check-table", "--url", url, "--table", "orders", "--store", f"{t}/state", "--json", f"{t}/r.json"]) == 1
        assert os.path.getsize(f"{t}/r.json") > 0


if __name__ == "__main__":
    test_table_runner(); print("ok")
