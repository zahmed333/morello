import os, runpy, subprocess, sys, tempfile
import numpy as np, pandas as pd
from morello import Monitor

HERE = os.path.dirname(os.path.abspath(__file__))


def make(n_days=60, n=800, seed=0):
    rng = np.random.default_rng(seed); parts = []
    for d in pd.date_range("2024-01-01", periods=n_days).strftime("%Y-%m-%d"):
        kind = rng.choice(["a", "b", "c"], n)
        parts.append(pd.DataFrame({"date": d, "id": [f"{d}-{i}" for i in range(n)], "kind": kind,
                                   "amount": np.where(kind == "a", 10, 50) + rng.normal(0, 1, n).round(2),
                                   "flag": kind == "c", "note": None}))
    return pd.concat(parts, ignore_index=True)


def test_fit_check_roundtrip():
    df = make(); last = df["date"].max()
    mon = Monitor(sample_rows=800).fit(df[df.date < last], "date")
    assert "id" in mon.schema.skipped and "note" in mon.schema.skipped
    today = df[df.date == last]
    assert not mon.check(today, update="never").alert
    bad = today.copy(); bad["amount"] = bad["amount"].sample(frac=1, random_state=1).values   # break kind<->amount
    rep = mon.check(bad, update="never")
    print(rep.text())
    assert rep.alert and "amount" in rep.suspects
    with tempfile.TemporaryDirectory() as t:
        mon.save(t)
        again = Monitor.load(t)
        r1, r2 = mon.check(bad, update="never"), again.check(bad, update="never")
        assert r2.alert and [a["check"] for a in r1.alerts] == [a["check"] for a in r2.alerts]
        assert abs(r1.alerts[0]["ratio"] - r2.alerts[0]["ratio"]) < 1e-6 * max(1, r1.alerts[0]["ratio"])
        assert max(os.path.getsize(f"{t}/{f}") for f in os.listdir(t)) > 0


if __name__ == "__main__":
    test_fit_check_roundtrip(); print("ok")
