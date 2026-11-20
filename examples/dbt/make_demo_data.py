import os, sys
import numpy as np, pandas as pd


def orders(n_days, n=1500, seed=0):
    rng = np.random.default_rng(seed); parts = []
    for d in pd.date_range("2024-01-01", periods=n_days):
        region = rng.choice(["EU", "US", "APAC"], n, p=[.5, .35, .15])
        parts.append(pd.DataFrame({"order_ts": d + pd.to_timedelta(rng.integers(0, 86400, n), unit="s"),
                                   "order_id": [f"{d:%Y%m%d}-{i}" for i in range(n)], "region": region,
                                   "currency": pd.Series(region).map({"EU": "EUR", "US": "USD", "APAC": "JPY"}).values,
                                   "amount": (np.where(region == "US", 40, 25) + rng.normal(0, 3, n)).round(2),
                                   "items": rng.integers(1, 6, n)}))
    return pd.concat(parts, ignore_index=True)


os.makedirs("data", exist_ok=True)
if sys.argv[1] == "clean":
    orders(62).to_parquet("data/orders.parquet", index=False)
else:
    df = orders(63)
    bad = (df.order_ts >= "2024-03-03") & (df.region == "US")
    df.loc[bad, "currency"] = "EUR"
    df.to_parquet("data/orders.parquet", index=False)
