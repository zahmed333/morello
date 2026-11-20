"""Synthetic orders table: 60 clean days, then one day where 5% of orders get another order's
product details (a broken join). Run:  python examples/demo.py"""
import numpy as np, pandas as pd
from morello import Monitor

rng = np.random.default_rng(0)
PRODUCTS = pd.DataFrame({"product": [f"P{i:02d}" for i in range(30)],
                         "category": rng.choice(["toys", "books", "garden", "kitchen"], 30),
                         "unit_price": rng.uniform(3, 80, 30).round(2)})


def day(date, n=3000):
    p = PRODUCTS.iloc[rng.integers(0, 30, n)].reset_index(drop=True)
    country = rng.choice(["US", "DE", "JP"], n, p=[.6, .25, .15])
    qty = rng.integers(1, 5, n)
    return pd.DataFrame({
        "date": date, "order_id": [f"{date}-{i}" for i in range(n)], "country": country,
        "currency": pd.Series(country).map({"US": "USD", "DE": "EUR", "JP": "JPY"}),
        "product": p["product"], "category": p["category"], "unit_price": p["unit_price"],
        "quantity": qty, "total": (p["unit_price"] * qty).round(2),
        "express": rng.random(n) < np.where(country == "US", 0.3, 0.1)})


dates = pd.date_range("2024-01-01", periods=61).strftime("%Y-%m-%d")
history = pd.concat([day(d) for d in dates[:60]], ignore_index=True)
mon = Monitor().fit(history, batch_col="date")
print("skipped:", mon.schema.skipped)

clean = day(dates[60])
print(mon.check(clean, update="never").text())

broken = clean.copy()
idx = rng.choice(len(broken), int(0.05 * len(broken)), replace=False)
broken.loc[idx, ["product", "category", "unit_price"]] = broken.loc[np.roll(idx, 1), ["product", "category", "unit_price"]].values
print(mon.check(broken, update="never").text())
