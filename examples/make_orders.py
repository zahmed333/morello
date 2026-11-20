"""Writes the two files used in the README's terminal session:

    orders_history.parquet   60 clean days of a synthetic orders table
    orders_new.parquet       two more days; on the second, 5% of orders carry another order's
                             product, category and unit price (a broken join)

    python examples/make_orders.py
    morello fit   orders_history.parquet --batch-col date --model orders.monitor
    morello check orders_new.parquet     --model orders.monitor
"""
import numpy as np, pandas as pd

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


if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=62).strftime("%Y-%m-%d")
    pd.concat([day(d) for d in dates[:60]], ignore_index=True).to_parquet("orders_history.parquet")
    clean, broken = day(dates[60]), day(dates[61])
    idx = rng.choice(len(broken), int(0.05 * len(broken)), replace=False)
    cols = ["product", "category", "unit_price"]
    broken.loc[idx, cols] = broken.loc[np.roll(idx, 1), cols].values
    pd.concat([clean, broken], ignore_index=True).to_parquet("orders_new.parquet")
    print("wrote orders_history.parquet and orders_new.parquet")
