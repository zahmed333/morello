"""A synthetic orders table shared by the examples.

Each order's total is unit_price * quantity, and product, category and unit_price always go
together. break_join() gives some orders another order's product details, the way a join on
the wrong key would. Every column still looks normal on its own afterwards.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
PRODUCT_COLUMNS = ["product", "category", "unit_price"]
PRODUCTS = pd.DataFrame(
    {
        "product": [f"P{i:02d}" for i in range(30)],
        "category": rng.choice(["toys", "books", "garden", "kitchen"], 30),
        "unit_price": rng.uniform(3, 80, 30).round(2),
    }
)
CURRENCY = {"US": "USD", "DE": "EUR", "JP": "JPY"}


def day(date: str, n_rows: int = 3000) -> pd.DataFrame:
    """One clean day of orders."""
    product = PRODUCTS.iloc[rng.integers(0, 30, n_rows)].reset_index(drop=True)
    country = rng.choice(["US", "DE", "JP"], n_rows, p=[0.6, 0.25, 0.15])
    quantity = rng.integers(1, 5, n_rows)
    return pd.DataFrame(
        {
            "date": date,
            "order_id": [f"{date}-{i}" for i in range(n_rows)],
            "country": country,
            "currency": pd.Series(country).map(CURRENCY),
            "product": product["product"],
            "category": product["category"],
            "unit_price": product["unit_price"],
            "quantity": quantity,
            "total": (product["unit_price"] * quantity).round(2),
            "express": rng.random(n_rows) < np.where(country == "US", 0.3, 0.1),
        }
    )


def break_join(orders: pd.DataFrame, share: float = 0.05) -> pd.DataFrame:
    """Give `share` of the orders another order's product, category and unit price."""
    broken = orders.copy()
    rows = rng.choice(len(broken), int(share * len(broken)), replace=False)
    broken.loc[rows, PRODUCT_COLUMNS] = broken.loc[np.roll(rows, 1), PRODUCT_COLUMNS].to_numpy()
    return broken
