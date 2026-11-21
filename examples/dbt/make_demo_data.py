"""Write data/orders.parquet for the dbt demo.

python make_demo_data.py clean     62 clean days
python make_demo_data.py broken    one more day, where US orders carry the wrong currency
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENCY = {"EU": "EUR", "US": "USD", "APAC": "JPY"}


def orders(n_days: int, rows_per_day: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = []
    for day in pd.date_range("2024-01-01", periods=n_days):
        region = rng.choice(["EU", "US", "APAC"], rows_per_day, p=[0.5, 0.35, 0.15])
        seconds = rng.integers(0, 86400, rows_per_day)
        amount = np.where(region == "US", 40, 25) + rng.normal(0, 3, rows_per_day)
        days.append(
            pd.DataFrame(
                {
                    "order_ts": day + pd.to_timedelta(seconds, unit="s"),
                    "order_id": [f"{day:%Y%m%d}-{i}" for i in range(rows_per_day)],
                    "region": region,
                    "currency": pd.Series(region).map(CURRENCY).to_numpy(),
                    "amount": amount.round(2),
                    "items": rng.integers(1, 6, rows_per_day),
                }
            )
        )
    return pd.concat(days, ignore_index=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("clean", "broken"):
        sys.exit(__doc__)
    table = orders(62 if mode == "clean" else 63)
    if mode == "broken":
        wrong = (table["order_ts"] >= "2024-03-03") & (table["region"] == "US")
        table.loc[wrong, "currency"] = "EUR"
    Path("data").mkdir(exist_ok=True)
    table.to_parquet("data/orders.parquet", index=False)
