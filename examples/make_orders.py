"""Write the two files used in the README's terminal session.

orders_history.parquet   60 clean days of orders
orders_new.parquet       two more days; on the second, 5% of orders carry another order's
                         product, category and unit price

python examples/make_orders.py
morello fit   orders_history.parquet --batch-col date --model orders.monitor
morello check orders_new.parquet     --model orders.monitor
"""

import pandas as pd
from orders import break_join, day

dates = pd.date_range("2024-01-01", periods=62).strftime("%Y-%m-%d")
history = pd.concat([day(date) for date in dates[:60]], ignore_index=True)
clean_day, broken_day = day(dates[60]), break_join(day(dates[61]))

history.to_parquet("orders_history.parquet")
pd.concat([clean_day, broken_day], ignore_index=True).to_parquet("orders_new.parquet")
print("wrote orders_history.parquet and orders_new.parquet")
