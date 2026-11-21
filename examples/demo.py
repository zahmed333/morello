"""Morello from Python: fit on 60 clean days, check a clean day and a day with a broken join.

python examples/demo.py
"""

import pandas as pd
from orders import break_join, day

from morello import Monitor

dates = pd.date_range("2024-01-01", periods=61).strftime("%Y-%m-%d")
history = pd.concat([day(date) for date in dates[:60]], ignore_index=True)

monitor = Monitor().fit(history, batch_col="date")
print("skipped columns:", monitor.schema.skipped)

today = day(dates[60])
print(monitor.check(today, update="never").text())
print(monitor.check(break_join(today), update="never").text())
