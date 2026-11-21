"""Accept a batch that the morello_ok test flagged, after a person has looked at it.

    python accept.py 2024-03-03

The next `dbt build` then passes. With an incremental results model, also pass
results_table="main.orders_quality" so the decision is recorded next to the alert.
"""

import sys

import duckdb

from morello import TableStore
from morello.sql import connect
from morello.table import accept_batch

source = connect(duckdb.connect("demo.duckdb"))
state = TableStore(source, "main.morello_state")
accept_batch(source, state, batch=sys.argv[1], key="orders", table="main.orders")
print(f"batch {sys.argv[1]} accepted")
