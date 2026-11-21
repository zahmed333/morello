"""Scoring a batch on a Spark cluster.

Each partition is scored independently and returns sums and counts. Adding them up gives the
same statistics as scoring all rows in one place, in any order (see checks.merge_statistics).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .checks import Statistics, merge_statistics
from .monitor import Monitor
from .state import pack_statistics, unpack_statistics

# One loaded monitor per executor process, keyed by the broadcast it came from.
_loaded_monitor: dict[int, Monitor] = {}


def distributed_statistics(monitor: Monitor, frame: Any) -> Statistics:
    """Score a Spark DataFrame on the executors and return the batch's additive statistics."""
    broadcast = frame.sparkSession.sparkContext.broadcast(monitor.model_bytes())

    def score_partition(chunks):
        key = id(broadcast)
        if key not in _loaded_monitor:
            _loaded_monitor.clear()
            _loaded_monitor[key] = Monitor.from_bytes(broadcast.value)
        loaded = _loaded_monitor[key]
        for chunk in chunks:
            if len(chunk):
                stats, _ = loaded.statistics(loaded.schema.transform(chunk))
                yield pd.DataFrame({"stats": [pack_statistics(stats)]})

    rows = frame.mapInPandas(score_partition, "stats binary").collect()
    return merge_statistics([unpack_statistics(row["stats"]) for row in rows])
