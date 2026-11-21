"""The alert rule, and the record of clean batches it learns from.

For every check the monitor keeps two things per segment of the table:

- recent clean batches, whose median is what the check "usually" reads;
- the largest change from usual ever seen on a clean batch.

A new batch alerts when a check moves more than `margin` times that largest change. The rule is
causal: a batch is only ever compared with batches that came before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MIN_BATCHES_FOR_BASELINE = 10  # fewer recent batches than this: no baseline yet
MIN_SAME_DAY_TYPE = 3  # weekday/weekend batches needed to use a day-type baseline
RATE_FLOOR_ROWS = 3.0  # a share-of-rows check never alerts on fewer rows than this
MIN_ENTRIES_KEPT = 60  # older batches only matter through largest_deviation


def is_weekend(batch: str | None) -> bool | None:
    """True/False when the batch key is a date, None when it is not."""
    if batch is None:
        return None
    try:
        return bool(pd.Timestamp(batch).dayofweek >= 5)
    except (ValueError, TypeError, OverflowError):
        return None


def deviation_from(metrics: np.ndarray, usual: np.ndarray, increase_only: np.ndarray) -> np.ndarray:
    """How far each check moved from usual. Checks that count bad things only count increases."""
    change = metrics - usual
    moved = np.where(increase_only, np.maximum(change, 0), np.abs(change))
    return np.nan_to_num(moved, nan=0.0)


@dataclass
class HistoryEntry:
    batch: str | None  # None for a batch without a key; such batches are never deduplicated
    weekend: bool | None
    metrics: np.ndarray
    deviation: np.ndarray | None  # None when there was no baseline to compare with yet


@dataclass
class SegmentHistory:
    """Clean batches of one segment of the table (the whole table, or one slice of it)."""

    entries: list[HistoryEntry] = field(default_factory=list)
    largest_deviation: np.ndarray | None = None
    calibrated_batches: int = 0  # batches that contributed to largest_deviation

    def usual(self, weekend: bool | None, window: int) -> np.ndarray | None:
        """Median of each check over recent clean batches, of the same day type if possible."""
        recent = self.entries[-window:]
        if len(recent) < MIN_BATCHES_FOR_BASELINE:
            return None
        same_day_type = [entry for entry in recent if entry.weekend == weekend]
        if weekend is not None and len(same_day_type) >= MIN_SAME_DAY_TYPE:
            recent = same_day_type
        return np.nanmedian(np.array([entry.metrics for entry in recent]), axis=0)

    def ratio_to_largest(self, deviation: np.ndarray, is_rate: np.ndarray, rows: int) -> np.ndarray:
        """Each check's deviation as a multiple of the largest one seen on a clean batch."""
        floor = np.where(is_rate, RATE_FLOOR_ROWS / max(rows, 1), 0.0)
        return deviation / (np.maximum(self.largest_deviation, floor) + 1e-12)

    def record(self, entry: HistoryEntry, window: int) -> None:
        """Learn a clean batch. Recording the same batch key again replaces the earlier entry,
        so a job that is retried or re-run does not count a batch twice."""
        existing = self._find(entry.batch)
        if existing is None:
            self.entries.append(entry)
            newly_calibrated = entry.deviation is not None
        else:
            newly_calibrated = (
                self.entries[existing].deviation is None and entry.deviation is not None
            )
            self.entries[existing] = entry

        if entry.deviation is not None:
            if self.largest_deviation is None:
                self.largest_deviation = entry.deviation.copy()
            else:
                self.largest_deviation = np.maximum(self.largest_deviation, entry.deviation)
        if newly_calibrated:
            self.calibrated_batches += 1
        del self.entries[: -max(window, MIN_ENTRIES_KEPT)]

    def _find(self, batch: str | None) -> int | None:
        if batch is None:
            return None
        for index, entry in enumerate(self.entries):
            if entry.batch == batch:
                return index
        return None

    def to_state(self) -> dict[str, Any]:
        return {
            "entries": [vars(entry) for entry in self.entries],
            "largest_deviation": self.largest_deviation,
            "calibrated_batches": self.calibrated_batches,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> SegmentHistory:
        return cls(
            entries=[HistoryEntry(**entry) for entry in state["entries"]],
            largest_deviation=state["largest_deviation"],
            calibrated_batches=state["calibrated_batches"],
        )
