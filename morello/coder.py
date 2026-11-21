"""Turn one numeric column into classes a classifier can predict, plus a within-class cost.

A column value is coded in two parts. The class says which bucket the value is in: null, one of
the column's frequent exact values, a quantile bin, or a tail. The extra bits say where inside
the bucket it is. Together they give the code length of the value in bits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ESCAPE_BITS = 32.0  # cost of a value the coder has no code for
TAIL_CAP_BITS = 64.0  # upper limit on the cost of a value far out in a tail
NULL_CLASS = 0

MAX_DISTINCT_SMALL = 20  # a column with this few distinct values is categorical
MAX_DISTINCT_INTEGER = 300  # an all-integer column with this few distinct values is categorical
MAX_CATEGORY_CLASSES = 40  # frequent categories that get a class of their own
ATOM_MIN_SHARE = 0.01  # an exact numeric value this frequent gets a class of its own
MAX_ATOMS = 20
MIN_DISTINCT_TO_BIN = 30  # fewer distinct values than this are not worth binning
QUANTILE_EDGES = 25  # bin edges between the 0.5% and 99.5% quantiles


class ColumnCoder:
    """Maps a column's values to (class index, extra bits)."""

    STATE_FIELDS = (
        "resolution",
        "categorical",
        "atoms",
        "n_classes",
        "other_class",
        "other_cost",
        "edges",
        "low_class",
        "first_bin_class",
        "high_class",
        "bin_cost",
        "low_scale",
        "high_scale",
    )

    def __init__(self, resolution: float) -> None:
        self.resolution = resolution
        self.categorical = False
        self.atoms: dict[float, int] = {}  # exact value -> class
        self.n_classes = 0
        # categorical columns
        self.other_class = 0
        self.other_cost: dict[float, float] = {}  # rare category -> bits within the "other" class
        # numeric columns
        self.edges = np.empty(0)
        self.low_class = 0
        self.first_bin_class = 0
        self.high_class = 0
        self.bin_cost = np.empty(0)
        self.low_scale = resolution
        self.high_scale = resolution

    def fit(self, values: np.ndarray) -> ColumnCoder:
        present = values[~np.isnan(values)]
        distinct, counts = np.unique(present, return_counts=True)
        all_integers = bool(np.all(distinct == np.round(distinct)))
        if len(distinct) <= MAX_DISTINCT_SMALL or (
            all_integers and len(distinct) <= MAX_DISTINCT_INTEGER
        ):
            return self._fit_categorical(distinct, counts)

        frequent = distinct[counts / len(present) >= ATOM_MIN_SHARE][:MAX_ATOMS]
        rest = present[~np.isin(present, frequent)]
        if len(np.unique(rest)) < MIN_DISTINCT_TO_BIN:
            return self._fit_categorical(distinct, counts)
        return self._fit_numeric(frequent, rest)

    def _fit_categorical(self, distinct: np.ndarray, counts: np.ndarray) -> ColumnCoder:
        self.categorical = True
        by_frequency = np.argsort(-counts)
        top = distinct[by_frequency][:MAX_CATEGORY_CLASSES]
        rare = distinct[by_frequency][MAX_CATEGORY_CLASSES:]
        rare_counts = counts[by_frequency][MAX_CATEGORY_CLASSES:]
        rare_total = max(rare_counts.sum(), 1)
        self.atoms = {float(value): i + 1 for i, value in enumerate(top)}
        self.other_class = len(top) + 1
        self.other_cost = {
            float(value): float(-np.log2(count / rare_total))
            for value, count in zip(rare, rare_counts)
        }
        self.n_classes = len(top) + 2
        return self

    def _fit_numeric(self, frequent: np.ndarray, rest: np.ndarray) -> ColumnCoder:
        self.categorical = False
        self.atoms = {float(value): i + 1 for i, value in enumerate(frequent)}
        self.edges = np.unique(np.quantile(rest, np.linspace(0.005, 0.995, QUANTILE_EDGES)))
        self.low_class = len(frequent) + 1
        self.first_bin_class = self.low_class + 1
        self.high_class = self.first_bin_class + len(self.edges) - 1
        self.n_classes = self.high_class + 1
        self.bin_cost = np.log2(np.maximum(np.diff(self.edges) / self.resolution, 1.0))
        below = rest[rest < self.edges[0]]
        above = rest[rest > self.edges[-1]]
        low_scale = float(np.mean(self.edges[0] - below)) if len(below) else self.resolution
        high_scale = float(np.mean(above - self.edges[-1])) if len(above) else self.resolution
        self.low_scale = max(low_scale, self.resolution)
        self.high_scale = max(high_scale, self.resolution)
        return self

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (class index, extra bits) per value. Nulls are class 0 with no extra bits."""
        n = len(values)
        classes = np.full(n, NULL_CLASS, dtype=np.int32)
        extra_bits = np.zeros(n)
        is_null = np.isnan(values)

        is_atom = np.zeros(n, dtype=bool)
        if self.atoms:
            atom_values = np.array(list(self.atoms.keys()))
            atom_classes = np.array(list(self.atoms.values()))
            order = np.argsort(atom_values)
            position = np.searchsorted(atom_values[order], values)
            position = np.clip(position, 0, len(atom_values) - 1)
            is_atom = (~is_null) & (atom_values[order][position] == values)
            classes[is_atom] = atom_classes[order][position[is_atom]]

        remaining = (~is_null) & (~is_atom)
        if self.categorical:
            classes[remaining] = self.other_class
            extra_bits[remaining] = [
                self.other_cost.get(float(value), ESCAPE_BITS) for value in values[remaining]
            ]
            return classes, extra_bits

        classes[remaining], extra_bits[remaining] = self._transform_binned(values[remaining])
        return classes, extra_bits

    def _transform_binned(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Classes and extra bits for numeric values that are not atoms.

        Inside the quantile range a value costs log2(bin width / resolution). In a tail it costs
        what an exponential distribution with the tail's mean distance would charge.
        """
        classes = np.empty(len(values), dtype=np.int32)
        bits = np.empty(len(values))
        low = values < self.edges[0]
        high = values > self.edges[-1]
        middle = ~(low | high)

        bin_index = np.searchsorted(self.edges, values[middle], side="right") - 1
        bin_index = np.clip(bin_index, 0, len(self.edges) - 2)
        classes[middle] = self.first_bin_class + bin_index
        bits[middle] = self.bin_cost[bin_index]

        ln2 = np.log(2)
        classes[low] = self.low_class
        bits[low] = np.log2(self.low_scale / self.resolution) + (self.edges[0] - values[low]) / (
            self.low_scale * ln2
        )
        classes[high] = self.high_class
        bits[high] = np.log2(self.high_scale / self.resolution) + (
            values[high] - self.edges[-1]
        ) / (self.high_scale * ln2)
        return classes, np.clip(bits, 0, TAIL_CAP_BITS)

    def to_state(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.STATE_FIELDS}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ColumnCoder:
        coder = cls(state["resolution"])
        for field in cls.STATE_FIELDS:
            setattr(coder, field, state[field])
        return coder
