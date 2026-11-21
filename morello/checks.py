"""What is measured on every batch.

Three steps, kept separate so that a batch can be scored in pieces on different machines:

1. statistics()         sums and counts over a chunk of rows. Chunks add up (merge_statistics).
2. sample_statistics()  the two measurements that do not add up: medians and duplicate rows.
3. metrics()            one number per check, computed from the totals.

The checks come in two families. Model checks ask how surprising rows are to the table model.
Standard checks look at one column at a time: nulls, unexpected values, distribution shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .coder import ColumnCoder
from .model import RESOLUTION, Scores

WHOLE_ROW = "<whole row>"

EXTREME_EXCESS_BITS = 10  # a value this many bits over expectation counts as extreme
RELATIONSHIP_BITS = 6  # a value fits its row this much worse than it fits its column
STRONG_RELATIONSHIP_BITS = 10
ROW_RELATIONSHIP_BITS = (0, 5, 10, 20)  # thresholds on the same quantity summed over a row
RANGE_QUANTILES = (0.001, 0.999)  # numeric values outside this training range are unexpected
TOP_CATEGORIES = 5  # category shares tracked per categorical column
DIGIT_PATTERNS = 100  # the last two digits of a numeric value, at RESOLUTION

Statistics = dict[str, np.ndarray]


class CheckSpec(NamedTuple):
    name: str
    column: str
    increase_only: bool  # only an increase is a problem (a count of bad things)
    is_rate: bool  # a share of rows, so a floor of a few rows applies (see history)


@dataclass
class ColumnReference:
    """What a column looked like in training, for the standard checks."""

    coder: ColumnCoder
    histogram: np.ndarray  # share of training rows in each coder class
    low: float
    high: float
    seen_values: np.ndarray
    digit_bits: np.ndarray | None  # numeric columns: bits of each last-two-digits pattern
    digit_entropy: float

    @property
    def categorical(self) -> bool:
        return self.coder.categorical

    @property
    def top_classes(self) -> np.ndarray:
        return np.argsort(-self.histogram)[:TOP_CATEGORIES]

    def to_state(self) -> dict[str, Any]:
        return {**vars(self), "coder": self.coder.to_state()}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ColumnReference:
        return cls(**{**state, "coder": ColumnCoder.from_state(state["coder"])})


class BatchChecks:
    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)
        self.references: dict[str, ColumnReference] = {}
        self.specs: list[CheckSpec] = []

    @property
    def numeric_columns(self) -> list[str]:
        return [c for c in self.columns if not self.references[c].categorical]

    def fit(self, encoded: pd.DataFrame) -> BatchChecks:
        self.references = {c: _fit_reference(_values(encoded, c)) for c in self.columns}
        self.specs = self._build_specs()
        return self

    def _build_specs(self) -> list[CheckSpec]:
        """One entry per metric, in the order metrics() returns them."""
        specs = [CheckSpec("surprise", c, True, False) for c in self.columns]
        specs += _per_row_and_column("extreme values", self.columns)
        specs += [CheckSpec("digit pattern", c, False, False) for c in self.numeric_columns]
        specs += _per_row_and_column("broken relationship", self.columns)
        specs += _per_row_and_column("broken relationship (strong)", self.columns)
        specs += [
            CheckSpec(
                f"rows contradicting their own columns (> {bits} bits)", WHOLE_ROW, True, True
            )
            for bits in ROW_RELATIONSHIP_BITS
        ]
        for c in self.columns:
            specs.append(CheckSpec("null rate", c, False, True))
            specs.append(CheckSpec("unexpected values", c, True, True))
            specs.append(CheckSpec("distribution shift", c, True, False))
            if self.references[c].categorical:
                specs += [
                    CheckSpec(f"category share #{k}", c, False, True)
                    for k in self.references[c].top_classes
                ]
            else:
                specs.append(CheckSpec("mean", c, False, False))
                specs.append(CheckSpec("spread", c, False, False))
                specs.append(CheckSpec("median", c, False, False))
                specs.append(CheckSpec("zero rate", c, False, True))
        specs.append(CheckSpec("duplicate rows", WHOLE_ROW, True, True))
        return specs

    # ------------------------------------------------------------------ step 1: additive
    def statistics(self, encoded: pd.DataFrame, scores: Scores) -> Statistics:
        """Sums and counts over a chunk of encoded rows."""
        relationship = scores.relationship_bits
        row_relationship = relationship.sum(1)
        stats: Statistics = {
            "rows": _floats([len(encoded)]),
            "excess_sum": scores.excess.sum(0, dtype=np.float64),
            "variance_sum": scores.variance.sum(0, dtype=np.float64),
            "extreme": _count_over(scores.excess, EXTREME_EXCESS_BITS),
            "relationship": _count_over(relationship, RELATIONSHIP_BITS),
            "strong_relationship": _count_over(relationship, STRONG_RELATIONSHIP_BITS),
            "row_relationship": _floats(
                [(row_relationship > bits).sum() for bits in ROW_RELATIONSHIP_BITS]
            ),
        }

        digit_bits, digit_rows = [], []
        for c in self.numeric_columns:
            reference = self.references[c]
            values = _values(encoded, c)
            finite = values[np.isfinite(values)]
            over_entropy = reference.digit_bits[_digit_pattern(finite)] - reference.digit_entropy
            digit_bits.append(float(over_entropy.sum()))
            digit_rows.append(len(finite))
        stats["digit_bits"] = _floats(digit_bits)
        stats["digit_rows"] = _floats(digit_rows)

        present, unexpected, total, total_of_squares, zeros = [], [], [], [], []
        for c in self.columns:
            reference = self.references[c]
            values = _values(encoded, c)
            finite = values[np.isfinite(values)]
            classes, _ = reference.coder.transform(values)
            counts = np.bincount(classes, minlength=reference.coder.n_classes)
            stats[f"histogram|{c}"] = counts.astype(float)
            present.append(len(finite))
            if reference.categorical:
                unexpected.append((~np.isin(finite, reference.seen_values)).sum())
                total.append(0.0)
                total_of_squares.append(0.0)
                zeros.append(0.0)
            else:
                unexpected.append(((finite < reference.low) | (finite > reference.high)).sum())
                total.append(finite.sum())
                total_of_squares.append((finite * finite).sum())
                zeros.append((finite == 0).sum())
        stats["present"] = _floats(present)
        stats["unexpected"] = _floats(unexpected)
        stats["sum"] = _floats(total)
        stats["sum_of_squares"] = _floats(total_of_squares)
        stats["zeros"] = _floats(zeros)
        return stats

    # ------------------------------------------------------------------ step 2: one sample
    def sample_statistics(self, encoded: pd.DataFrame) -> Statistics:
        """Medians and the duplicate-row share. These need all rows in one place."""
        medians = [
            float(np.nanmedian(_values(encoded, c))) if encoded[c].notna().any() else np.nan
            for c in self.numeric_columns
        ]
        duplicates = float(encoded.duplicated().mean()) if len(encoded) else 0.0
        return {"median": _floats(medians), "duplicates": _floats([duplicates])}

    # ------------------------------------------------------------------ step 3: metrics
    def metrics(self, stats: Statistics, sample: Statistics) -> np.ndarray:
        """One value per entry of self.specs."""
        rows = max(float(stats["rows"][0]), 1.0)
        values = list(stats["excess_sum"] / np.sqrt(stats["variance_sum"] + 1e-6))
        values += list(stats["extreme"] / rows)
        has_digits = stats["digit_rows"] > 0
        values += list(
            np.where(has_digits, stats["digit_bits"] / np.maximum(stats["digit_rows"], 1), 0.0)
        )
        values += list(stats["relationship"] / rows)
        values += list(stats["strong_relationship"] / rows)
        values += list(stats["row_relationship"] / rows)

        numeric_index = 0
        for index, c in enumerate(self.columns):
            reference = self.references[c]
            shares = stats[f"histogram|{c}"] / rows
            present = stats["present"][index]
            values.append(1 - present / rows)
            values.append(stats["unexpected"][index] / present if present else 0.0)
            values.append(_divergence(shares, reference.histogram))
            if reference.categorical:
                values += [shares[k] for k in reference.top_classes]
                continue
            if present:
                mean = stats["sum"][index] / present
                variance = stats["sum_of_squares"][index] / present - mean * mean
                spread = float(np.sqrt(max(variance, 0.0)))
                zero_rate = stats["zeros"][index] / present
            else:
                mean = spread = zero_rate = np.nan
            values += [mean, spread, sample["median"][numeric_index], zero_rate]
            numeric_index += 1
        values.append(sample["duplicates"][0])
        return _floats(values)

    def to_state(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "references": {c: ref.to_state() for c, ref in self.references.items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BatchChecks:
        checks = cls(state["columns"])
        checks.references = {
            c: ColumnReference.from_state(ref) for c, ref in state["references"].items()
        }
        checks.specs = checks._build_specs()
        return checks


def merge_statistics(parts: list[Statistics]) -> Statistics:
    """Add up statistics computed on separate chunks of one batch. Order does not matter."""
    parts = [part for part in parts if part]
    if not parts:
        return {}
    return {key: sum(np.asarray(part[key], dtype=float) for part in parts) for key in parts[0]}


def _fit_reference(values: np.ndarray) -> ColumnReference:
    coder = ColumnCoder(RESOLUTION).fit(values)
    classes, _ = coder.transform(values)
    finite = values[np.isfinite(values)]
    low, high = (0.0, 0.0)
    if len(finite):
        low, high = (float(q) for q in np.quantile(finite, RANGE_QUANTILES))
    digit_bits, digit_entropy = None, 0.0
    if not coder.categorical:
        counts = np.bincount(_digit_pattern(finite), minlength=DIGIT_PATTERNS) + 0.5
        probability = counts / counts.sum()
        digit_bits = -np.log2(probability)
        digit_entropy = float((probability * digit_bits).sum())
    return ColumnReference(
        coder=coder,
        histogram=np.bincount(classes, minlength=coder.n_classes) / len(classes),
        low=low,
        high=high,
        seen_values=np.unique(finite),
        digit_bits=digit_bits,
        digit_entropy=digit_entropy,
    )


def _per_row_and_column(name: str, columns: list[str]) -> list[CheckSpec]:
    whole_row = [CheckSpec(name, WHOLE_ROW, True, True)]
    return whole_row + [CheckSpec(name, c, True, True) for c in columns]


def _count_over(per_value: np.ndarray, threshold: float) -> np.ndarray:
    """Rows with any column over the threshold, then the count per column."""
    return np.r_[(per_value.max(1) > threshold).sum(), (per_value > threshold).sum(0)].astype(float)


def _digit_pattern(values: np.ndarray) -> np.ndarray:
    return np.mod(np.round(values / RESOLUTION), DIGIT_PATTERNS).astype(int)


def _divergence(shares: np.ndarray, reference: np.ndarray) -> float:
    """Symmetric divergence between two histograms (the population stability index)."""
    smoothing = 1e-4
    return float(
        np.sum((shares - reference) * np.log((shares + smoothing) / (reference + smoothing)))
    )


def _values(encoded: pd.DataFrame, column: str) -> np.ndarray:
    return encoded[column].to_numpy(dtype=float)


def _floats(values: Any) -> np.ndarray:
    return np.array(values, dtype=float)
