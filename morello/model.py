"""A model of the whole table that says how surprising each value is, in bits.

Each column is predicted from the columns before it, in a fixed order. The code length of a
value is -log2 of the probability the model gave it: the size an arithmetic coder driven by
this model would spend on the value. A row whose columns contradict each other costs many more
bits than the model expected, and that excess is what the monitor watches.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .coder import ESCAPE_BITS, ColumnCoder

log = logging.getLogger(__name__)

RESOLUTION = 0.01  # numeric values are coded to this precision
MIN_PROBABILITY = 1e-6
UNSEEN_CLASS_BITS = -np.log2(MIN_PROBABILITY) + ESCAPE_BITS  # a class absent from training

BOOSTING_ROUNDS = 40
EARLY_STOPPING_ROUNDS = 4
VALIDATION_SHARE = 8  # the last 1/8 of the training rows decides when to stop boosting
LIGHTGBM_PARAMS = {
    "objective": "multiclass",
    "learning_rate": 0.12,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "lambda_l2": 5.0,
    "min_sum_hessian_in_leaf": 1.0,
    "feature_fraction": 0.8,
    "max_bin": 63,
    "verbose": -1,
    "seed": 0,
}

# A numeric column is coded relative to a grouping column when the group explains this much
# of its variance (eta squared).
GROUP_MIN_ETA_SQUARED = 0.5
GROUP_MIN_KEYS, GROUP_MAX_KEYS = 5, 300
GROUP_MIN_ROWS = 100
GROUP_MIN_ROWS_PER_KEY = 5
IQR_TO_SIGMA = 1.349  # interquartile range of a normal distribution, in standard deviations
TRIM_QUANTILES = (0.001, 0.999)  # rows outside these are left out of the linear fit


def available_threads() -> int:
    """CPU cores this process may use. LightGBM's own default oversubscribes in containers."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on macOS and Windows
        return max(1, os.cpu_count() or 1)


@dataclass
class LinearFit:
    """Least-squares prediction of a numeric column from the columns before it."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, earlier: np.ndarray) -> np.ndarray:
        standardized = (np.nan_to_num(earlier, nan=0.0) - self.mean) / self.scale
        return np.c_[standardized, np.ones(len(standardized))] @ self.coefficients


@dataclass
class GroupFit:
    """Median and spread of a numeric column within each value of one earlier column."""

    column_index: int
    by_key: dict[float, tuple[float, float]]  # group value -> (median, robust scale)
    overall: tuple[float, float]

    def standardize(self, values: np.ndarray, keys: np.ndarray) -> np.ndarray:
        median = np.full(len(values), self.overall[0])
        scale = np.full(len(values), self.overall[1])
        for key, (key_median, key_scale) in self.by_key.items():
            selected = keys == key
            median[selected], scale[selected] = key_median, key_scale
        return np.round((values - median) / scale, 2) + 0.0  # + 0.0 turns -0.0 into 0.0


@dataclass
class ColumnModel:
    """Everything needed to code one column given the columns before it."""

    coder: ColumnCoder
    linear: LinearFit | None
    code_residual: bool  # code the column minus its linear prediction
    group: GroupFit | None  # code the column standardized within a group
    booster: lgb.Booster | None  # None when the column has one class or no earlier columns
    prior: np.ndarray | None  # class frequencies, used when there is no booster
    class_map: np.ndarray  # coder class -> index among the classes seen in training, or -1
    extra_mean: np.ndarray  # mean extra bits per seen class
    extra_variance: np.ndarray
    marginal_bits: np.ndarray  # bits of each seen class from its plain frequency

    STATE_FIELDS = (
        "code_residual",
        "booster",
        "prior",
        "class_map",
        "extra_mean",
        "extra_variance",
        "marginal_bits",
    )

    def to_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {field: getattr(self, field) for field in self.STATE_FIELDS}
        state["coder"] = self.coder.to_state()
        state["linear"] = None if self.linear is None else vars(self.linear)
        state["group"] = None if self.group is None else vars(self.group)
        return state

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ColumnModel:
        return cls(
            coder=ColumnCoder.from_state(state["coder"]),
            linear=None if state["linear"] is None else LinearFit(**state["linear"]),
            group=None if state["group"] is None else GroupFit(**state["group"]),
            **{field: state[field] for field in cls.STATE_FIELDS},
        )


@dataclass
class Scores:
    """Per (row, column) results of scoring a batch."""

    bits: np.ndarray  # code length of the value given the columns before it
    excess: np.ndarray  # bits minus the code length the model expected for that row
    variance: np.ndarray  # the model's variance of that code length
    marginal_bits: np.ndarray  # code length from the value's plain frequency, ignoring the row

    @property
    def relationship_bits(self) -> np.ndarray:
        """How much worse the value fits its row than it fits the column on its own."""
        return self.bits - self.marginal_bits


class TableModel:
    """Predicts each column from the columns before it."""

    def __init__(self, columns: list[str], threads: int = 0) -> None:
        self.columns = list(columns)
        self.threads = threads or available_threads()
        self.column_models: list[ColumnModel] = []

    # ------------------------------------------------------------------ fitting
    def fit(self, table: pd.DataFrame) -> TableModel:
        """Train on a table whose columns are all numeric (see schema.TableSchema)."""
        self.column_models = []
        for index, name in enumerate(self.columns):
            column_model = self._fit_column(table, index)
            self.column_models.append(column_model)
            log.debug(
                "fitted %s: %d classes, %s, %s",
                name,
                len(column_model.extra_mean),
                "categorical" if column_model.coder.categorical else "numeric",
                self._coding_label(column_model),
            )
        return self

    def _fit_column(self, table: pd.DataFrame, index: int) -> ColumnModel:
        raw = self._values(table, index)
        earlier = self._earlier(table, index)
        coder = ColumnCoder(RESOLUTION).fit(raw)
        can_use_earlier = not coder.categorical and index > 0
        linear = _fit_linear(earlier, raw) if can_use_earlier else None
        group = self._fit_group(table, index) if can_use_earlier else None
        code_residual = False

        if group is not None:
            target = group.standardize(raw, self._values(table, group.column_index))
            coder = ColumnCoder(RESOLUTION).fit(target)
        elif linear is not None:
            residual = _round_to_resolution(raw - linear.predict(earlier))
            residual_coder = ColumnCoder(RESOLUTION).fit(residual)
            if _marginal_bits_per_value(residual_coder, residual) < _marginal_bits_per_value(
                coder, raw
            ):
                coder, code_residual, target = residual_coder, True, residual
            else:
                target = raw
        else:
            target = raw

        classes, extra_bits = coder.transform(target)
        seen = np.unique(classes)
        class_map = -np.ones(coder.n_classes, dtype=np.int32)
        class_map[seen] = np.arange(len(seen))
        labels = class_map[classes]
        extra_mean = np.array([extra_bits[labels == k].mean() for k in range(len(seen))])
        extra_variance = np.array([extra_bits[labels == k].var() for k in range(len(seen))])
        smoothed_counts = np.bincount(labels, minlength=len(seen)) + 0.5
        frequencies = smoothed_counts / smoothed_counts.sum()

        features = _with_prediction(earlier, linear)
        booster, prior = None, None
        if len(seen) == 1 or features.shape[1] == 0:
            prior = frequencies
        else:
            booster = self._train_booster(features, labels, n_classes=len(seen))

        return ColumnModel(
            coder=coder,
            linear=linear,
            code_residual=code_residual,
            group=group,
            booster=booster,
            prior=prior,
            class_map=class_map,
            extra_mean=extra_mean,
            extra_variance=extra_variance,
            marginal_bits=-np.log2(frequencies),
        )

    def _train_booster(self, features: np.ndarray, labels: np.ndarray, n_classes: int):
        n_validation = max(len(labels) // VALIDATION_SHARE, 1)
        train = lgb.Dataset(features[:-n_validation], labels[:-n_validation])
        validation = lgb.Dataset(features[-n_validation:], labels[-n_validation:], reference=train)
        params = dict(LIGHTGBM_PARAMS, num_class=n_classes, num_threads=self.threads)
        return lgb.train(
            params,
            train,
            num_boost_round=BOOSTING_ROUNDS,
            valid_sets=[validation],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )

    def _fit_group(self, table: pd.DataFrame, index: int) -> GroupFit | None:
        """Find the earlier column whose values best explain this one, if any does well enough."""
        values = self._values(table, index)
        present = ~np.isnan(values)
        best: tuple[float, int, np.ndarray] | None = None
        for candidate in range(index):
            keys = self._values(table, candidate)
            distinct = np.unique(keys[~np.isnan(keys)])
            if not (GROUP_MIN_KEYS <= len(distinct) <= GROUP_MAX_KEYS):
                continue
            usable = present & ~np.isnan(keys)
            if usable.sum() < GROUP_MIN_ROWS:
                continue
            eta_squared = _eta_squared(values[usable], np.searchsorted(distinct, keys[usable]))
            if eta_squared > GROUP_MIN_ETA_SQUARED and (best is None or eta_squared > best[0]):
                best = (eta_squared, candidate, distinct)
        if best is None:
            return None

        _, column_index, distinct = best
        keys = self._values(table, column_index)
        by_key = {}
        for key in distinct:
            in_group = values[present & (keys == key)]
            if len(in_group) >= GROUP_MIN_ROWS_PER_KEY:
                by_key[float(key)] = _median_and_scale(in_group)
        return GroupFit(column_index, by_key, _median_and_scale(values[present]))

    # ------------------------------------------------------------------ scoring
    def score(self, table: pd.DataFrame) -> Scores:
        n_rows, n_columns = len(table), len(self.columns)
        bits, excess, variance, marginal = (
            np.zeros((n_rows, n_columns), np.float32) for _ in range(4)
        )
        row = np.arange(n_rows)
        for index, column_model in enumerate(self.column_models):
            classes, extra_bits = column_model.coder.transform(self._target(table, index))
            labels = column_model.class_map[classes]
            probabilities = np.maximum(self._predict(table, index), MIN_PROBABILITY)

            class_cost = -np.log2(probabilities) + column_model.extra_mean[None, :]
            expected = (probabilities * class_cost).sum(1)
            second_moment = probabilities * (class_cost**2 + column_model.extra_variance[None, :])
            cost_variance = second_moment.sum(1) - expected**2

            seen = labels >= 0
            given_row = np.full(n_rows, UNSEEN_CLASS_BITS)
            given_row[seen] = -np.log2(probabilities[row[seen], labels[seen]])
            alone = np.full(n_rows, UNSEEN_CLASS_BITS)
            alone[seen] = column_model.marginal_bits[labels[seen]]

            bits[:, index] = given_row + extra_bits
            excess[:, index] = given_row + extra_bits - expected
            variance[:, index] = np.maximum(cost_variance, 0)
            marginal[:, index] = alone + extra_bits
        return Scores(bits, excess, variance, marginal)

    def _predict(self, table: pd.DataFrame, index: int) -> np.ndarray:
        column_model = self.column_models[index]
        if column_model.booster is None:
            return np.tile(column_model.prior, (len(table), 1))
        features = _with_prediction(self._earlier(table, index), column_model.linear)
        predicted = column_model.booster.predict(features, num_threads=self.threads)
        return predicted.reshape(len(table), -1)

    def _target(self, table: pd.DataFrame, index: int) -> np.ndarray:
        """What is coded: the column itself, its value within a group, or a residual."""
        column_model = self.column_models[index]
        values = self._values(table, index)
        if column_model.group is not None:
            keys = self._values(table, column_model.group.column_index)
            return column_model.group.standardize(values, keys)
        if column_model.code_residual:
            prediction = column_model.linear.predict(self._earlier(table, index))
            return _round_to_resolution(values - prediction)
        return values

    def _values(self, table: pd.DataFrame, index: int) -> np.ndarray:
        return table[self.columns[index]].to_numpy(dtype=np.float64)

    def _earlier(self, table: pd.DataFrame, index: int) -> np.ndarray:
        return table[self.columns[:index]].to_numpy(dtype=np.float64)

    @staticmethod
    def _coding_label(column_model: ColumnModel) -> str:
        if column_model.group is not None:
            return "standardized within a group"
        return "residual of a linear fit" if column_model.code_residual else "raw value"

    # ------------------------------------------------------------------ state
    def to_state(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "column_models": [column_model.to_state() for column_model in self.column_models],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], threads: int = 0) -> TableModel:
        model = cls(state["columns"], threads)
        model.column_models = [ColumnModel.from_state(s) for s in state["column_models"]]
        return model


def _fit_linear(earlier: np.ndarray, values: np.ndarray) -> LinearFit:
    """Ridge-stabilized least squares on rows inside the trim quantiles.

    The prediction is given to the trees as one more feature, which lets them learn exact
    arithmetic relationships such as total = price * quantity + tax.
    """
    features = np.nan_to_num(earlier, nan=0.0)
    target = np.nan_to_num(values, nan=0.0)
    low, high = np.quantile(target, TRIM_QUANTILES)
    keep = (target >= low) & (target <= high)
    mean = features[keep].mean(0)
    scale = features[keep].std(0) + 1e-9
    design = np.c_[(features[keep] - mean) / scale, np.ones(keep.sum())]
    gram = design.T @ design + 1e-8 * np.eye(design.shape[1])
    return LinearFit(mean, scale, np.linalg.solve(gram, design.T @ target[keep]))


def _with_prediction(earlier: np.ndarray, linear: LinearFit | None) -> np.ndarray:
    return earlier if linear is None else np.c_[earlier, linear.predict(earlier)]


def _round_to_resolution(values: np.ndarray) -> np.ndarray:
    return np.round(np.round(values / RESOLUTION) * RESOLUTION, 6) + 0.0


def _marginal_bits_per_value(coder: ColumnCoder, values: np.ndarray) -> float:
    """Average bits per value when each class is coded by its plain frequency."""
    classes, extra_bits = coder.transform(values)
    frequency = np.bincount(classes) / len(classes)
    return float(np.mean(-np.log2(frequency[classes]) + extra_bits))


def _eta_squared(values: np.ndarray, group_index: np.ndarray) -> float:
    """Share of the variance of `values` explained by group membership."""
    n_groups = int(group_index.max()) + 1
    counts = np.bincount(group_index, minlength=n_groups).astype(float)
    sums = np.bincount(group_index, weights=values, minlength=n_groups)
    group_means = sums / np.maximum(counts, 1)
    overall_mean = values.mean()
    between = np.sum(counts * (group_means - overall_mean) ** 2)
    total = max(np.sum((values - overall_mean) ** 2), 1e-12)
    return float(between / total)


def _median_and_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and a robust standard deviation (from the interquartile range)."""
    first_quartile, third_quartile = np.quantile(values, [0.25, 0.75])
    scale = max((third_quartile - first_quartile) / IQR_TO_SIGMA, 1e-9)
    return float(np.median(values)), float(scale)
