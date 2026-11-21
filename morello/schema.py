"""Turn an arbitrary table into numeric columns the model can read.

The schema is inferred once from the training rows and then applied unchanged to every batch,
so a batch is always encoded the way the model was trained.

Identifiers and free text have no values worth modelling, but their form is still checkable.
Such a column contributes its length and, for text, whether the value has a format seen in
training ("ORD-00412" and "ORD-7" share the format "A-9"). Empty columns are skipped.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

ISO_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?$"
)
DATETIME_SAMPLE_ROWS = 200
DATETIME_MIN_SHARE = 0.95  # share of values that must look like, and parse as, datetimes
NUMERIC_TEXT_MIN_SHARE = 0.99
IDENTIFIER_MIN_ROWS = 100
IDENTIFIER_MIN_DISTINCT_SHARE = 0.9  # nearly all-distinct integers are an identifier
FREE_TEXT_MIN_DISTINCT_SHARE = 0.5

FORMAT_MAX_CHARS = 40  # a format is read from the first characters of a value
FORMAT_MIN_SHARE = 0.001  # formats rarer than this in training are not "known"
FORMAT_MAX_KNOWN = 50

BOOLEAN, NUMBER, NUMERIC_TEXT, DATETIME, CATEGORY = "bool", "num", "numstr", "datetime", "cat"
TEXT_FORMAT, DIGITS = "text_format", "digits"


def as_datetime(column: pd.Series) -> pd.Series | None:
    """Return the column as datetimes if it is one, or is text that looks like one. Else None."""
    if pdt.is_datetime64_any_dtype(column):
        return column
    if pdt.is_bool_dtype(column) or pdt.is_numeric_dtype(column):
        return None
    sample = column.dropna().astype(str).head(DATETIME_SAMPLE_ROWS)
    if len(sample) == 0:
        return None
    looks_like_datetime = sample.map(lambda value: bool(ISO_DATETIME.match(value))).mean()
    if looks_like_datetime <= DATETIME_MIN_SHARE:
        return None
    parsed = pd.to_datetime(column, errors="coerce", format="ISO8601")
    if parsed.notna().sum() >= DATETIME_MIN_SHARE * column.notna().sum():
        return parsed
    return None


@dataclass
class ColumnPlan:
    """How one source column becomes one or more numeric features."""

    column: str
    kind: str
    features: list[str]
    reference: str | None = None  # datetime columns: the earlier datetime to measure from
    # category columns: value -> code. text_format columns: known format -> 1.0
    mapping: dict[str, float] = field(default_factory=dict)
    other_code: float = -1.0  # category columns: code for values outside the mapping


class TableSchema:
    def __init__(self, max_categories: int = 300) -> None:
        self.max_categories = max_categories
        self.plans: list[ColumnPlan] = []
        self.skipped: dict[str, str] = {}  # column -> reason
        self.format_only: dict[str, str] = {}  # column -> reason; checked for format and length

    @property
    def features(self) -> list[str]:
        return [feature for plan in self.plans for feature in plan.features]

    @property
    def unmodelled(self) -> dict[str, str]:
        """Columns whose values the model does not read: skipped, or checked for format only."""
        return {**self.skipped, **self.format_only}

    @property
    def source_column(self) -> dict[str, str]:
        """Feature name -> the source column it was derived from."""
        return {feature: plan.column for plan in self.plans for feature in plan.features}

    def fit(self, table: pd.DataFrame, exclude: tuple[str, ...] | list[str] = ()) -> TableSchema:
        self.plans, self.skipped, self.format_only = [], {}, {}
        first_datetime: str | None = None
        for name in table.columns:
            if name in exclude:
                continue
            column = table[name]
            if column.notna().sum() == 0:
                self.skipped[name] = "empty"
            elif pdt.is_bool_dtype(column):
                self.plans.append(ColumnPlan(name, BOOLEAN, [name]))
            elif pdt.is_numeric_dtype(column):
                if _is_identifier(column):
                    self.format_only[name] = "identifier"
                    self.plans.append(ColumnPlan(name, DIGITS, [f"{name}__length"]))
                else:
                    self.plans.append(ColumnPlan(name, NUMBER, [name]))
            elif as_datetime(column) is not None:
                features = [f"{name}__hour"]
                if first_datetime:
                    features.append(f"{name}__minus__{first_datetime}")
                self.plans.append(ColumnPlan(name, DATETIME, features, reference=first_datetime))
                first_datetime = first_datetime or name
            else:
                self._plan_text_column(name, column)
        return self

    def _plan_text_column(self, name: str, column: pd.Series) -> None:
        text = column.dropna().astype(str)
        if pd.to_numeric(text, errors="coerce").notna().mean() > NUMERIC_TEXT_MIN_SHARE:
            self.plans.append(ColumnPlan(name, NUMERIC_TEXT, [name]))
            return
        counts = text.value_counts()
        too_many = len(counts) > self.max_categories
        if too_many and len(counts) > FREE_TEXT_MIN_DISTINCT_SHARE * len(text):
            self.format_only[name] = "identifier or free text"
            formats = text_formats(text).value_counts(normalize=True)
            known = formats[formats >= FORMAT_MIN_SHARE].index[:FORMAT_MAX_KNOWN]
            features = [f"{name}__length", f"{name}__known_format"]
            mapping = dict.fromkeys(known, 1.0)
            self.plans.append(ColumnPlan(name, TEXT_FORMAT, features, mapping=mapping))
            return
        kept = counts.index[: self.max_categories]
        self.plans.append(
            ColumnPlan(
                name,
                CATEGORY,
                [name],
                mapping={value: float(code) for code, value in enumerate(kept)},
                other_code=float(len(kept)) if too_many else -1.0,
            )
        )

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        """Encode a batch. A column missing from the batch becomes all nulls."""
        encoded: dict[str, np.ndarray] = {}
        datetimes: dict[str, pd.Series] = {}
        for plan in self.plans:
            if plan.column not in table.columns:
                for feature in plan.features:
                    encoded[feature] = np.full(len(table), np.nan)
                continue
            column = table[plan.column]
            if plan.kind == BOOLEAN:
                encoded[plan.column] = column.astype("float64").to_numpy()
            elif plan.kind == NUMBER:
                encoded[plan.column] = _to_float(pd.to_numeric(column, errors="coerce"))
            elif plan.kind == NUMERIC_TEXT:
                text = column.astype(str).where(column.notna())
                encoded[plan.column] = _to_float(pd.to_numeric(text, errors="coerce"))
            elif plan.kind == DIGITS:
                whole = pd.to_numeric(column, errors="coerce").abs().clip(lower=1)
                encoded[f"{plan.column}__length"] = _to_float(np.floor(np.log10(whole)) + 1)
            elif plan.kind == TEXT_FORMAT:
                text = column.astype(str).where(column.notna())
                known = text_formats(text).map(plan.mapping).fillna(0.0)
                encoded[f"{plan.column}__length"] = _to_float(text.str.len())
                encoded[f"{plan.column}__known_format"] = _to_float(known.where(text.notna()))
            elif plan.kind == DATETIME:
                parsed = column
                if not pdt.is_datetime64_any_dtype(column):
                    parsed = pd.to_datetime(column, errors="coerce", format="ISO8601")
                datetimes[plan.column] = parsed
                encoded[f"{plan.column}__hour"] = _to_float(parsed.dt.hour)
                if plan.reference:
                    elapsed = (parsed - datetimes[plan.reference]).dt.total_seconds()
                    encoded[f"{plan.column}__minus__{plan.reference}"] = _to_float(elapsed)
            else:
                codes = column.astype(str).map(plan.mapping).fillna(plan.other_code)
                encoded[plan.column] = np.where(column.isna().to_numpy(), np.nan, _to_float(codes))
        return pd.DataFrame(encoded, index=table.index)[self.features]

    def to_state(self) -> dict[str, Any]:
        return {
            "max_categories": self.max_categories,
            "plans": [asdict(plan) for plan in self.plans],
            "skipped": self.skipped,
            "format_only": self.format_only,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TableSchema:
        schema = cls(state["max_categories"])
        schema.plans = [ColumnPlan(**plan) for plan in state["plans"]]
        schema.skipped = state["skipped"]
        schema.format_only = state.get("format_only", {})  # absent before format 3
        return schema


def text_formats(text: pd.Series) -> pd.Series:
    """Each value with runs of digits, capitals and small letters collapsed to 9, A and a."""
    formats = text.str.slice(0, FORMAT_MAX_CHARS)
    for pattern, symbol in ((r"\d+", "9"), (r"[A-Z]+", "A"), (r"[a-z]+", "a"), (r"\s+", " ")):
        formats = formats.str.replace(pattern, symbol, regex=True)
    return formats


def _is_identifier(column: pd.Series) -> bool:
    values = column.dropna()
    return bool(
        len(values) > IDENTIFIER_MIN_ROWS
        and values.nunique() > IDENTIFIER_MIN_DISTINCT_SHARE * len(values)
        and np.all(values == np.round(values))
    )


def _to_float(values: pd.Series) -> np.ndarray:
    return values.to_numpy(dtype="float64")
