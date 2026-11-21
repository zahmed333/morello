"""Monitor: learn what normal batches of a table look like, then check new batches."""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import state
from .checks import WHOLE_ROW, BatchChecks, Statistics
from .history import HistoryEntry, SegmentHistory, deviation_from, is_weekend
from .model import Scores, TableModel
from .report import ALERT, OK, WARMING_UP, Alert, Report
from .schema import TableSchema, as_datetime
from .store import Store, open_store

log = logging.getLogger(__name__)

MIN_FIT_BATCHES = 20  # batches of history needed to train and calibrate
WHOLE_TABLE = ""  # the segment name of the table as a whole
UPDATE_MODES = ("auto", "always", "never")
MAX_ALERT_RATIO = 1e6
EXAMPLE_ROWS, EXAMPLE_COLUMNS, EXAMPLE_FIELDS = 3, 2, 7


@dataclass
class TableConfig:
    """How a monitor reads its table in a warehouse. Set by morello.table.fit_table."""

    table: str
    batch_col: str
    as_date: bool = True
    where: str | None = None
    slice_col: str | None = None
    slices: list[str] = field(default_factory=list)
    slice_sample_rows: int = 2000
    score_rows: int | None = None


def batch_keys(column: pd.Series, freq: str = "D") -> pd.Series:
    """Batch key per row: the calendar period for datetime columns, the value itself otherwise."""
    as_dates = as_datetime(column)
    if as_dates is not None:
        return as_dates.dt.to_period(freq).astype(str).where(as_dates.notna())
    return column.astype(str).where(column.notna())


class Monitor:
    """Learns a table from clean history and checks new batches against it.

    margin          a check alerts when it moves more than margin x the largest change seen on
                    any earlier clean batch
    window          recent clean batches that define what a check usually reads
    min_history     clean batches needed before alerts switch on
    sample_rows     rows scored per batch; larger batches are sampled. Keep it the same for
                    calibration and checking, because how much a check moves depends on it
    max_train_rows  rows used to train the model
    max_categories  distinct values kept per text column
    threads         CPU threads for training and scoring; 0 uses all available cores
    seed            seed for row sampling
    """

    def __init__(
        self,
        margin: float = 3.0,
        window: int = 28,
        min_history: int = 14,
        sample_rows: int = 5000,
        max_train_rows: int = 150_000,
        max_categories: int = 300,
        threads: int = 0,
        seed: int = 0,
    ) -> None:
        self.margin = margin
        self.window = window
        self.min_history = min_history
        self.sample_rows = sample_rows
        self.max_train_rows = max_train_rows
        self.max_categories = max_categories
        self.threads = threads
        self.seed = seed

        # set by fit_model
        self.model_id = ""
        self.batch_col: str | None = None
        self.freq = "D"
        self.schema = TableSchema(max_categories)
        self.model = TableModel([], threads)
        self.checks = BatchChecks([])
        self.table_config: TableConfig | None = None

        # changes after every check
        self.histories: dict[str, SegmentHistory] = {}
        self.last_checked: str | None = None  # last batch a table run has processed

    # ------------------------------------------------------------------ fitting
    def fit(
        self, table: pd.DataFrame, batch_col: str, freq: str = "D", train_fraction: float = 0.5
    ) -> Monitor:
        """Learn from a history table that is believed to be clean.

        The earliest `train_fraction` of batches trains the model. The remaining batches are
        scored out-of-sample, in order, to learn how much every check normally moves.
        """
        keys = batch_keys(table[batch_col], freq)
        table, keys = table[keys.notna()], keys[keys.notna()]
        batches = sorted(keys.unique())
        if len(batches) < MIN_FIT_BATCHES:
            raise ValueError(
                f"need at least {MIN_FIT_BATCHES} batches of history to calibrate alerts, "
                f"got {len(batches)}"
            )
        n_train = max(1, int(len(batches) * train_fraction))
        self.fit_model(table[keys.isin(set(batches[:n_train]))], batch_col, freq)
        for batch in batches[n_train:]:
            self.calibrate(table[keys == batch], batch=batch)
        return self

    def fit_model(
        self, train: pd.DataFrame, batch_col: str | None = None, freq: str = "D"
    ) -> Monitor:
        """Train on rows from the training batches. Follow with calibrate() for later batches."""
        self.batch_col, self.freq = batch_col, freq
        if len(train) > self.max_train_rows:
            train = train.sample(self.max_train_rows, random_state=self.seed)
        train = train.sample(frac=1.0, random_state=self.seed)

        self.schema = TableSchema(self.max_categories).fit(
            train, exclude=self._columns_to_exclude(train)
        )
        encoded = self.schema.transform(train)
        self.model = TableModel(self.schema.features, self.threads).fit(encoded)
        self.checks = BatchChecks(self.schema.features).fit(encoded)
        self.model_id = uuid.uuid4().hex
        self.histories, self.last_checked = {}, None
        log.info(
            "trained on %d rows, %d columns; skipped %s",
            len(train),
            len(self.columns),
            self.schema.skipped or "none",
        )
        return self

    def _columns_to_exclude(self, train: pd.DataFrame) -> list[str]:
        """The batch column is left out of the model unless it also carries a time of day."""
        if self.batch_col is None or self.batch_col not in train.columns:
            return []
        as_dates = as_datetime(train[self.batch_col])
        has_time_of_day = as_dates is not None and bool((as_dates.dt.hour != 0).any())
        return [] if has_time_of_day else [self.batch_col]

    def calibrate(self, table: pd.DataFrame, batch: str, segment: str = WHOLE_TABLE) -> Report:
        """Record a clean batch so the monitor learns how much every check normally moves."""
        return self.check(table, batch=batch, update="always", segment=segment)

    # ------------------------------------------------------------------ checking
    def check(
        self,
        table: pd.DataFrame,
        batch: str | None = None,
        update: str = "auto",
        segment: str = WHOLE_TABLE,
    ) -> Report:
        """Check one batch.

        update  "auto" learns the batch when it does not alert, "always" learns it regardless,
                "never" leaves the monitor unchanged.
        """
        if batch is None and self.batch_col in table.columns:
            keys = batch_keys(table[self.batch_col], self.freq).dropna()
            batch = keys.iloc[0] if len(keys) else None
        if len(table) > self.sample_rows:
            table = table.sample(self.sample_rows, random_state=self.seed)
        encoded = self.schema.transform(table)
        stats, scores = self.statistics(encoded)
        sample = self.checks.sample_statistics(encoded)
        return self.evaluate(stats, sample, batch, update, segment, rows=table, scores=scores)

    def statistics(self, encoded: pd.DataFrame) -> tuple[Statistics, Scores]:
        """Additive statistics of a chunk of encoded rows, and the per-row scores behind them."""
        scores = self.model.score(encoded)
        return self.checks.statistics(encoded, scores), scores

    def evaluate(
        self,
        stats: Statistics,
        sample: Statistics,
        batch: str | None,
        update: str = "auto",
        segment: str = WHOLE_TABLE,
        rows: pd.DataFrame | None = None,
        scores: Scores | None = None,
    ) -> Report:
        """Turn statistics into a report.

        Use this directly when the statistics were computed elsewhere, for example on a Spark
        cluster. `rows` and `scores`, when given, supply example rows for the report.
        """
        if update not in UPDATE_MODES:
            raise ValueError(f"update must be one of {UPDATE_MODES}, got {update!r}")
        batch = None if batch is None else str(batch)
        rows_checked = int(stats["rows"][0])
        metrics = self.checks.metrics(stats, sample)
        history = self.histories.get(segment, SegmentHistory())
        weekend = is_weekend(batch)
        usual = history.usual(weekend, self.window)

        entry = HistoryEntry(batch=batch, weekend=weekend, metrics=metrics, deviation=None)
        report = Report(
            batch=batch,
            status=WARMING_UP,
            rows_checked=rows_checked,
            history_batches=len(history.entries),
            segment=segment,
            skipped_columns=dict(self.schema.skipped),
            history_entry=entry,
        )
        if usual is not None:
            specs = self.checks.specs
            increase_only = np.array([spec.increase_only for spec in specs])
            entry.deviation = deviation_from(metrics, usual, increase_only)
            if history.calibrated_batches >= self.min_history:
                is_rate = np.array([spec.is_rate for spec in specs])
                ratio = history.ratio_to_largest(entry.deviation, is_rate, rows_checked)
                alerting = np.where(ratio > self.margin)[0]
                alerting = alerting[np.argsort(-ratio[alerting])]  # largest ratio first
                report.status = ALERT if len(alerting) else OK
                report.alerts = [
                    Alert(
                        check=specs[i].name,
                        column=self._source(specs[i].column),
                        value=float(metrics[i]),
                        usual=float(usual[i]),
                        ratio=float(min(ratio[i], MAX_ALERT_RATIO)),
                    )
                    for i in alerting
                ]
                report.suspect_columns = _unique(
                    alert.column for alert in report.alerts if alert.column != WHOLE_ROW
                )
                if rows is not None and scores is not None and len(alerting):
                    report.examples = self._examples(rows, scores, alerting)

        if update == "always" or (update == "auto" and report.status != ALERT):
            self._record(entry, segment)
        return report

    def accept(self, report: Report) -> None:
        """Tell the monitor an alerting batch was fine after all, so it widens its tolerance."""
        if report.history_entry is not None:
            self._record(report.history_entry, report.segment)

    def _record(self, entry: HistoryEntry, segment: str) -> None:
        self.histories.setdefault(segment, SegmentHistory()).record(entry, self.window)

    def _source(self, feature: str) -> str:
        return self.schema.source_column.get(feature, feature)

    def _examples(
        self, rows: pd.DataFrame, scores: Scores, alerting: np.ndarray
    ) -> dict[str, list[dict[str, Any]]]:
        """The most suspicious rows for the first few alerting columns."""
        examples: dict[str, list[dict[str, Any]]] = {}
        relationship = scores.relationship_bits
        for i in alerting:
            spec = self.checks.specs[i]
            if spec.column == WHOLE_ROW or len(examples) >= EXAMPLE_COLUMNS:
                continue
            source = self._source(spec.column)
            if source in examples:
                continue
            j = self.columns.index(spec.column)
            per_row = relationship[:, j] if "relationship" in spec.name else scores.excess[:, j]
            worst = rows.iloc[np.argsort(-per_row)[:EXAMPLE_ROWS]]
            others = [c for c in rows.columns if c != source and c not in self.schema.skipped]
            shown = [source] + others[:EXAMPLE_FIELDS]
            examples[source] = [{c: _plain(row[c]) for c in shown} for _, row in worst.iterrows()]
        return examples

    # ------------------------------------------------------------------ reading the monitor
    @property
    def columns(self) -> list[str]:
        """The numeric features the model reads, in model order."""
        return self.model.columns

    @property
    def history(self) -> list[HistoryEntry]:
        """Clean batches recorded for the table as a whole."""
        return self.histories.get(WHOLE_TABLE, SegmentHistory()).entries

    @property
    def batches_until_ready(self) -> int:
        """Clean batches still needed before alerts switch on."""
        calibrated = self.histories.get(WHOLE_TABLE, SegmentHistory()).calibrated_batches
        return max(0, self.min_history - calibrated)

    # ------------------------------------------------------------------ saving and loading
    SETTINGS = (
        "margin",
        "window",
        "min_history",
        "sample_rows",
        "max_train_rows",
        "max_categories",
        "seed",
    )

    def model_bytes(self) -> bytes:
        """Everything fixed at fit time."""
        return state.pack(
            state.MODEL,
            {
                "model_id": self.model_id,
                "settings": {name: getattr(self, name) for name in self.SETTINGS},
                "batch_col": self.batch_col,
                "freq": self.freq,
                "schema": self.schema.to_state(),
                "model": self.model.to_state(),
                "checks": self.checks.to_state(),
                "table_config": None if self.table_config is None else asdict(self.table_config),
            },
        )

    def history_bytes(self) -> bytes:
        """What changes after each check."""
        return state.pack(
            state.HISTORY,
            {
                "model_id": self.model_id,
                "histories": {seg: hist.to_state() for seg, hist in self.histories.items()},
                "last_checked": self.last_checked,
            },
        )

    @classmethod
    def from_bytes(
        cls, model_bytes: bytes, history_bytes: bytes | None = None, threads: int = 0
    ) -> Monitor:
        saved = state.unpack(model_bytes, state.MODEL)
        monitor = cls(threads=threads, **saved["settings"])
        monitor.model_id = saved["model_id"]
        monitor.batch_col, monitor.freq = saved["batch_col"], saved["freq"]
        monitor.schema = TableSchema.from_state(saved["schema"])
        monitor.model = TableModel.from_state(saved["model"], threads)
        monitor.checks = BatchChecks.from_state(saved["checks"])
        if saved["table_config"] is not None:
            monitor.table_config = TableConfig(**saved["table_config"])
        if history_bytes is not None:
            history = state.unpack(history_bytes, state.HISTORY)
            if history["model_id"] != monitor.model_id:
                raise state.StateError(
                    "the saved history belongs to a different fit of the model. Fit the monitor "
                    "again, or restore the model and history from the same save."
                )
            monitor.histories = {
                seg: SegmentHistory.from_state(hist) for seg, hist in history["histories"].items()
            }
            monitor.last_checked = history["last_checked"]
        return monitor

    def save(self, target: str | Store, key: str = "monitor", history_only: bool = False) -> None:
        """Save to a folder, a cloud URL (needs fsspec) or a store object.

        The model is written once. The small history is rewritten after every check, so pass
        history_only=True after checking.
        """
        store = open_store(target)
        if not history_only:
            store.put(f"{key}.model", self.model_bytes())
        store.put(f"{key}.history", self.history_bytes())

    @classmethod
    def load(cls, target: str | Store, key: str = "monitor", threads: int = 0) -> Monitor:
        store = open_store(target)
        model_bytes = store.get(f"{key}.model")
        if model_bytes is None:
            raise FileNotFoundError(f"no saved monitor named {key!r} in {target!r}")
        return cls.from_bytes(model_bytes, store.get(f"{key}.history"), threads)


def _unique(items) -> list:
    """Distinct items in first-seen order."""
    return list(dict.fromkeys(items))


def _plain(value: Any) -> Any:
    """A cell value as a plain Python value, for reports and JSON."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else str(value)
