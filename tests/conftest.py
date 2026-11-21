"""Shared test data. Fitting a monitor takes a few seconds, so fitted monitors are built once."""

import numpy as np
import pandas as pd
import pytest

from morello import Monitor

LAST_DAY = "2024-02-29"


def make_events(n_days: int = 60, rows_per_day: int = 800, seed: int = 0) -> pd.DataFrame:
    """Daily batches where `amount` depends on `kind`. `id` and `note` should be skipped."""
    rng = np.random.default_rng(seed)
    days = []
    for day in pd.date_range("2024-01-01", periods=n_days).strftime("%Y-%m-%d"):
        kind = rng.choice(["a", "b", "c"], rows_per_day)
        noise = rng.normal(0, 1, rows_per_day).round(2)
        days.append(
            pd.DataFrame(
                {
                    "date": day,
                    "id": [f"{day}-{i}" for i in range(rows_per_day)],
                    "kind": kind,
                    "amount": np.where(kind == "a", 10, 50) + noise,
                    "flag": kind == "c",
                    "note": None,
                }
            )
        )
    return pd.concat(days, ignore_index=True)


def make_orders(n_days: int = 62, rows_per_day: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Daily orders where `currency` is determined by `region`."""
    rng = np.random.default_rng(seed)
    currency = {"EU": "EUR", "US": "USD", "APAC": "JPY"}
    days = []
    for day in pd.date_range("2024-01-01", periods=n_days):
        region = rng.choice(["EU", "US", "APAC"], rows_per_day, p=[0.5, 0.35, 0.15])
        seconds = rng.integers(0, 86400, rows_per_day)
        days.append(
            pd.DataFrame(
                {
                    "order_ts": day + pd.to_timedelta(seconds, unit="s"),
                    "order_id": [f"{day:%Y%m%d}-{i}" for i in range(rows_per_day)],
                    "region": region,
                    "currency": pd.Series(region).map(currency).to_numpy(),
                    "amount": (
                        np.where(region == "US", 40, 25) + rng.normal(0, 3, rows_per_day)
                    ).round(2),
                    "items": rng.integers(1, 6, rows_per_day),
                }
            )
        )
    return pd.concat(days, ignore_index=True)


def break_relationship(batch: pd.DataFrame) -> pd.DataFrame:
    """Shuffle `amount` across rows: every column keeps its distribution, the rows stop fitting."""
    broken = batch.copy()
    broken["amount"] = broken["amount"].sample(frac=1, random_state=1).to_numpy()
    return broken


@pytest.fixture(scope="session")
def events() -> pd.DataFrame:
    return make_events()


@pytest.fixture(scope="session")
def history(events) -> pd.DataFrame:
    return events[events["date"] < LAST_DAY]


@pytest.fixture(scope="session")
def today(events) -> pd.DataFrame:
    return events[events["date"] == LAST_DAY]


@pytest.fixture(scope="session")
def fitted_bytes(history) -> tuple[bytes, bytes]:
    monitor = Monitor(sample_rows=800).fit(history, "date")
    return monitor.model_bytes(), monitor.history_bytes()


@pytest.fixture
def monitor(fitted_bytes) -> Monitor:
    """A fresh copy of the fitted monitor, so tests cannot affect each other."""
    return Monitor.from_bytes(*fitted_bytes)
