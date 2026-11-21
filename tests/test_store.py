"""Stores: what a reader sees while and after a writer writes."""

import os

import duckdb
import pandas as pd
import pytest

from morello import LocalStore, TableStore
from morello.sql import connect
from morello.store import open_store


def test_local_store_round_trip_leaves_no_temporary_file(tmp_path):
    store = LocalStore(tmp_path / "state")
    assert store.get("a") is None
    store.put("a", b"one")
    store.put("a", b"two")
    assert store.get("a") == b"two"
    assert os.listdir(tmp_path / "state") == ["a.json.gz"]


def test_open_store_picks_the_store_from_the_target(tmp_path):
    assert isinstance(open_store(str(tmp_path)), LocalStore)
    existing = LocalStore(tmp_path)
    assert open_store(existing) is existing


@pytest.fixture
def table_store():
    source = connect(duckdb.connect())
    return TableStore(source, "morello_state"), source


def test_table_store_reads_the_latest_version(table_store):
    store, source = table_store
    assert store.get("m") is None  # the table does not exist yet
    store.put("m", b"first")
    store.put("m", b"second")
    store.put("other", b"x")
    assert store.get("m") == b"second"
    versions = source.query("SELECT name, version FROM morello_state ORDER BY name, version")
    assert versions.to_dict("list") == {"name": ["m", "m", "other"], "version": [1, 2, 1]}


def test_table_store_splits_large_values_into_rows(table_store, monkeypatch):
    store, source = table_store
    monkeypatch.setattr("morello.store.CHUNK_CHARS", 8)
    data = bytes(range(40))
    store.put("m", data)
    assert store.get("m") == data
    assert source.query("SELECT COUNT(*) AS n FROM morello_state")["n"][0] > 1


def test_table_store_skips_a_version_that_is_still_being_written(table_store):
    store, source = table_store
    store.put("m", b"complete")
    half_written = pd.DataFrame(
        {"name": "m", "version": 2, "part": [0], "n_parts": 2, "chunk": ["AAAA"]}
    )
    source.append("morello_state", half_written)
    assert store.get("m") == b"complete"
