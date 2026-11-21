"""The saved format: round trips, valid JSON, and refusing files that do not belong."""

import gzip
import json

import numpy as np
import pytest

from conftest import break_relationship
from morello import Monitor, StateError, state


def test_a_loaded_monitor_gives_the_same_report(monitor, today, tmp_path):
    broken = break_relationship(today)
    expected = monitor.check(broken, update="never").to_dict()
    monitor.save(tmp_path)
    assert Monitor.load(tmp_path).check(broken, update="never").to_dict() == expected


def test_saved_files_are_valid_json_with_a_version(monitor, tmp_path):
    monitor.save(tmp_path)
    for kind in ("model", "history"):
        raw = gzip.decompress((tmp_path / f"monitor.{kind}.json.gz").read_bytes())
        document = json.loads(raw, parse_constant=_reject)  # NaN and Infinity are not JSON
        assert document["format"] == state.FORMAT_VERSION
        assert document["kind"] == kind


def _reject(token):
    raise AssertionError(f"{token} is not valid JSON")


def test_another_format_version_is_refused(monitor):
    document = json.loads(gzip.decompress(monitor.model_bytes()))
    document["format"] = state.FORMAT_VERSION + 1
    newer = gzip.compress(json.dumps(document).encode())
    with pytest.raises(StateError, match="format"):
        Monitor.from_bytes(newer)


def test_history_from_another_fit_is_refused(monitor, history):
    other = Monitor(sample_rows=200, max_train_rows=2000).fit(history, "date")
    with pytest.raises(StateError, match="different fit"):
        Monitor.from_bytes(monitor.model_bytes(), other.history_bytes())


def test_a_model_file_is_not_accepted_as_history(monitor):
    with pytest.raises(StateError, match="expected a saved history"):
        Monitor.from_bytes(monitor.model_bytes(), monitor.model_bytes())


def test_garbage_is_refused():
    with pytest.raises(StateError):
        Monitor.from_bytes(b"not a monitor")


def test_loading_a_missing_monitor_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no saved monitor"):
        Monitor.load(tmp_path)


def test_values_survive_encoding():
    value = {
        "array": np.array([1.5, np.nan, np.inf]),
        "integers": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "pair": (1, "two"),
        "by_number": {1.0: (2.0, 3.0)},
        "__looks_like_a_marker__": 1,
        "nothing": None,
        "flag": True,
    }
    restored = state.decode(json.loads(json.dumps(state.encode(value), allow_nan=False)))
    assert np.array_equal(restored["array"], value["array"], equal_nan=True)
    assert restored["integers"].dtype == np.int32
    assert np.array_equal(restored["integers"], value["integers"])
    assert restored["pair"] == (1, "two")
    assert restored["by_number"] == {1.0: (2.0, 3.0)}
    assert restored["__looks_like_a_marker__"] == 1
    assert restored["nothing"] is None and restored["flag"] is True
