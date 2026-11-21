"""The saved format of a monitor.

A saved monitor is two gzip-compressed JSON documents:

- the model     everything fixed at fit time. Large, written once.
- the history   what changes after each check. Small, rewritten after every run.

Each document carries a format version and its kind, and both carry the id of the fit they
belong to, so a history is never applied to a model it was not built from. Loading only builds
numbers, strings, arrays and LightGBM models from the file. It cannot run code, unlike pickle.

Format history:
  1  internal attribute dump (0.1 to 0.4)
  2  explicit fields per class; valid JSON (non-finite numbers are written as strings)
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import lightgbm as lgb
import numpy as np

FORMAT_VERSION = 2
MODEL, HISTORY = "model", "history"

_ARRAY, _TUPLE, _PAIRS, _BOOSTER, _FLOAT = "__array__", "__tuple__", "__pairs__", "__lgb__", "__f__"


class StateError(ValueError):
    """A saved monitor could not be read, or its parts do not belong together."""


def encode(value: Any) -> Any:
    """Turn a state value into something json.dumps accepts. The inverse of decode()."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {_FLOAT: repr(value)}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, np.ndarray):
        return {_ARRAY: [encode(item) for item in value.tolist()], "dtype": str(value.dtype)}
    if isinstance(value, tuple):
        return {_TUPLE: [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) and not key.startswith("__") for key in value):
            return {key: encode(item) for key, item in value.items()}
        return {_PAIRS: [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, lgb.Booster):
        return {_BOOSTER: value.model_to_string()}
    raise TypeError(f"cannot save a value of type {type(value).__name__}")


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if _FLOAT in value:
        return float(value[_FLOAT])
    if _ARRAY in value:
        return np.array(decode(value[_ARRAY]), dtype=value["dtype"])
    if _TUPLE in value:
        return tuple(decode(item) for item in value[_TUPLE])
    if _PAIRS in value:
        return {decode(key): decode(item) for key, item in value[_PAIRS]}
    if _BOOSTER in value:
        return lgb.Booster(model_str=value[_BOOSTER])
    return {key: decode(item) for key, item in value.items()}


def pack(kind: str, state: dict[str, Any]) -> bytes:
    document = {"format": FORMAT_VERSION, "kind": kind, "state": encode(state)}
    return gzip.compress(json.dumps(document, allow_nan=False).encode("utf-8"), compresslevel=6)


def unpack(data: bytes, kind: str) -> dict[str, Any]:
    try:
        document = json.loads(gzip.decompress(data).decode("utf-8"))
    except (OSError, ValueError) as error:
        raise StateError(f"not a saved morello {kind}: {error}") from error
    if document.get("format") != FORMAT_VERSION:
        raise StateError(
            f"saved {kind} has format {document.get('format')!r}; this version of morello reads "
            f"format {FORMAT_VERSION}. Fit the monitor again."
        )
    if document.get("kind") != kind:
        raise StateError(f"expected a saved {kind}, found a {document.get('kind')!r}")
    return decode(document["state"])


def pack_statistics(statistics: dict[str, np.ndarray]) -> bytes:
    """Statistics of one chunk of rows, for sending between machines."""
    as_lists = {key: np.asarray(value, dtype=float).tolist() for key, value in statistics.items()}
    return json.dumps(as_lists).encode("utf-8")


def unpack_statistics(data: bytes) -> dict[str, np.ndarray]:
    as_lists = json.loads(bytes(data).decode("utf-8"))
    return {key: np.array(value, dtype=float) for key, value in as_lists.items()}
