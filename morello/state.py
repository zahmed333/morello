"""Save and load a Monitor without pickle.

The state is gzip-compressed JSON. Only the four morello classes, numpy arrays and LightGBM
model text can be reconstructed from it, so loading a file cannot run arbitrary code."""
import gzip, json
import numpy as np
import lightgbm as lgb

FORMAT = 1
HISTORY_FIELDS = ("histories", "runmax", "n_valid", "cursor")


def _classes():
    from .model import TableModel, ColumnCoder
    from .schema import Schema
    from .monitor import Monitor
    return {c.__name__: c for c in (TableModel, ColumnCoder, Schema, Monitor)}


def _enc(o, skip=()):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, float)):
        return o
    if isinstance(o, np.generic):
        return _enc(o.item())
    if isinstance(o, np.ndarray):
        return {"__nd__": o.tolist(), "dtype": str(o.dtype)}
    if isinstance(o, tuple):
        return {"__tuple__": [_enc(x) for x in o]}
    if isinstance(o, list):
        return [_enc(x) for x in o]
    if isinstance(o, dict):
        if all(isinstance(k, str) for k in o):
            return {"__map__": {k: _enc(v) for k, v in o.items()}}
        return {"__dict__": [[_enc(k), _enc(v)] for k, v in o.items()]}
    if isinstance(o, lgb.Booster):
        return {"__lgb__": o.model_to_string()}
    name = type(o).__name__
    if name in _classes():
        return {"__obj__": name, "state": {k: _enc(v) for k, v in vars(o).items() if k not in skip}}
    raise TypeError(f"cannot save object of type {name}")


def _dec(o):
    if isinstance(o, list):
        return [_dec(x) for x in o]
    if not isinstance(o, dict):
        return o
    if "__nd__" in o:
        return np.array(o["__nd__"], dtype=o["dtype"])
    if "__tuple__" in o:
        return tuple(_dec(x) for x in o["__tuple__"])
    if "__map__" in o:
        return {k: _dec(v) for k, v in o["__map__"].items()}
    if "__dict__" in o:
        return {_dec(k): _dec(v) for k, v in o["__dict__"]}
    if "__lgb__" in o:
        return lgb.Booster(model_str=o["__lgb__"])
    if "__obj__" in o:
        cls = _classes().get(o["__obj__"])
        if cls is None:
            raise ValueError(f"unknown class in saved state: {o['__obj__']}")
        obj = cls.__new__(cls)
        obj.__dict__.update({k: _dec(v) for k, v in o["state"].items()})
        return obj
    raise ValueError("unrecognised structure in saved state")


def _pack(payload):
    return gzip.compress(json.dumps({"format": FORMAT, "payload": payload}).encode("utf-8"), compresslevel=6)


def _unpack(data):
    doc = json.loads(gzip.decompress(data).decode("utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"unsupported state format {doc.get('format')}")
    return doc["payload"]


def dump_model(mon):
    """Everything fixed at fit time (large)."""
    return _pack(_enc(mon, skip=HISTORY_FIELDS))


def dump_history(mon):
    """What changes after each check (small)."""
    return _pack(_enc({k: getattr(mon, k) for k in HISTORY_FIELDS}))


def load_monitor(model_bytes, history_bytes=None):
    mon = _dec(_unpack(model_bytes))
    hist = _dec(_unpack(history_bytes)) if history_bytes else {}
    for k in HISTORY_FIELDS:
        setattr(mon, k, hist.get(k, {}))
    return mon


def pack_stats(st):
    return json.dumps({k: np.asarray(v, dtype=float).tolist() for k, v in st.items()}).encode("utf-8")


def unpack_stats(data):
    return {k: np.array(v, dtype=float) for k, v in json.loads(bytes(data).decode("utf-8")).items()}
