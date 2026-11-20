"""NYC yellow taxi trips (real, public TLC parquet files), 2024."""
import numpy as np, pandas as pd
from common import split_by_date

BATCH = 5000
COLS = ["hour", "VendorID", "PULocationID", "DOLocationID", "passenger_count", "RatecodeID",
        "store_and_fwd_flag", "trip_distance", "duration_s", "payment_type", "fare_amount", "extra",
        "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge", "congestion_surcharge",
        "Airport_fee", "total_amount"]
CONTEXT, RES = ["dow"], {"duration_s": 1.0, "hour": 1.0}
PAY = ["payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
       "improvement_surcharge", "congestion_surcharge", "Airport_fee", "total_amount"]
PARTS = PAY[1:-1]
MONTHS = {"dev": dict(train=1, ref=2, calib=3, test=4), "holdout": dict(train=2, ref=3, calib=4, test=5)}


def _month(raw, month):
    df = pd.read_parquet(f"{raw}/yellow_2024-{month:02d}.parquet")
    t = df["tpep_pickup_datetime"]
    df = df[(t.dt.year == 2024) & (t.dt.month == month)].copy()
    df["batch"] = df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%d")
    df["dow"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["duration_s"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0.0, "Y": 1.0})
    return df[["batch"] + CONTEXT + COLS].astype({c: "float64" for c in CONTEXT + COLS})


def load(raw, phase="dev"):
    out = {}
    for k, mo in MONTHS[phase].items():
        df = _month(raw, mo)
        if k == "train":
            out[k] = df.sample(150_000, random_state=0).reset_index(drop=True)
        else:
            out[k] = pd.concat([g.sample(min(10_000, len(g)), random_state=1) for _, g in df.groupby("batch")]).reset_index(drop=True)
    return out


def extra_rules(df):
    return {"neg_duration_rate": (df["duration_s"] < 0).mean(),
            "neg_distance_rate": (df["trip_distance"] < 0).mean(),
            "total_mismatch_rate": ((df["total_amount"] - df[PARTS].sum(axis=1)).abs() > 0.011).mean()}


def scenarios(ci):
    def unit_change(d):
        d = d.copy(); d["trip_distance"] = (d["trip_distance"] * 1.609).round(2)
        return d, ["trip_distance"]

    def swapped_columns(d):
        d = d.copy(); d["PULocationID"], d["DOLocationID"] = d["DOLocationID"].values, d["PULocationID"].values
        return d, ["PULocationID", "DOLocationID"]

    def default_fill(d):
        d = d.copy(); d["total_amount"] = (d["total_amount"] - d["tip_amount"]).round(2); d["tip_amount"] = 0.0
        return d, ["tip_amount", "total_amount"]

    def timezone_shift(d):
        d = d.copy(); d["hour"] = (d["hour"] + 5) % 24
        return d, ["hour"]

    def scale_x100(d):
        d = d.copy(); d["fare_amount"] = d["fare_amount"] * 100
        return d, ["fare_amount"]

    def int_cast(d):
        d = d.copy(); d["trip_distance"] = d["trip_distance"].round(0)
        return d, ["trip_distance"]

    def enum_remap(d):
        d = d.copy(); d["payment_type"] = d["payment_type"].map(lambda v: {1.0: 2.0, 2.0: 1.0}.get(v, v))
        return d, ["payment_type"]

    return {
        "broken_join": ("join", PAY),
        "unit_change": ("rowwise", unit_change), "swapped_columns": ("rowwise", swapped_columns),
        "default_fill": ("rowwise", default_fill), "timezone_shift": ("rowwise", timezone_shift),
        "scale_x100": ("rowwise", scale_x100), "int_cast": ("rowwise", int_cast),
        "enum_remap": ("rowwise", enum_remap), "duplicate_rows": ("dup",),
        "missing_segment": ("missing", lambda d: ((d["VendorID"] == 1.0).to_numpy(), ["VendorID"])),
    }
