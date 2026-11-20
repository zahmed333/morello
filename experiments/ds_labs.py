"""Synthea synthetic patient records: lab and vital-sign results joined to the patient
and the encounter. One row per result. Jun 2019 - Jan 2020 (before the simulated outbreak)."""
import numpy as np, pandas as pd
from common import encode_categoricals, split_by_date

BATCH = 5000
COLS = ["gender", "age", "race", "enc_class", "code", "units", "value"]
CONTEXT, RES = [], {"age": 1.0, "value": 0.01}
SEGMENT = ("code", "value", 20)
PATIENT = ["gender", "age", "race"]
MAPS, PAIRS, RANGE = {}, {}, {}


def load(raw, phase='dev'):
    obs = pd.read_csv(f"{raw}/observations.csv")
    obs = obs[(obs.TYPE == "numeric") & (obs.DATE >= "2019-06-01") & (obs.DATE <= "2020-01-31")]
    pat = pd.read_csv(f"{raw}/patients.csv", usecols=["Id", "BIRTHDATE", "GENDER", "RACE"])
    enc = pd.read_csv(f"{raw}/encounters.csv", usecols=["Id", "ENCOUNTERCLASS"])
    df = obs.merge(pat, left_on="PATIENT", right_on="Id", how="left").merge(
        enc, left_on="ENCOUNTER", right_on="Id", how="left", suffixes=("", "_e"))
    out = pd.DataFrame({
        "batch": df.DATE.str[:10],
        "gender": df.GENDER, "race": df.RACE, "enc_class": df.ENCOUNTERCLASS,
        "age": np.floor((pd.to_datetime(df.DATE.str[:10]) - pd.to_datetime(df.BIRTHDATE)).dt.days / 365.25),
        "code": df.CODE, "units": df.UNITS, "value": pd.to_numeric(df.VALUE, errors="coerce"),
    })
    splits = split_by_date(out, {"train": ("2019-06-01", "2019-07-31"), "ref": ("2019-08-01", "2019-09-30"),
                                 "calib": ("2019-10-01", "2019-11-30"), "test": ("2019-12-01", "2020-01-31")})
    MAPS.update(encode_categoricals(splits, ["gender", "race", "enc_class", "code", "units"]))
    tr = splits["train"]
    PAIRS["code_units"] = set(zip(tr.code, tr.units))
    g = tr.groupby("code")["value"]
    RANGE["lo"], RANGE["hi"] = g.quantile(0.001).to_dict(), g.quantile(0.999).to_dict()
    for k in splits:
        splits[k] = splits[k][["batch"] + COLS].astype({c: "float64" for c in COLS})
    return splits


def extra_rules(df):
    lo = df["code"].map(RANGE["lo"]); hi = df["code"].map(RANGE["hi"])
    known = lo.notna()
    return {
        "neg_value_rate": (df["value"] < 0).mean(),
        "age_range_rate": ((df["age"] < 0) | (df["age"] > 110)).mean(),
        "code_units_pair_rate": np.mean([p not in PAIRS["code_units"] for p in zip(df.code, df.units)]),
        "value_outside_range_for_code_rate": ((df["value"] < lo) | (df["value"] > hi))[known].mean() if known.any() else 0.0,
    }


def scenarios(ci):
    weight = MAPS["code"].get("29463-7")          # body weight, kg

    def unit_change(d):                              # one lab switches kg -> lb
        d = d.copy(); sel = d["code"] == weight
        d.loc[sel, "value"] = (d.loc[sel, "value"] * 2.2046).round(1)
        return d, ["value"]

    def swapped_columns(d):
        d = d.copy(); d["age"], d["value"] = d["value"].values, d["age"].values
        return d, ["age", "value"]

    def default_fill(d):
        d = d.copy(); d["value"] = 0.0
        return d, ["value"]

    def scale_x100(d):
        d = d.copy(); d["value"] = d["value"] * 100
        return d, ["value"]

    def int_cast(d):
        d = d.copy(); d["value"] = d["value"].round(0)
        return d, ["value"]

    def enum_remap(d):
        d = d.copy(); d["gender"] = d["gender"].map(lambda v: {0.0: 1.0, 1.0: 0.0}.get(v, v))
        return d, ["gender"]

    return {
        "wrong_patient_join": ("join", PATIENT),
        "wrong_test_label_join": ("join", ["code", "units"]),
        "unit_change": ("rowwise", unit_change),
        "swapped_columns": ("rowwise", swapped_columns),
        "default_fill": ("rowwise", default_fill),
        "scale_x100": ("rowwise", scale_x100),
        "int_cast": ("rowwise", int_cast),
        "enum_remap": ("rowwise", enum_remap),
        "duplicate_rows": ("dup",),
        "missing_segment": ("missing", lambda d: ((d["enc_class"] == 1.0).to_numpy(), ["enc_class"])),
    }
