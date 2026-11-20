"""NYC EMS incident dispatch records (real, public), Jan-Apr 2024. One row per incident."""
import glob
import numpy as np, pandas as pd
from common import encode_categoricals, split_by_date

BATCH = 5000
CALL = ["initial_call_type", "initial_severity", "final_call_type", "final_severity",
        "borough", "dispatch_area", "zipcode", "precinct"]
RESPONSE = ["held", "valid_dispatch", "dispatch_s", "valid_incident", "travel_s", "response_s",
            "to_hosp_s", "hosp_travel_s", "close_s", "disposition", "reopen", "special_event",
            "standby", "transfer"]
COLS = ["hour"] + CALL + RESPONSE
CONTEXT = ["dow"]
RES = {c: 1.0 for c in ["hour", "dispatch_s", "travel_s", "response_s", "to_hosp_s", "hosp_travel_s", "close_s"]}
PAIRS = {}


RANGES = {
    'dev': {'train': ('2024-01-01', '2024-01-31'), 'ref': ('2024-02-01', '2024-02-29'), 'calib': ('2024-03-01', '2024-03-31'), 'test': ('2024-04-01', '2024-04-30')},
    'holdout': {'train': ('2024-02-01', '2024-02-29'), 'ref': ('2024-03-01', '2024-03-31'), 'calib': ('2024-04-01', '2024-04-30'), 'test': ('2024-05-01', '2024-05-31')},
}


def load(raw, phase='dev'):
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(f"{raw}/ems_*.csv"))], ignore_index=True)
    t = {c: pd.to_datetime(df[c], errors="coerce") for c in
         ["incident_datetime", "first_on_scene_datetime", "first_to_hosp_datetime",
          "first_hosp_arrival_datetime", "incident_close_datetime"]}
    sec = lambda a, b: (t[a] - t[b]).dt.total_seconds()
    out = pd.DataFrame({
        "batch": t["incident_datetime"].dt.strftime("%Y-%m-%d"),
        "dow": t["incident_datetime"].dt.dayofweek.astype(float),
        "hour": t["incident_datetime"].dt.hour.astype(float),
        "initial_call_type": df.initial_call_type, "initial_severity": df.initial_severity_level_code,
        "final_call_type": df.final_call_type, "final_severity": df.final_severity_level_code,
        "borough": df.borough, "dispatch_area": df.incident_dispatch_area,
        "zipcode": pd.to_numeric(df.zipcode, errors="coerce"),
        "precinct": pd.to_numeric(df.policeprecinct, errors="coerce"),
        "held": df.held_indicator, "valid_dispatch": df.valid_dispatch_rspns_time_indc,
        "dispatch_s": pd.to_numeric(df.dispatch_response_seconds_qy, errors="coerce"),
        "valid_incident": df.valid_incident_rspns_time_indc,
        "travel_s": pd.to_numeric(df.incident_travel_tm_seconds_qy, errors="coerce"),
        "response_s": pd.to_numeric(df.incident_response_seconds_qy, errors="coerce"),
        "to_hosp_s": sec("first_to_hosp_datetime", "first_on_scene_datetime"),
        "hosp_travel_s": sec("first_hosp_arrival_datetime", "first_to_hosp_datetime"),
        "close_s": sec("incident_close_datetime", "incident_datetime"),
        "disposition": pd.to_numeric(df.incident_disposition_code, errors="coerce"),
        "reopen": df.reopen_indicator, "special_event": df.special_event_indicator,
        "standby": df.standby_indicator, "transfer": df.transfer_indicator,
    })
    splits = split_by_date(out, RANGES[phase])
    encode_categoricals(splits, ["initial_call_type", "final_call_type", "borough", "dispatch_area", "held",
                                 "valid_dispatch", "valid_incident", "reopen", "special_event", "standby", "transfer"])
    tr = splits["train"]
    PAIRS["borough_zip"] = set(zip(tr.borough, tr.zipcode))
    PAIRS["borough_area"] = set(zip(tr.borough, tr.dispatch_area))
    for k in splits:
        splits[k] = splits[k][["batch"] + CONTEXT + COLS].astype({c: "float64" for c in CONTEXT + COLS})
    return splits


def extra_rules(df):
    out = {f"neg_rate|{c}": (df[c] < 0).mean() for c in ["dispatch_s", "travel_s", "response_s", "to_hosp_s", "hosp_travel_s", "close_s"]}
    full = df[["dispatch_s", "travel_s", "response_s"]].notna().all(axis=1)
    out["response_sum_mismatch_rate"] = ((df["response_s"] - df["dispatch_s"] - df["travel_s"]).abs() > 1)[full].mean() if full.any() else 0.0
    zz = df["zipcode"].notna()
    out["borough_zip_pair_rate"] = np.mean([p not in PAIRS["borough_zip"] for p in zip(df.borough[zz], df.zipcode[zz])]) if zz.any() else 0.0
    out["borough_area_pair_rate"] = np.mean([p not in PAIRS["borough_area"] for p in zip(df.borough, df.dispatch_area)])
    return out


def scenarios(ci):
    def unit_change(d):                              # seconds -> minutes on one duration
        d = d.copy(); d["close_s"] = (d["close_s"] / 60).round(2)
        return d, ["close_s"]

    def swapped_columns(d):                          # two duration columns mapped the wrong way round
        d = d.copy(); d["dispatch_s"], d["travel_s"] = d["travel_s"].values, d["dispatch_s"].values
        return d, ["dispatch_s", "travel_s"]

    def default_fill(d):
        d = d.copy(); d["response_s"] = d["response_s"] - d["dispatch_s"]; d["dispatch_s"] = d["dispatch_s"] * 0
        return d, ["dispatch_s", "response_s"]

    def timezone_shift(d):
        d = d.copy(); d["hour"] = (d["hour"] + 5) % 24
        return d, ["hour"]

    def scale_x1000(d):                              # seconds -> milliseconds
        d = d.copy(); d["travel_s"] = d["travel_s"] * 1000
        return d, ["travel_s"]

    def int_cast(d):                                 # precision lost: rounded to whole minutes
        d = d.copy(); d["travel_s"] = (d["travel_s"] / 60).round(0) * 60
        return d, ["travel_s"]

    def enum_remap(d):
        d = d.copy(); d["borough"] = d["borough"].map(lambda v: {0.0: 1.0, 1.0: 0.0}.get(v, v))
        return d, ["borough"]

    return {
        "broken_join": ("join", RESPONSE),
        "unit_change": ("rowwise", unit_change),
        "swapped_columns": ("rowwise", swapped_columns),
        "default_fill": ("rowwise", default_fill),
        "timezone_shift": ("rowwise", timezone_shift),
        "scale_x1000": ("rowwise", scale_x1000),
        "int_cast": ("rowwise", int_cast),
        "enum_remap": ("rowwise", enum_remap),
        "duplicate_rows": ("dup",),
        "missing_segment": ("missing", lambda d: ((d["borough"] == 2.0).to_numpy(), ["borough"])),
    }
