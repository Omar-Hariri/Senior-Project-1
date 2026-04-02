# src/preprocessing/uah_preprocessor.py
# =======================================
# Takes raw accel and GPS DataFrames for one session.
# Returns one clean merged DataFrame at 10Hz.
# Called by src/data/build_uah_dataset.py

import numpy as np
import pandas as pd


# ─────────────────────────────────────────
# Step 1 — Upsample GPS from 1Hz to 10Hz
# ─────────────────────────────────────────

def upsample_gps(gps, target_hz):
    """
    GPS records at 1Hz — one row per second.
    Accelerometer records at 10Hz — one row per 0.1 second.
    We upsample GPS to 10Hz using linear interpolation.

    Example:
        real GPS:      t=0.0 speed=65.2    t=1.0 speed=64.5
        after upsample:
          t=0.0 speed=65.2  ← real
          t=0.1 speed=65.1  ← interpolated
          t=0.2 speed=65.0  ← interpolated
          ...
          t=1.0 speed=64.5  ← real
    """
    # Drop duplicate timestamps — some DROWSY sessions have them
    gps = (
        gps
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .set_index("timestamp")
    )

    # Build new index at 10Hz
    step      = round(1.0 / target_hz, 6)
    new_index = np.arange(gps.index[0], gps.index[-1] + step, step)
    new_index = np.round(new_index, 6)

    # Reindex and interpolate
    gps_up = (
        gps
        .reindex(gps.index.union(new_index))
        .interpolate(method="index")
        .loc[new_index]
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )

    return gps_up


# ─────────────────────────────────────────
# Step 2 — Merge accel + GPS on timestamp
# ─────────────────────────────────────────

def merge_signals(accel, gps_up):
    """
    Merge accelerometer and upsampled GPS into one DataFrame.
    Uses merge_asof — matches each accel row to nearest GPS timestamp.
    Tolerance = 0.15 seconds handles small misalignments.
    """
    merged = pd.merge_asof(
        accel.sort_values("timestamp"),
        gps_up.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=0.15
    )

    return merged


# ─────────────────────────────────────────
# Step 3 — Fill any NaN values
# ─────────────────────────────────────────

def fill_gaps(df):
    """
    Fill any NaN values that appeared after merge.
    Uses linear interpolation first.
    Then forward fill and back fill for edges.
    """
    df = df.copy()
    df = df.interpolate(method="linear")
    df = df.ffill()
    df = df.bfill()
    return df


# ─────────────────────────────────────────
# Main — process one session
# ─────────────────────────────────────────

def process_session(session, cfg):
    """
    Run full preprocessing for one session.

    Input:  session dict from uah_loader.load_all_sessions()
    Output: one clean merged DataFrame ready for feature engineering

    Steps:
        1. upsample_gps()    — GPS 1Hz → 10Hz
        2. merge_signals()   — accel + GPS → one table
        3. fill_gaps()       — fix any NaN values
        4. add metadata      — label, driver, state, session_id
    """
    target_hz = cfg["preprocessing"]["target_hz"]

    # Step 1 — upsample GPS
    gps_up = upsample_gps(session["gps"], target_hz)

    # Step 1.5 — Filter accel
    # drop rows where sys_active == 0 (sensors not yet recording)
    accel_active = session["accel"][session["accel"]["sys_active"] == 1].copy()

    # Step 2 — merge accel + GPS
    merged = merge_signals(accel_active, gps_up)

    if merged.empty:
        print(f"  ERROR: {session['session_id']} — merge returned empty")
        return None

    # Step 3 — fill NaN values
    merged = fill_gaps(merged)

    # Step 4 — add metadata columns
    merged["label"]      = session["label"]
    merged["driver"]     = session["driver"]
    merged["state"]      = session["state"]
    merged["session_id"] = session["session_id"]

    return merged