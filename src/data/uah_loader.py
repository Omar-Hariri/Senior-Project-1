# src/data/uah_loader.py
# ========================
# Reads UAH-DriveSet raw files.
# Simple version — no regex, no pattern matching.
# Called by src/data/build_uah_dataset.py

import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────

ACCEL_COLUMNS = [
    "timestamp",
    "sys_active",
    "ax", "ay", "az",           # raw — will be dropped
    "ax_kf", "ay_kf", "az_kf", # kalman filtered (Gs)
    "roll", "pitch", "yaw",
]

GPS_COLUMNS = ["timestamp", "speed"]


# ─────────────────────────────────────────
# Helper — get state from folder name
# ─────────────────────────────────────────

def get_state(folder_name):
    """
    Look for AGGRESSIVE, DROWSY, or NORMAL in the folder name.
    Returns the state string or None if not found.

    Examples:
        "20151111-D1-AGGRESSIVE-MOTORWAY" → "AGGRESSIVE"
        "20151111-D1-NORMAL1-SECONDARY"   → "NORMAL"
        "20151111-D1-DROWSY-MOTORWAY"     → "DROWSY"
    """
    folder_upper = folder_name.upper()

    if "AGGRESSIVE" in folder_upper:
        return "AGGRESSIVE"
    elif "DROWSY" in folder_upper:
        return "DROWSY"
    elif "NORMAL" in folder_upper:
        return "NORMAL"
    else:
        return None


def get_driver(folder_name):
    """
    Look for D1-D6 in the folder name.
    Returns driver string or None if not found.

    Example:
        "20151111-D1-AGGRESSIVE-MOTORWAY" → "D1"
    """
    folder_upper = folder_name.upper()

    for driver in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        if driver in folder_upper:
            return driver

    return None


# ─────────────────────────────────────────
# Read raw files
# ─────────────────────────────────────────

def read_accel(session_folder):
    """
    Read RAW_ACCELEROMETERS.txt.
    Drops raw ax, ay, az — keeps only kalman filtered values.

    Returns DataFrame with these columns at 10Hz:
        timestamp, sys_active, ax_kf, ay_kf, az_kf, roll, pitch, yaw

    Returns None if file is missing.
    """
    path = Path(session_folder) / "RAW_ACCELEROMETERS.txt"

    if not path.exists():
        print(f"  WARNING: missing {path}")
        return None

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ACCEL_COLUMNS,
        engine="python"
    )

    # Drop raw axes — we only want kalman filtered
    df = df.drop(columns=["ax", "ay", "az"])

    return df


def read_gps(session_folder):
    """
    Read RAW_GPS.txt — keep only timestamp and speed.

    Returns DataFrame with 2 columns at 1Hz:
        timestamp, speed (km/h)

    Returns None if file is missing.
    """
    path = Path(session_folder) / "RAW_GPS.txt"

    if not path.exists():
        print(f"  WARNING: missing {path}")
        return None

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=GPS_COLUMNS,
        engine="python"
    )


# ─────────────────────────────────────────
# Load all sessions
# ─────────────────────────────────────────

def load_all_sessions(cfg):
    """
    Walk D1-D6 folders and load all sessions.

    Returns list of session dicts. Each dict has:
        session_id : str   e.g. "D1_AGGRESSIVE"
        driver     : str   e.g. "D1"
        state      : str   e.g. "AGGRESSIVE"
        label      : int   0=NORMAL  1=AGGRESSIVE  2=DROWSY
        accel      : DataFrame  8 cols  10Hz
        gps        : DataFrame  2 cols  1Hz
    """
    root          = Path(cfg["dataset"]["root_dir"])
    label_map     = cfg["dataset"]["label_map"]
    target_states = [s.upper() for s in cfg["dataset"]["target_states"]]
    drivers       = [d.upper() for d in cfg["dataset"]["drivers"]]

    sessions = []

    for driver_dir in sorted(root.iterdir()):

        if not driver_dir.is_dir():
            continue
        if driver_dir.name.upper() not in drivers:
            continue

        for session_dir in sorted(driver_dir.iterdir()):

            if not session_dir.is_dir():
                continue

            folder_name = session_dir.name

            # Get state and driver from folder name
            state  = get_state(folder_name)
            driver = get_driver(folder_name)

            # Skip if we could not read state or driver
            if state is None or driver is None:
                continue

            # Skip states not wanted in config
            if state not in target_states:
                continue

            # Get label from config
            label = label_map.get(state, -1)
            if label == -1:
                continue

            # Read raw files
            accel = read_accel(session_dir)
            gps   = read_gps(session_dir)

            if accel is None or gps is None:
                continue

            session_id = f"{driver}_{state}"

            sessions.append({
                "session_id": session_id,
                "folder_name": folder_name,
                "driver":     driver,
                "state":      state,
                "label":      label,
                "accel":      accel,
                "gps":        gps,
            })

            print(f"  OK  {session_id:25s}"
                  f"  label={label}"
                  f"  accel={len(accel):5d}rows"
                  f"  gps={len(gps):4d}rows")

    print()
    print(f"Total loaded: {len(sessions)} sessions")

    return sessions
