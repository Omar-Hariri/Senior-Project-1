# src/data/build_folds.py
# ========================
# Reads processed CSVs and produces ready-to-train
# numpy arrays organized by fold and window size.
#
# Run with:
#   python -m src.data.build_folds
#   python -m src.data.build_folds --window w5s
#
# Input:
#   data/processed/UAH-DRIVESET-v1/D1/*.csv ... D6/*.csv
#
# Output:
#   data/ready/UAH-DRIVESET-v1/
#     w3s/
#       fold1/
#         scaler.pkl
#         lstm/  X_train.npy  X_test.npy  y_train.npy  y_test.npy
#         ml/    X_train.npy  X_test.npy  y_train.npy  y_test.npy
#       fold2/ ... fold6/
#     w5s/
#       fold1/ ... fold6/

import argparse
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.feature_engineering import (
    fit_scaler,
    apply_scaler,
    save_scaler,
    build_all_windows,
    compute_window_stats,
    LSTM_FEATURES,
)


# ─────────────────────────────────────────
# Load config
# ─────────────────────────────────────────

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────
# Load CSVs for given drivers
# ─────────────────────────────────────────

def load_driver_csvs(processed_dir, drivers):
    """
    Load all CSV files for the given list of drivers.

    Parameters
    ----------
    processed_dir : str or Path
    drivers       : list of str  e.g. ["D1", "D2"]

    Returns
    -------
    list of DataFrames — one per session file
    """
    processed_dir = Path(processed_dir)
    dfs = []

    for driver in drivers:
        driver_path = processed_dir / driver
        csv_files   = sorted(driver_path.glob("*.csv"))

        if not csv_files:
            print(f"  WARNING: no CSV files found for {driver}")
            continue

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"    loaded {csv_file.name}  ({len(df)} rows)")

    return dfs


# ─────────────────────────────────────────
# Shuffle windows (rows) while keeping X/y aligned
# ─────────────────────────────────────────

def shuffle_windows(X, y, seed):
    """
    Shuffle windows (rows) in X and y together.
    Does NOT shuffle timesteps inside a window.

    Parameters
    ----------
    X    : numpy array  shape (n_windows, timesteps, features)  or  (n_windows, features)
    y    : numpy array  shape (n_windows,)
    seed : int          random seed for reproducibility

    Returns
    -------
    X_shuffled, y_shuffled
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ─────────────────────────────────────────
# Save numpy arrays
# ─────────────────────────────────────────

def save_arrays(folder, X, y, split_name):
    """
    Save X and y arrays to folder/X_{split_name}.npy
    and folder/y_{split_name}.npy

    Parameters
    ----------
    folder     : Path
    X          : numpy array
    y          : numpy array
    split_name : str — "train" or "test"
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    np.save(folder / f"X_{split_name}.npy", X)
    np.save(folder / f"y_{split_name}.npy", y)

    print(f"    saved X_{split_name}.npy  shape={X.shape}")
    print(f"    saved y_{split_name}.npy  shape={y.shape}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(split_config_path, windows_config_path, window_filter=None):

    split_cfg    = load_config(split_config_path)
    windows_cfg  = load_config(windows_config_path)

    processed_dir = split_cfg["processed_dir"]
    ready_dir     = Path(split_cfg["ready_dir"])
    folds         = split_cfg["folds"]
    windows       = windows_cfg["windows"]
    shuffle_seed  = split_cfg.get("shuffle_seed", 42)

    # Filter to a single window if requested
    if window_filter:
        windows = [w for w in windows if w["name"] == window_filter]
        if not windows:
            print(f"ERROR: window '{window_filter}' not found in config.")
            print(f"Available: {[w['name'] for w in windows_cfg['windows']]}")
            return

    print("=" * 60)
    print("build_folds.py")
    print(f"processed_dir : {processed_dir}")
    print(f"ready_dir     : {ready_dir}")
    print(f"folds         : {len(folds)}")
    print(f"window configs: {[w['name'] for w in windows]}")
    print(f"shuffle_seed  : {shuffle_seed}")
    print("=" * 60)

    for window_cfg in windows:
        window_name   = window_cfg["name"]
        window_length = window_cfg["window_length"]
        stride        = window_cfg["stride"]

        print()
        print("=" * 60)
        print(f"Window config: {window_name}")
        print(f"  window_length = {window_length} timesteps")
        print(f"  stride        = {stride} timesteps")
        print("=" * 60)

        for fold_cfg in folds:
            fold_num      = fold_cfg["fold"]
            test_driver   = fold_cfg["test"]
            train_drivers = fold_cfg["train"]

            fold_dir = ready_dir / window_name / f"fold{fold_num}"

            lstm_train = fold_dir / "lstm" / "X_train.npy"
            if lstm_train.exists():
                print(f"  SKIP: fold{fold_num} already exists")
                continue

            print()
            print(f"  Fold {fold_num} — test={test_driver}  "
                  f"train={train_drivers}")
            print(f"  output → {fold_dir}")
            print()

            # ── Load train CSVs ───────────────
            print(f"  Loading train drivers: {train_drivers}")
            train_dfs = load_driver_csvs(processed_dir, train_drivers)
            print(f"  Loaded {len(train_dfs)} train sessions")

            # ── Fit scaler on train only ──────
            print()
            print("  Fitting scaler on train data...")
            scaler = fit_scaler(train_dfs)

            # Save scaler for this fold
            scaler_path = fold_dir / "scaler.pkl"
            fold_dir.mkdir(parents=True, exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            print(f"  Saved scaler → {scaler_path}")

            # ── Apply scaler to train ─────────
            train_dfs = [apply_scaler(df, scaler) for df in train_dfs]

            # ── Build train windows ───────────
            print()
            print("  Building train windows...")
            X_train, y_train = build_all_windows(
                train_dfs, window_length, stride
            )
            print(f"  X_train shape (before shuffle): {X_train.shape}")

            # ── Shuffle train windows ─────────
            X_train, y_train = shuffle_windows(X_train, y_train, seed=shuffle_seed)
            print(f"  X_train shape (after shuffle) : {X_train.shape}  seed={shuffle_seed}")

            # ── Compute train ML stats ────────
            X_train_ml = compute_window_stats(X_train)
            print(f"  X_train_ml shape: {X_train_ml.shape}")

            # ── Save train arrays ─────────────
            print()
            print("  Saving train arrays...")
            save_arrays(fold_dir / "lstm", X_train,    y_train, "train")
            save_arrays(fold_dir / "ml",   X_train_ml, y_train, "train")

            # ── Load test CSVs ────────────────
            print()
            print(f"  Loading test driver: {test_driver}")
            test_dfs = load_driver_csvs(processed_dir, [test_driver])
            print(f"  Loaded {len(test_dfs)} test sessions")

            # ── Apply scaler to test ──────────
            test_dfs = [apply_scaler(df, scaler) for df in test_dfs]

            # ── Build test windows ────────────
            print()
            print("  Building test windows...")
            X_test, y_test = build_all_windows(
                test_dfs, window_length, stride
            )
            print(f"  X_test shape: {X_test.shape}")

            # ── Compute test ML stats ─────────
            X_test_ml = compute_window_stats(X_test)
            print(f"  X_test_ml shape: {X_test_ml.shape}")

            # ── Save test arrays ──────────────
            print()
            print("  Saving test arrays...")
            save_arrays(fold_dir / "lstm", X_test,    y_test, "test")
            save_arrays(fold_dir / "ml",   X_test_ml, y_test, "test")

            print()
            print(f"  Fold {fold_num} done.")
            print("-" * 60)

    # ── Summary ───────────────────────────
    print()
    print("=" * 60)
    print("BUILD FOLDS COMPLETE")
    print("=" * 60)
    print()
    print("Output structure:")
    for p in sorted(ready_dir.rglob("*.npy")):
        print(f"  {p.relative_to(ready_dir)}")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build LODO-CV fold arrays from processed CSVs"
    )
    parser.add_argument(
        "--split_config",
        type=str,
        default="configs/data_split.yaml",
        help="Path to data split / folds config",
    )
    parser.add_argument(
        "--windows_config",
        type=str,
        default="configs/windows.yaml",
        help="Path to windows config",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="Build only this window (e.g. w5s). Omit to build all.",
    )
    args = parser.parse_args()
    main(args.split_config, args.windows_config, args.window)