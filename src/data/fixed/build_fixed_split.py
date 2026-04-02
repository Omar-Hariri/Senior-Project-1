# src/data/build_fixed_split.py
# ============================
# Builds a fixed train/val/test split of driver behavior data.
# This script is separate from the cross-validation fold builder.
#
# Run with:
#   python -m src.data.build_fixed_split --config configs/fixed_split_config.yaml

import argparse
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.feature_engineering import (
    fit_scaler, apply_scaler, save_scaler,
    build_all_windows, compute_window_stats,
    LSTM_FEATURES
)

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_driver_data(processed_dir, drivers):
    """Load all processed CSV files for the given drivers."""
    processed_dir = Path(processed_dir)
    dfs = []
    
    for driver in drivers:
        driver_path = processed_dir / driver
        csv_files = sorted(driver_path.glob("*.csv"))
        
        if not csv_files:
            print(f"  WARNING: No CSV files found for driver {driver}")
            continue
            
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"    Loaded {csv_file.name} ({len(df)} rows)")
            
    return dfs

def save_split_arrays(base_path, X_lstm, X_ml, y, split_name):
    """Save LSTM and ML arrays to their respective subdirectories."""
    # Paths: base_path / split_name / {lstm, ml}
    lstm_dir = base_path / split_name / "lstm"
    ml_dir   = base_path / split_name / "ml"
    
    lstm_dir.mkdir(parents=True, exist_ok=True)
    ml_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LSTM arrays
    np.save(lstm_dir / "X.npy", X_lstm)
    np.save(lstm_dir / "y.npy", y)
    
    # Save ML arrays
    np.save(ml_dir / "X.npy", X_ml)
    np.save(ml_dir / "y.npy", y)

def main():
    parser = argparse.ArgumentParser(description="Build fixed train/val/test split data")
    parser.add_argument("--config", type=str, default="configs/fixed_split_config.yaml",
                        help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    # 1. Load Config
    cfg = load_config(args.config)
    processed_dir = cfg["dataset"]["processed_dir"]
    train_drivers = cfg["split"]["train"]
    val_drivers   = cfg["split"]["val"]
    test_drivers  = cfg["split"]["test"]
    ready_base    = Path(cfg["output"]["ready_dir"])
    
    print("=" * 60)
    print("BUILDING FIXED SPLIT DATA")
    print(f"Train Drivers: {train_drivers}")
    print(f"Val Drivers:   {val_drivers}")
    print(f"Test Drivers:  {test_drivers}")
    print("=" * 60)
    
    for win_cfg in cfg["windows"]:
        win_name = win_cfg["name"]
        win_len  = win_cfg["window_length"]
        stride   = win_cfg["stride"]
        
        win_output_dir = ready_base / win_name
        win_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing Window Configuration: {win_name}")
        print("-" * 40)
        
        # --- A. TRAIN SPLIT ---
        print(f"Step 1: Processing TRAIN data ({train_drivers})")
        train_dfs = load_driver_data(processed_dir, train_drivers)
        
        print("  Fitting scaler on train data...")
        scaler = fit_scaler(train_dfs)
        save_scaler(scaler, win_output_dir)
        
        print("  Applying scaler and building windows...")
        train_dfs_scaled = [apply_scaler(df, scaler) for df in train_dfs]
        X_train_lstm, y_train = build_all_windows(train_dfs_scaled, win_len, stride)
        
        print("  Computing ML statistical features...")
        X_train_ml = compute_window_stats(X_train_lstm)
        
        print("  Saving train arrays...")
        save_split_arrays(win_output_dir, X_train_lstm, X_train_ml, y_train, "train")
        
        # --- B. VAL SPLIT ---
        print(f"\nStep 2: Processing VAL data ({val_drivers})")
        val_dfs = load_driver_data(processed_dir, val_drivers)
        
        print("  Applying scaler and building windows...")
        val_dfs_scaled = [apply_scaler(df, scaler) for df in val_dfs]
        X_val_lstm, y_val = build_all_windows(val_dfs_scaled, win_len, stride)
        
        print("  Computing ML statistical features...")
        X_val_ml = compute_window_stats(X_val_lstm)
        
        print("  Saving val arrays...")
        save_split_arrays(win_output_dir, X_val_lstm, X_val_ml, y_val, "val")
        
        # --- C. TEST SPLIT ---
        print(f"\nStep 3: Processing TEST data ({test_drivers})")
        test_dfs = load_driver_data(processed_dir, test_drivers)
        
        print("  Applying scaler and building windows...")
        test_dfs_scaled = [apply_scaler(df, scaler) for df in test_dfs]
        X_test_lstm, y_test = build_all_windows(test_dfs_scaled, win_len, stride)
        
        print("  Computing ML statistical features...")
        X_test_ml = compute_window_stats(X_test_lstm)
        
        print("  Saving test arrays...")
        save_split_arrays(win_output_dir, X_test_lstm, X_test_ml, y_test, "test")
        
        # --- D. SUMMARY ---
        print("\n" + "=" * 40)
        print(f"SHAPE SUMMARY - Window: {win_name}")
        print("=" * 40)
        print(f"  {'Split':<10} | {'Type':<6} | {'X Shape':<18} | {'y Shape':<10}")
        print("-" * 55)
        print(f"  {'Train':<10} | {'LSTM':<6} | {str(X_train_lstm.shape):<18} | {str(y_train.shape):<10}")
        print(f"  {'Train':<10} | {'ML':<6} | {str(X_train_ml.shape):<18} | {str(y_train.shape):<10}")
        print(f"  {'Val':<10} | {'LSTM':<6} | {str(X_val_lstm.shape):<18} | {str(y_val.shape):<10}")
        print(f"  {'Val':<10} | {'ML':<6} | {str(X_val_ml.shape):<18} | {str(y_val.shape):<10}")
        print(f"  {'Test':<10} | {'LSTM':<6} | {str(X_test_lstm.shape):<18} | {str(y_test.shape):<10}")
        print(f"  {'Test':<10} | {'ML':<6} | {str(X_test_ml.shape):<18} | {str(y_test.shape):<10}")
        print("-" * 55)

    print(f"\nCOMPLETED. Ready data stored in: {ready_base}")

if __name__ == "__main__":
    main()
