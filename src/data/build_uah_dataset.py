# src/data/build_uah_dataset.py
# ====================
# Main entry point for dataset extraction.
#
# Run with:
#   python -m src.data.build_uah_dataset --config configs/uah_data.yaml
#
# What it does:
#   1. Reads uah_data.yaml
#   2. Loads all sessions      (uah_loader.py)
#   3. Processes each session  (uah_preprocessor.py)
#   4. Computes features       (feature_engineering.py)
#   5. Saves each session as a CSV file organized by driver

import argparse
import yaml
import os
from pathlib import Path

from src.data.uah_loader import load_all_sessions
from src.preprocessing.uah_preprocessor import process_session
from src.features.feature_engineering import compute_features

# ─────────────────────────────────────────
# Helper — load config file
# ─────────────────────────────────────────

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

# ─────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────

def main(config_path):

    # ── Load config ───────────────────────
    print("=" * 50)
    print("Loading config")
    print("=" * 50)
    cfg      = load_config(config_path)
    # Use processed_dir from config, fallback to save_dir if not present
    save_dir = Path(cfg["output"].get("processed_dir", cfg["output"]["save_dir"]))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load raw sessions ─────────
    print()
    print("=" * 50)
    print("Step 1 — Loading raw sessions")
    print("=" * 50)

    sessions = load_all_sessions(cfg)

    if not sessions:
        print("ERROR: No sessions found. Check root_dir in uah_data.yaml")
        return

    # ── Step 2, 3 & 4: Process, Feature extraction & Save 
    print()
    print("=" * 50)
    print("Step 2 & 3 — Processing & Feature Engineering")
    print("=" * 50)

    for session in sessions:
        # Step 2: Process session
        df = process_session(session, cfg)

        if df is None:
            print(f"  SKIP: {session['session_id']} — processing failed")
            continue
            
        # Step 3: Compute features
        df_feat = compute_features(df)
        
        KEEP_COLUMNS = [
            "timestamp",
            "ax_kf", "ay_kf", "az_kf",
            "roll", "pitch", "yaw",
            "speed",
            "ax_kf_diff", "ay_kf_diff", "az_kf_diff",
            "speed_diff",
            "acc_mag",
            "yaw_diff",
            "label"
        ]
        existing_cols = [c for c in KEEP_COLUMNS if c in df_feat.columns]
        df_feat = df_feat[existing_cols]

        # Step 4: Save to CSV organized by driver with original folder name
        driver_id = session["driver"].upper()
        session_name = session["folder_name"]
        
        driver_dir = save_dir / driver_id
        driver_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = driver_dir / f"{session_name}.csv"
        
        df_feat.to_csv(out_path, index=False)
        print(f"  SAVED: {out_path} ({len(df_feat)} rows)")


    # ── Done ──────────────────────────────
    print()
    print("=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UAH-DriveSet dataset build pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/uah_data.yaml",
        help="Path to uah_data.yaml config file"
    )
    args = parser.parse_args()

    main(args.config)
