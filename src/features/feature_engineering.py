# src/features/feature_engineering.py
# =================================

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

# ─────────────────────────────────────────
# Feature Definitions
# ─────────────────────────────────────────

LSTM_FEATURES = [
    "ax_kf", "ay_kf", "az_kf",
    "roll", "pitch", "yaw",
    "speed"
]

ML_STAT_FEATURES = [
    "ax_kf", "ay_kf", "az_kf",
    "roll", "pitch", "yaw",
    "speed"
]


# ─────────────────────────────────────────
# Computing Features
# ─────────────────────────────────────────

def compute_features(df):
    """
    1. Drops inactive rows.
    2. Computes exactly 6 new columns.
    """
    df = df.copy()

    # 2. Compute the 6 new columns
    df["ax_kf_diff"] = df["ax_kf"].diff().fillna(0)
    df["ay_kf_diff"] = df["ay_kf"].diff().fillna(0)
    df["az_kf_diff"] = df["az_kf"].diff().fillna(0)
    
    df["speed_diff"] = df["speed"].diff().fillna(0)
    
    # acc_mag = sqrt(ax_kf² + ay_kf² + az_kf²)
    df["acc_mag"] = np.sqrt(df["ax_kf"]**2 + df["ay_kf"]**2 + df["az_kf"]**2)
    
    df["yaw_diff"] = df["yaw"].diff().fillna(0)

    return df


# ─────────────────────────────────────────
# Normalization (Scalers)
# ─────────────────────────────────────────

def fit_scaler(train_dfs):
    combined = pd.concat(train_dfs, ignore_index=True)

    scaler = StandardScaler()
    scaler.fit(combined[LSTM_FEATURES])

    print(f"  Scaler fitted on {len(combined)} training rows")
    return scaler

def apply_scaler(df, scaler):
    df = df.copy()
    df[LSTM_FEATURES] = scaler.transform(df[LSTM_FEATURES])
    return df

def save_scaler(scaler, save_dir):
    path = Path(save_dir) / "scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved → {path}")

def load_scaler(save_dir):
    path = Path(save_dir) / "scaler.pkl"
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Scaler loaded ← {path}")
    return scaler


# ─────────────────────────────────────────
# Splitting logic and Stat aggregation
# ─────────────────────────────────────────

def build_windows(df, window_length, stride):
    X_list = []
    y_list = []

    values = df[LSTM_FEATURES].values
    labels = df["label"].values
    T = len(values)

    for start in range(0, T - window_length + 1, stride):
        end = start + window_length

        window_x = values[start:end]
        window_y = labels[start:end]
        label = int(np.bincount(window_y.astype(int)).argmax())

        X_list.append(window_x)
        y_list.append(label)

    if not X_list:
        return (
            np.empty((0, window_length, len(LSTM_FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64)
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y


def build_all_windows(session_dfs, window_length, stride):
    X_all = []
    y_all = []

    for df in session_dfs:
        X, y = build_windows(df, window_length, stride)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        return (
            np.empty((0, window_length, len(LSTM_FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64)
        )

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def compute_window_stats(X_3d):
    """
    Computes global statistics for each window over the time axis.
    
    Parameters
    ----------
    X_3d : numpy array shape (N, window_length, 7)
        The 3D tensor of LSTM features.
        
    Returns
    -------
    X_stats : numpy array shape (N, 49)
        Flattened 2D array containing 7 stats for the 7 ML_STAT_FEATURES.
        (mean, std, min, max, range, skew, kurtosis) sequentially stacked.
    """
    N, window_length, num_features = X_3d.shape
    
    # 1. Find the index of each ML_STAT_FEATURES signal inside LSTM_FEATURES
    indices = [LSTM_FEATURES.index(f) for f in ML_STAT_FEATURES]
    
    # 2. Extract those 7 signals from X_3d
    # Shape becomes (N, window_length, 7)
    X_target = X_3d[:, :, indices]
    
    # 3. Compute 7 statistics across the time axis (axis=1) for each signal
    # Each resulting array will have shape (N, 8)
    means  = np.mean(X_target, axis=1)
    stds   = np.std(X_target, axis=1)
    mins   = np.min(X_target, axis=1)
    maxs   = np.max(X_target, axis=1)
    ranges = maxs - mins
    
    # skew and kurtosis can return NaN if signal variance is perfectly 0 in a window
    skews = skew(X_target, axis=1, nan_policy='omit')
    skews = np.nan_to_num(skews, nan=0.0)
    
    kurts = kurtosis(X_target, axis=1, nan_policy='omit')
    kurts = np.nan_to_num(kurts, nan=0.0)
    
    # 4. Concatenate and return shape (N, 56) (which is 8 signals x 7 stats)
    X_stats = np.concatenate([means, stds, mins, maxs, ranges, skews, kurts], axis=1)
    
    return X_stats
