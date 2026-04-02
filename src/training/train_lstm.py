# src/training/train_lstm.py
# ============================
# Trains an LSTM model on pre-processed sensor sequences
# Performs Leave-One-Driver-Out (LODO) cross-validation.
# One W&B run per window (consistent with all other models)
#
# Run with:
#   python -m src.training.train_lstm

import argparse
import yaml
import numpy as np
import wandb
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from src.utils.plotting import log_classic_confusion_matrix

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from wandb.integration.keras import WandbMetricsLogger

from src.models.deep.lstm import build_lstm_model


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_fold_data(fold_dir):
    """Loads LSTM sequences and labels for a specific fold."""
    X_train = np.load(fold_dir / "lstm" / "X_train.npy")
    y_train = np.load(fold_dir / "lstm" / "y_train.npy")
    X_test  = np.load(fold_dir / "lstm" / "X_test.npy")
    y_test  = np.load(fold_dir / "lstm" / "y_test.npy")
    return X_train, y_train, X_test, y_test


def main(split_config_path, windows_config_path, lstm_config_path, window_filter=None):
    split_cfg   = load_config(split_config_path)
    windows_cfg = load_config(windows_config_path)
    lstm_cfg    = load_config(lstm_config_path)

    ready_dir    = Path(split_cfg["ready_dir"])
    windows      = windows_cfg["windows"]
    folds        = split_cfg["folds"]
    class_names  = lstm_cfg["data"]["classes"]
    model_params = lstm_cfg["model"]
    train_params = lstm_cfg["training"]

    if window_filter:
        windows = [w for w in windows if w["name"] == window_filter]
        if not windows:
            print(f"ERROR: window '{window_filter}' not found in windows config.")
            return

    print("=" * 60)
    print("Training LSTM with LODO-CV")
    print(f"W&B Project : {lstm_cfg['logging']['project_name']}")
    print(f"Windows     : {[w['name'] for w in windows]}")
    print("=" * 60)

    for window_cfg in windows:
        window_name = window_cfg["name"]

        run = wandb.init(
            project=lstm_cfg["logging"]["project_name"],
            name=f"LSTM_LODO_{window_name}",
            group="LODO_CV_Benchmarks",
            config={**model_params, **train_params, "window": window_name},
            reinit="finish_previous",
        )

        print(f"\n[Window: {window_name}]  W&B Run: {run.id}")

        fold_results = []

        for fold_cfg in folds:
            fold_num    = fold_cfg["fold"]
            test_driver = fold_cfg["test"]
            fold_dir    = ready_dir / window_name / f"fold{fold_num}"

            if not fold_dir.exists():
                print(f"  SKIP: fold{fold_num} directory not found.")
                continue

            print(f"  Fold {fold_num} (Test: {test_driver}) ...", end=" ", flush=True)

            X_train, y_train, X_test, y_test = load_fold_data(fold_dir)

            model = build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                cfg={**model_params, "lr": train_params["learning_rate"]},
            )

            callbacks = [
                EarlyStopping(
                    monitor=train_params.get("monitor", "val_loss"),
                    patience=train_params["early_stopping_patience"],
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor=train_params.get("monitor", "val_loss"),
                    patience=train_params["reduction_patience"],
                    factor=0.5,
                ),
                WandbMetricsLogger(),
            ]

            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=train_params["epochs"],
                batch_size=train_params["batch_size"],
                callbacks=callbacks,
                verbose=1,
            )

            y_probs = model.predict(X_test)
            y_pred  = np.argmax(y_probs, axis=1)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

            fold_results.append({"acc": acc, "prec": prec, "rec": rec, "f1": f1})

            report = classification_report(
                y_test, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )

            fold_logs = {
                f"fold{fold_num}_accuracy" : acc,
                f"fold{fold_num}_precision": prec,
                f"fold{fold_num}_recall"   : rec,
                f"fold{fold_num}_f1"       : f1,
            }
            for cls in class_names:
                if cls in report:
                    fold_logs[f"fold{fold_num}_{cls}_f1"]       = report[cls]["f1-score"]
                    fold_logs[f"fold{fold_num}_{cls}_precision"] = report[cls]["precision"]
                    fold_logs[f"fold{fold_num}_{cls}_recall"]    = report[cls]["recall"]

            wandb.log(fold_logs)

            log_classic_confusion_matrix(
                y_test, y_pred, class_names,
                f"fold{fold_num}_confusion_matrix",
            )

            # Save model locally
            models_dir = Path("artifacts/models/lstm")
            models_dir.mkdir(parents=True, exist_ok=True)
            model.save(models_dir / f"model_{window_name}_fold{fold_num}.keras")

            print(f"Done.  Acc: {acc:.4f}  F1: {f1:.4f}")

        # ── Window-level summary ──────────────────────────────────────────────
        if fold_results:
            avg_acc  = np.mean([r["acc"]  for r in fold_results])
            avg_prec = np.mean([r["prec"] for r in fold_results])
            avg_rec  = np.mean([r["rec"]  for r in fold_results])
            avg_f1   = np.mean([r["f1"]   for r in fold_results])
            std_acc  = np.std([r["acc"]   for r in fold_results])
            std_f1   = np.std([r["f1"]    for r in fold_results])

            print(f"\n  Summary [{window_name}]:")
            print(f"    Avg Accuracy : {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"    Avg F1 Macro : {avg_f1:.4f} ± {std_f1:.4f}")

            wandb.log({
                "final_avg_accuracy" : avg_acc,
                "final_std_accuracy" : std_acc,
                "final_avg_precision": avg_prec,
                "final_avg_recall"   : avg_rec,
                "final_avg_f1"       : avg_f1,
                "final_std_f1"       : std_f1,
            })

        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_config",   type=str, default="configs/data_split.yaml")
    parser.add_argument("--windows_config", type=str, default="configs/windows.yaml")
    parser.add_argument("--lstm_config",    type=str, default="configs/lstm_config.yaml")
    parser.add_argument("--window",         type=str, default=None,
                        help="Train on a single window (e.g. w5s). Omit to train all.")
    args = parser.parse_args()
    main(args.split_config, args.windows_config, args.lstm_config, args.window)
