# src/models/ml/train_lr.py
# ========================
# Trains a Logistic Regression model with LODO-CV
# Tracks experiments using Weights & Biases (W&B)
# One W&B run per window (consistent with all other models)
#
# Run with:
#   python -m src.models.ml.train_lr

import argparse
import yaml
import numpy as np
import wandb
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from src.utils.plotting import log_classic_confusion_matrix


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_fold_data(fold_dir):
    X_train = np.load(fold_dir / "ml" / "X_train.npy")
    y_train = np.load(fold_dir / "ml" / "y_train.npy")
    X_test  = np.load(fold_dir / "ml" / "X_test.npy")
    y_test  = np.load(fold_dir / "ml" / "y_test.npy")
    return X_train, y_train, X_test, y_test


def main(split_config_path, windows_config_path, model_config_path, window_filter=None):
    split_cfg   = load_config(split_config_path)
    windows_cfg = load_config(windows_config_path)
    model_cfg   = load_config(model_config_path)

    wandb_cfg    = model_cfg["wandb"]
    ready_dir    = Path(split_cfg["ready_dir"])
    windows      = windows_cfg["windows"]
    folds        = split_cfg["folds"]
    model_params = model_cfg["model"]
    class_names  = model_cfg["output"]["class_names"]

    if window_filter:
        windows = [w for w in windows if w["name"] == window_filter]

    print("=" * 60)
    print("Training Logistic Regression with LODO-CV")
    print(f"W&B Project : {wandb_cfg.get('project', 'driver-monitoring-system')}")
    print(f"Windows     : {[w['name'] for w in windows]}")
    print("=" * 60)

    for window_cfg in windows:
        window_name = window_cfg["name"]
        run = wandb.init(
            project=wandb_cfg.get("project", "driver-monitoring-system"),
            name=f"LR_LODO_{window_name}",
            group="LODO_CV_Benchmarks",
            entity=wandb_cfg.get("entity", None),
            config={**model_params, "window": window_name},
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

            model = LogisticRegression(**model_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

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
                    fold_logs[f"fold{fold_num}_{cls}_f1"]        = report[cls]["f1-score"]
                    fold_logs[f"fold{fold_num}_{cls}_precision"]  = report[cls]["precision"]
                    fold_logs[f"fold{fold_num}_{cls}_recall"]     = report[cls]["recall"]

            wandb.log(fold_logs)

            log_classic_confusion_matrix(
                y_test, y_pred, class_names,
                f"fold{fold_num}_confusion_matrix",
            )

            # Save model locally
            models_dir = Path("artifacts/models/lr")
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, models_dir / f"model_{window_name}_fold{fold_num}.joblib")

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
    parser.add_argument("--model_config",   type=str, default="configs/lr_config.yaml")
    parser.add_argument("--window",         type=str, default=None,
                        help="Train on a single window (e.g. w5s). Omit to train all.")
    args = parser.parse_args()
    main(args.split_config, args.windows_config, args.model_config, args.window)
