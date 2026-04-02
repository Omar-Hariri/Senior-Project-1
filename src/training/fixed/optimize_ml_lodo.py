# src/models/ml/optimize_ml_lodo.py
# ================================
# Hyperparameter optimization using the full LODO-CV process.
# This averages precision/recall/f1 across all 6 folds for each trial.
#
# Run with:
#   python -m src.models.ml.optimize_ml_lodo --model rf
#   python -m src.models.ml.optimize_ml_lodo --model xgb

import argparse
import yaml
import numpy as np
import optuna
import wandb
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

def load_config(path):
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)

def load_fold_data(fold_dir):
    """Load ML data for a specific fold."""
    X_train = np.load(fold_dir / "ml" / "X_train.npy")
    y_train = np.load(fold_dir / "ml" / "y_train.npy")
    X_test  = np.load(fold_dir / "ml" / "X_test.npy")
    y_test  = np.load(fold_dir / "ml" / "y_test.npy")
    return X_train, y_train, X_test, y_test

def objective(trial, model_type, ready_dir, window_name, folds, cfg):
    """Optuna objective function using full 6-fold LODO-CV."""
    
    # 1. Define model parameters based on model type
    if model_type == "rf":
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", *cfg["rf"]["params"]["n_estimators"]),
            "max_depth":         trial.suggest_int("max_depth", *cfg["rf"]["params"]["max_depth"]),
            "min_samples_split": trial.suggest_int("min_samples_split", *cfg["rf"]["params"]["min_samples_split"]),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", *cfg["rf"]["params"]["min_samples_leaf"]),
            "class_weight":      "balanced",
            "random_state":      42,
            "n_jobs":            -1
        }
    elif model_type == "xgb":
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", *cfg["xgb"]["params"]["n_estimators"]),
            "max_depth":        trial.suggest_int("max_depth", *cfg["xgb"]["params"]["max_depth"]),
            "learning_rate":    trial.suggest_float("learning_rate", *cfg["xgb"]["params"]["learning_rate"]),
            "subsample":        trial.suggest_float("subsample", *cfg["xgb"]["params"]["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg["xgb"]["params"]["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *cfg["xgb"]["params"]["min_child_weight"]),
            "objective":        "multi:softprob",
            "num_class":        3,
            "random_state":     42,
            "n_jobs":           -1
        }

    # 2. Run over all folds
    fold_f1s = []
    for fold_cfg in folds:
        fold_num = fold_cfg["fold"]
        fold_dir = ready_dir / window_name / f"fold{fold_num}"
        
        if not fold_dir.exists():
            continue
            
        X_train, y_train, X_test, y_test = load_fold_data(fold_dir)
        
        if model_type == "rf":
            model = RandomForestClassifier(**params)
        else:
            model = XGBClassifier(**params)
            
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        fold_f1s.append(f1)

    mean_f1 = np.mean(fold_f1s) if fold_f1s else 0
    
    # Log trial result
    wandb.log({"mean_f1": mean_f1, **params})
    
    return mean_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["rf", "xgb"])
    parser.add_argument("--window", type=str, default="w10s")
    parser.add_argument("--opt_config", type=str, default="configs/optuna_config.yaml")
    parser.add_argument("--split_config", type=str, default="configs/data_split.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.opt_config)
    split_cfg = load_config(args.split_config)
    
    ready_dir = Path(split_cfg["ready_dir"])
    folds = split_cfg["folds"]
    
    print("=" * 60)
    print(f"LODO-CV Optimization - Model: {args.model.upper()} - Window: {args.window}")
    print(f"Optimizing across {len(folds)} folds.")
    print("=" * 60)
    
    # Initialize W&B
    wandb.init(
        project=cfg["wandb"]["project"],
        name=f"LODO_Opt_{args.model.upper()}_{args.window}",
        group="LODO_Optimization",
        config={**cfg[args.model], "window": args.window}
    )
    
    # Run Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, args.model, ready_dir, args.window, folds, cfg),
        n_trials=cfg[args.model]["n_trials"]
    )
    
    print("\n" + "=" * 60)
    print(f"Optimization Finished for {args.model.upper()}")
    print(f"Best LODO-F1: {study.best_trial.value:.4f}")
    print("Best parameters:", study.best_params)
    print("=" * 60)
    
    # Log summary
    wandb.run.summary["best_lodo_f1"] = study.best_trial.value
    for k, v in study.best_params.items():
        wandb.run.summary[f"best_{k}"] = v
    
    wandb.finish()

if __name__ == "__main__":
    main()
