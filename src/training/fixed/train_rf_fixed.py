# src/models/ml/train_rf_fixed.py
# ==============================
# Trains a Random Forest on a fixed train/val/test split.
# Evaluates performance across all window configurations (3s, 6s, 8s, 10s).
#
# Run with:
#   python -m src.models.ml.train_rf_fixed

import argparse
import yaml
import numpy as np
import wandb
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_ml_data(ready_dir, win_name):
    """Load ML feature arrays for a specific window."""
    base = Path(ready_dir) / win_name
    
    X_train = np.load(base / "train" / "ml" / "X.npy")
    y_train = np.load(base / "train" / "ml" / "y.npy")
    X_val   = np.load(base / "val" / "ml" / "X.npy")
    y_val   = np.load(base / "val" / "ml" / "y.npy")
    X_test  = np.load(base / "test" / "ml" / "X.npy")
    y_test  = np.load(base / "test" / "ml" / "y.npy")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def main():
    parser = argparse.ArgumentParser(description="Train RF on Fixed Split across multiple windows")
    parser.add_argument("--fixed_config", type=str, default="configs/fixed/fixed_split_config.yaml")
    parser.add_argument("--rf_config", type=str, default="configs/fixed/rf_fixed_config.yaml")
    args = parser.parse_args()
    
    # Load configs
    fixed_cfg = load_config(args.fixed_config)
    rf_cfg    = load_config(args.rf_config)
    
    ready_dir = Path(fixed_cfg["output"]["ready_dir"])
    windows   = fixed_cfg["windows"]
    model_params = rf_cfg["model"]
    class_names  = rf_cfg["output"]["class_names"]
    
    print("=" * 60)
    print("Training Random Forest on FIXED SPLIT")
    print(f"Windows to evaluate: {[w['name'] for w in windows]}")
    print("=" * 60)
    
    # Initialize W&B
    run = wandb.init(
        project=rf_cfg["wandb"]["project"],
        name=rf_cfg["wandb"]["experiment_name"],
        config=model_params
    )
    
    for win_cfg in windows:
        win_name = win_cfg["name"]
        print(f"\nEvaluating Window: {win_name}")
        
        # 1. Load Data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_ml_data(ready_dir, win_name)
        
        # 2. Initialize and Train Model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # 3. Evaluate on Val (D5)
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1  = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
        
        # 4. Evaluate on Test (D6)
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1  = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        print(f"  VAL (D5)  -> Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  TEST (D6) -> Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # 5. Logging to W&B
        cm_plot = wandb.plot.confusion_matrix(
            y_true=y_test,
            preds=y_test_pred,
            class_names=class_names
        )
        
        report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True, zero_division=0)
        
        metrics = {
            f"{win_name}_val_accuracy":  val_acc,
            f"{win_name}_val_f1":        val_f1,
            f"{win_name}_test_accuracy": test_acc,
            f"{win_name}_test_f1":       test_f1,
            f"{win_name}_conf_mat":      cm_plot
        }
        
        # Log per-class F1 for test
        for cls in class_names:
            metrics[f"{win_name}_{cls}_f1"] = report[cls]['f1-score']
            
        wandb.log(metrics)
        
        # 6. Save Model
        models_dir = Path("artifacts/models/rf_fixed")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"rf_model_{win_name}.joblib"
        joblib.dump(model, model_path)
        
    run.finish()
    print("\nDONE: All windows evaluated. Check W&B for the comparison.")

if __name__ == "__main__":
    main()
