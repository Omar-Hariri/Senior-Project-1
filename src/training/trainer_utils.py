# src/training/trainer_utils.py
# ========================
# Facade for training utilities.
# Delegates to data_utils, metrics, and logging_utils.
# This ensures existing pipelines (RF, SVM, etc.) continue to work without changing imports.

from .data_utils import load_config as _load_config
from .data_utils import load_fold_data as _load_fold_data
from .metrics import calculate_metrics, calculate_window_averages
from .logging_utils import log_fold_to_wandb, log_final_summary, log_summary_table

def load_config(config_path):
    return _load_config(config_path)

def load_fold_data(fold_dir, data_type="ml"):
    return _load_fold_data(fold_dir, data_type)

def evaluate_fold_metrics(y_true, y_pred, class_names, fold_num, log_to_wandb=True):
    """
    Evaluates predictions and optionally logs to W&B.
    This maintains the original API used by the pipelines.
    """
    # 1. Calculate
    res = calculate_metrics(y_true, y_pred, class_names)
    
    # 2. Log
    if log_to_wandb:
        log_fold_to_wandb(fold_num, res, class_names, y_true, y_pred)
        
    # Return the metrics dict for persistence in the pipeline's loop
    return res

def compute_window_summary(fold_results, window_name, log_to_wandb=True):
    """
    Computes and logs window-level cross-validation summary.
    This maintains the original API used by the pipelines.
    """
    if not fold_results:
        return None
        
    # 1. Calculate Averages
    summary = calculate_window_averages(fold_results)
    
    # 2. Log Final Summary
    if log_to_wandb:
        # Log the detailed W&B Summary Table (New Feature)
        # Assuming fold_results contains class names via the first result's report keys
        if fold_results:
            class_names = list(fold_results[0]["report"].keys())
            # filter out non-label keys from classification_report if any 
            # (though report=True usually is just class names + averages)
            class_names = [c for c in class_names if c not in ["accuracy", "macro avg", "weighted avg"]]
            log_summary_table(fold_results, window_name, class_names)
            
        log_final_summary(summary)

    # Print to console
    print(f"\n  Final Summary [{window_name}]:")
    print(f"    Avg Accuracy : {summary['avg_acc']:.4f} ± {summary['std_acc']:.4f}")
    print(f"    Avg F1 Macro : {summary['avg_f1']:.4f} ± {summary['std_f1']:.4f}")
        
    return summary
