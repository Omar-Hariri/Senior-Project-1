import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def _plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    """
    Creates a heatmap confusion matrix using Seaborn.
    Returns the matplotlib figure for W&B logging.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display, nan=0.0)
        fmt = ".2f"
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = "d"
        title = 'Confusion Matrix (Counts)'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_display, 
        annot=True, 
        fmt=fmt, 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def log_confusion_matrix(y_true, y_pred, class_names, log_key, normalize=False):
    """
    Plots the CM and logs it to W&B as an image.
    """
    fig = _plot_confusion_matrix(y_true, y_pred, class_names, normalize=normalize)
    wandb.log({log_key: wandb.Image(fig)})
    plt.close(fig)

def log_fold_to_wandb(fold_num, metrics_dict, class_names, y_true, y_pred):
    """
    Logs fold-level metrics and confusion matrix to W&B.
    Argument:
        metrics_dict: dict with 'acc', 'prec', 'rec', 'f1', and 'report'.
    """
    acc    = metrics_dict["acc"]
    prec   = metrics_dict["prec"]
    rec    = metrics_dict["rec"]
    f1     = metrics_dict["f1"]
    report = metrics_dict["report"]

    fold_logs = {
        f"fold{fold_num}_accuracy" : acc,
        f"fold{fold_num}_precision": prec,
        f"fold{fold_num}_recall"   : rec,
        f"fold{fold_num}_f1"       : f1,
    }
    
    # Per-class metrics
    for cls in class_names:
        if cls in report:
            fold_logs[f"fold{fold_num}_{cls}_f1"]        = report[cls]["f1-score"]
            fold_logs[f"fold{fold_num}_{cls}_precision"]  = report[cls]["precision"]
            fold_logs[f"fold{fold_num}_{cls}_recall"]     = report[cls]["recall"]

    wandb.log(fold_logs)
    
    # Confusion Matrix
    log_confusion_matrix(
        y_true, y_pred, class_names,
        f"fold{fold_num}_confusion_matrix",
    )

def log_summary_table(fold_results, window_name, class_names):
    """
    Creates and logs a W&B Summary Table with granular fold-level results.
    Arguments:
        fold_results: A list of dicts from metrics.calculate_metrics().
    """
    columns = ["Fold", "Accuracy", "Precision", "Recall", "F1 Macro"]
    # Add per-class columns
    for cls in class_names:
        columns.append(f"{cls} F1")

    table = wandb.Table(columns=columns)

    for i, res in enumerate(fold_results):
        row = [
            f"Fold {i+1}", 
            round(res["acc"], 4),
            round(res["prec"], 4),
            round(res["rec"], 4),
            round(res["f1"], 4),
        ]
        # Add per-class F1 for this fold
        for cls in class_names:
            f1_cls = res["report"].get(cls, {}).get("f1-score", 0)
            row.append(round(f1_cls, 4))
            
        table.add_data(*row)

    wandb.log({f"Evaluation_Summary_Table_{window_name}": table})

def log_final_summary(summary_metrics):
    """
    Logs the final cross-validated averages and standard deviations.
    """
    wandb.log({
        "final_avg_accuracy" : summary_metrics["avg_acc"],
        "final_std_accuracy" : summary_metrics["std_acc"],
        "final_avg_precision": summary_metrics["avg_prec"],
        "final_avg_recall"   : summary_metrics["avg_rec"],
        "final_avg_f1"       : summary_metrics["avg_f1"],
        "final_std_f1"       : summary_metrics["std_f1"],
    })
