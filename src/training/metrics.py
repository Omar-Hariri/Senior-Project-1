import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculates fold-level metrics and per-class reports.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "report": report
    }

def calculate_window_averages(fold_results):
    """
    Computes cross-validated average metrics for a set of fold results.
    Arguments:
        fold_results: A list of dicts with keys 'acc', 'prec', 'rec', 'f1'.
    """
    if not fold_results:
        return {}
        
    avg_acc  = np.mean([r["acc"]  for r in fold_results])
    avg_prec = np.mean([r["prec"] for r in fold_results])
    avg_rec  = np.mean([r["rec"]  for r in fold_results])
    avg_f1   = np.mean([r["f1"]   for r in fold_results])
    std_acc  = np.std([r["acc"]   for r in fold_results])
    std_f1   = np.std([r["f1"]    for r in fold_results])
    
    return {
        "avg_acc": avg_acc, "std_acc": std_acc,
        "avg_prec": avg_prec, "avg_rec": avg_rec, 
        "avg_f1": avg_f1, "std_f1": std_f1
    }
