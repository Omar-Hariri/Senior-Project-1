# src/utils/plotting.py
# =====================
# Utilities for standardizing visualization across models.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    """
    Creates a classic heatmap confusion matrix using Seaborn.
    Returns the matplotlib figure for W&B logging.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        normalize: If True, show proportions (0-1). If False, show absolute counts.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        # Normalize by row (true labels)
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

def log_classic_confusion_matrix(y_true, y_pred, class_names, log_key, normalize=False):
    """
    Plots the CM and logs it to W&B as an image.
    """
    fig = plot_confusion_matrix(y_true, y_pred, class_names, normalize=normalize)
    wandb.log({log_key: wandb.Image(fig)})
    plt.close(fig)
