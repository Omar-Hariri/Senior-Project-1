import yaml
import numpy as np
from pathlib import Path

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_fold_data(fold_dir, data_type="ml"):
    """
    Loads features and labels for a specific fold.
    Arguments:
        fold_dir: Path pointing to the specific fold directory (e.g. fold1).
        data_type: 'ml' for tabular/flattened data, 'lstm' for sequential data.
    """
    X_train = np.load(fold_dir / data_type / "X_train.npy")
    y_train = np.load(fold_dir / data_type / "y_train.npy")
    X_test  = np.load(fold_dir / data_type / "X_test.npy")
    y_test  = np.load(fold_dir / data_type / "y_test.npy")
    return X_train, y_train, X_test, y_test
