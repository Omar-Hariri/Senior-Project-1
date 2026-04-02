"""
YOLOv8 Training Script
Entry point: python -m src.training.train_yolo --config configs/yolo_config.yaml
Responsibility: Training loop + Weights & Biases (W&B) logging.
Functional style.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb

from src.data.adms_loader import load_yaml_config
from src.models.deep.yolo import build_yolo_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_yolo(config_path: str) -> None:
    """Main training loop for YOLOv8."""
    cfg = load_yaml_config(config_path)

    # ── Seed ──────────────────────────────────────────────────────────────
    seed = cfg["training"]["seed"]
    set_seed(seed)

    # ── Paths ─────────────────────────────────────────────────────────────
    processed_dir   = Path(cfg["dataset"]["train"]).parent  # data/processed/adms
    data_yaml_path  = processed_dir / "data.yaml"
    models_dir      = Path(cfg["artifacts"]["models"])
    metrics_dir     = Path(cfg["artifacts"]["metrics"])
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Weights & Biases ─────────────────────────────────────────────────
    wandb.init(
        project=cfg["logging"]["project_name"],
        name=cfg["logging"]["run_name"],
        config=cfg,
        mode=cfg["logging"].get("wandb_mode", "online"),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_yolo_model(variant=cfg["model"]["variant"])

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n[Training] Starting YOLOv8 training...")
    print(f"  Config : {config_path}")
    print(f"  Data   : {data_yaml_path}")
    print(f"  Epochs : {cfg['training']['epochs']}")
    print(f"  Device : {cfg['training']['device']}\n")

    results = model.train(
        data        = str(data_yaml_path),
        epochs      = cfg["training"]["epochs"],
        imgsz       = cfg["model"]["image_size"],
        batch       = cfg["training"]["batch_size"],
        lr0         = cfg["training"]["learning_rate"],
        optimizer   = cfg["training"]["optimizer"],
        seed        = seed,
        workers     = cfg["training"]["workers"],
        device      = cfg["training"]["device"],
        patience    = cfg["training"]["patience"],
        augment     = cfg["training"]["augment"],
        project     = str(cfg["artifacts"]["experiments"]),
        name        = cfg["logging"]["run_name"],
        exist_ok    = True,
        verbose     = True,
    )

    # ── Log Metrics ───────────────────────────────────────────────────
    metrics_summary = {
        "mAP50":      float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50_95":   float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision":  float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall":     float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    wandb.log(metrics_summary)

    metrics_json_path = metrics_dir / "train_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # ── Save & Log Weights ────────────────────────────────────────────
    exp_dir = Path(cfg["artifacts"]["experiments"]) / cfg["logging"]["run_name"]
    best_pt  = exp_dir / "weights" / "best.pt"
    last_pt  = exp_dir / "weights" / "last.pt"

    for pt_file in [best_pt, last_pt]:
        if pt_file.exists():
            dest = models_dir / pt_file.name
            shutil.copy(pt_file, dest)
            # Log weight as artifact to W&B
            artifact = wandb.Artifact(f"yolo-weights-{pt_file.stem}", type="model")
            artifact.add_file(str(dest))
            wandb.log_artifact(artifact)
            print(f"[Training] Saved and Logged {pt_file.name} → {dest}")

    # ── Log Plots ─────────────────────────────────────────────────────
    for plot_name in ["confusion_matrix.png", "PR_curve.png", "results.png"]:
        plot_path = exp_dir / plot_name
        if plot_path.exists():
             wandb.log({plot_name.split('.')[0]: wandb.Image(str(plot_path))})

    print(f"\n[Training] Complete!")
    print(f"  mAP50     : {metrics_summary['mAP50']:.4f}")
    print(f"  mAP50-95  : {metrics_summary['mAP50_95']:.4f}")
    print(f"  Precision : {metrics_summary['precision']:.4f}")
    print(f"  Recall    : {metrics_summary['recall']:.4f}")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on ADMS dataset")
    parser.add_argument("--config", type=str, default="configs/yolo_config.yaml")
    args = parser.parse_args()
    train_yolo(args.config)
