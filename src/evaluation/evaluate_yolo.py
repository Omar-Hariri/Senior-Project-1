"""
YOLOv8 Evaluation Script
Entry point: python -m src.evaluation.evaluate_yolo --config configs/yolo_config.yaml
Responsibility: Compute and save all evaluation metrics.
Functional style.
"""

import argparse
import json
from pathlib import Path

import wandb
import numpy as np

from src.data.adms_loader import load_yaml_config, CLASS_NAMES
from src.models.deep.yolo import build_yolo_model


def run_evaluation(config_path: str) -> None:
    """Runs YOLOv8 evaluation on the test split."""
    cfg = load_yaml_config(config_path)

    processed_dir  = Path(cfg["dataset"]["train"]).parent
    data_yaml_path = processed_dir / "data.yaml"
    models_dir     = Path(cfg["artifacts"]["models"])
    metrics_dir    = Path(cfg["artifacts"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_pt = models_dir / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"No trained model found at {best_pt}. Run training first.")

    print(f"\n[Evaluation] Loading model: {best_pt}")
    model = build_yolo_model(variant=cfg["model"]["variant"], weights_path=str(best_pt))

    print(f"[Evaluation] Running validation on test split...")
    metrics = model.val(
        data   = str(data_yaml_path),
        split  = "test",
        imgsz  = cfg["model"]["image_size"],
        batch  = cfg["training"]["batch_size"],
        device = cfg["training"]["device"],
        verbose= True,
    )

    # ── Per-class metrics ──────────────────────────────────────────────────
    results_dict   = metrics.results_dict
    per_class_ap50 = metrics.box.ap50
    per_class_ap   = metrics.box.ap
    per_class_p    = metrics.box.p
    per_class_r    = metrics.box.r
    per_class_f1   = (2 * per_class_p * per_class_r / (per_class_p + per_class_r + 1e-8))

    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        per_class[cls] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall":    round(float(per_class_r[i]), 4),
            "f1":        round(float(per_class_f1[i]), 4),
            "mAP50":     round(float(per_class_ap50[i]), 4),
            "mAP50_95":  round(float(per_class_ap[i]), 4),
        }

    summary = {
        "overall": {
            "mAP50":     round(float(results_dict.get("metrics/mAP50(B)", 0)), 4),
            "mAP50_95":  round(float(results_dict.get("metrics/mAP50-95(B)", 0)), 4),
            "precision": round(float(results_dict.get("metrics/precision(B)", 0)), 4),
            "recall":    round(float(results_dict.get("metrics/recall(B)", 0)), 4),
        },
        "per_class": per_class,
    }

    # ── Print table ────────────────────────────────────────────────────────
    print("\n=== Evaluation Results ===")
    print(f"{'Class':<20} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'mAP50-95':>9}")
    print("-" * 58)
    for cls, m in per_class.items():
        print(f"{cls:<20} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f} {m['mAP50']:>7.4f} {m['mAP50_95']:>9.4f}")
    print("-" * 58)
    o = summary["overall"]
    print(f"{'OVERALL':<20} {o['precision']:>6.4f} {o['recall']:>6.4f} {'—':>6} {o['mAP50']:>7.4f} {o['mAP50_95']:>9.4f}")

    # ── Save metrics JSON ──────────────────────────────────────────────────
    out_path = metrics_dir / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── W&B logging ─────────────────────────────────────────────────────
    wandb.init(
        project=cfg["logging"]["project_name"],
        name=cfg["logging"]["run_name"] + "-eval",
        config=cfg,
        job_type="evaluation"
    )
    wandb.log(o)
    wandb.save(str(out_path))
    print(f"[Evaluation] Logged to W&B.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on ADMS test set")
    parser.add_argument("--config", type=str, default="configs/yolo_config.yaml")
    args = parser.parse_args()
    run_evaluation(args.config)
