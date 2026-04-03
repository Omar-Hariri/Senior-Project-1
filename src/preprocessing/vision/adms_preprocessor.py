"""
ADMS Preprocessor
Responsibility: Split dataset into train/val/test and organize into YOLO folder structure.
Does NOT do training or model logic.
Functional style.

Output structure:
    data/processed/adms/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/
        data.yaml
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.data.vision.adms_loader import get_adms_image_paths, CLASS_NAMES, CLASS_TO_IDX


DUMMY_BBOX = "0.5 0.5 0.9 0.9"   # fallback if no real annotations exist


def _make_split_dirs(processed_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (processed_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def _split_data(items: List, ratios: Dict[str, float]) -> Tuple[List, List, List]:
    random.shuffle(items)
    n = len(items)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def _write_yolo_label(label_path: Path, class_idx: int, image_path: Path) -> None:
    """Write YOLO-format label file. Uses real .txt if present, else dummy bbox."""
    src_label = image_path.with_suffix(".txt")
    if src_label.exists():
        shutil.copy(src_label, label_path)
    else:
        with open(label_path, "w") as f:
            f.write(f"{class_idx} {DUMMY_BBOX}\n")


def run_adms_preprocessor(
    raw_data_dir: str = "data/raw/adms",
    processed_dir: str = "data/processed/adms",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Main function to preprocess ADMS dataset and prepare for YOLO training.
    """
    print("\n[Preprocessor] Starting ADMS dataset preparation...")
    random.seed(seed)
    processed_path = Path(processed_dir)
    _make_split_dirs(processed_path)

    # Load paths using functional loader
    class_paths = get_adms_image_paths(raw_data_dir)
    split_counts = {"train": 0, "val": 0, "test": 0}
    ratios = {"train": train_ratio, "val": val_ratio}

    for cls_name, img_paths in class_paths.items():
        cls_idx = CLASS_TO_IDX[cls_name]
        train_imgs, val_imgs, test_imgs = _split_data(img_paths, ratios)

        for split_name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for img_path in imgs:
                dst_img = processed_path / split_name / "images" / img_path.name
                dst_lbl = processed_path / split_name / "labels" / img_path.with_suffix(".txt").name
                shutil.copy(img_path, dst_img)
                _write_yolo_label(dst_lbl, cls_idx, img_path)
                split_counts[split_name] += 1

        print(f"  {cls_name}: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

    # Write data.yaml
    data_yaml = {
        "path": str(processed_path.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = processed_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"  data.yaml written → {yaml_path}")
    print(f"\n[Preprocessor] Done.")
    print(f"  Train: {split_counts['train']} | Val: {split_counts['val']} | Test: {split_counts['test']}")


if __name__ == "__main__":
    run_adms_preprocessor()
