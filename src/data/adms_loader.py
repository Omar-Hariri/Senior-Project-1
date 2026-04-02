"""
ADMS Dataset Loader
Responsibility: Load raw ADMS dataset from disk and return structured paths.
Does NOT do preprocessing or training logic.
Functional style.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import yaml


CLASS_NAMES = [
    "Eye_Closure",
    "Yawning",
    "Seat_Belt",
    "Smoking",
    "Phone_Usage",
]

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def get_adms_image_paths(raw_data_dir: str = "data/raw/adms") -> Dict[str, List[Path]]:
    """
    Returns a dict: {class_name: [list of image Paths]}
    Assumes raw data is organized as:
        data/raw/adms/Eye_Closure/img1.jpg
        ...
    """
    raw_dir = Path(raw_data_dir)
    if not raw_dir.exists():
         raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    class_paths: Dict[str, List[Path]] = {}
    for cls in CLASS_NAMES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"[WARNING] Class folder not found: {cls_dir}")
            class_paths[cls] = []
            continue
        
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        class_paths[cls] = images
        print(f"  {cls}: {len(images)} images found")
    
    return class_paths


def get_all_adms_image_paths(raw_data_dir: str = "data/raw/adms") -> List[Tuple[Path, str]]:
    """Returns flat list of (image_path, class_name) tuples."""
    class_paths = get_adms_image_paths(raw_data_dir)
    all_images = []
    for cls, paths in class_paths.items():
        for p in paths:
            all_images.append((p, cls))
    return all_images


def print_adms_summary(raw_data_dir: str = "data/raw/adms") -> None:
    """Print dataset summary."""
    class_paths = get_adms_image_paths(raw_data_dir)
    total = sum(len(v) for v in class_paths.values())
    print("\n=== ADMS Dataset Summary ===")
    for cls, paths in class_paths.items():
        print(f"  {cls:<20} {len(paths):>5} images")
    print(f"  {'TOTAL':<20} {total:>5} images")
    print("=" * 30)


def load_yaml_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    print_adms_summary()
