"""
YOLO Model Definition
Responsibility: Define and load the YOLOv8 model only.
Functional style.
"""

from pathlib import Path
from ultralytics import YOLO


def build_yolo_model(variant: str = "yolov8n", weights_path: str = None) -> YOLO:
    """
    Constructs and returns a YOLOv8 model instance.
    
    Args:
        variant: YOLOv8 variant (nano recommended for detection).
        weights_path: Path to .pt weights. If None, loads pretrained COCO weights.
    """
    if weights_path and Path(weights_path).exists():
        print(f"[YOLOModel] Loading weights from: {weights_path}")
        return YOLO(weights_path)
    
    print(f"[YOLOModel] Loading {variant}.pt")
    return YOLO(f"{variant}.pt")
