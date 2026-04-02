"""
YOLOv8 Inference Script
Entry point: python -m src.inference.predict_yolo --input path_to_video
Responsibility: Load model and run inference. Output must be compatible with Fusion Logic module.
Functional style.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2

from src.data.adms_loader import load_yaml_config, CLASS_NAMES
from src.models.deep.yolo import build_yolo_model


def build_fusion_output(frame_id: int, results) -> Dict:
    """
    Convert YOLO results into Fusion Logic-compatible JSON.
    """
    detections = []
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_idx    = int(box.cls[0].item())
            confidence = round(float(box.conf[0].item()), 4)
            x, y, w, h = box.xywh[0].tolist()
            detections.append({
                "class":      CLASS_NAMES[cls_idx],
                "confidence": confidence,
                "bbox":       [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            })
    return {"frame_id": frame_id, "detections": detections}


def run_prediction(
    input_path: str,
    config_path: str = "configs/yolo_config.yaml",
    conf_threshold: float = 0.5,
    save_output: bool = True,
    output_json: str = "artifacts/metrics/yolo/inference_output.json",
    is_image: bool = False,
) -> List[Dict]:
    """Runs YOLOv8 prediction on video or image."""
    cfg = load_yaml_config(config_path)
    models_dir = Path(cfg["artifacts"]["models"])
    best_pt    = models_dir / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"Model not found at {best_pt}. Run training first.")

    model = build_yolo_model(variant=cfg["model"]["variant"], weights_path=str(best_pt))

    # SINGLE IMAGE MODE
    if is_image:
        results = model.predict(source=input_path, conf=conf_threshold, imgsz=640, verbose=False)
        output  = build_fusion_output(0, results)
        print(json.dumps(output, indent=2))
        return [output]

    # VIDEO MODE
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open input: {input_path}")

    all_outputs = []
    frame_id    = 0
    total_ms    = 0.0

    print(f"\n[Inference] Processing: {input_path}")
    print(f"[Inference] Confidence threshold: {conf_threshold}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.predict(
            source = frame,
            conf   = conf_threshold,
            imgsz  = cfg["model"]["image_size"],
            verbose= False,
        )
        t1 = time.perf_counter()
        total_ms += (t1 - t0) * 1000

        output = build_fusion_output(frame_id, results)
        all_outputs.append(output)

        if output["detections"]:
            print(f"  Frame {frame_id:>5}: {output['detections']}")

        frame_id += 1

    cap.release()
    avg_ms = total_ms / max(frame_id, 1)
    fps    = 1000 / avg_ms if avg_ms > 0 else 0

    print(f"\n[Inference] Done. {frame_id} frames processed.")
    print(f"  Avg inference: {avg_ms:.1f} ms/frame  ({fps:.1f} FPS)")

    if save_output:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"  Output saved → {output_json}")

    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 inference for Driver Monitoring System")
    parser.add_argument("--input",  type=str, required=True, help="Path to video file or image")
    parser.add_argument("--config", type=str, default="configs/yolo_config.yaml")
    parser.add_argument("--conf",   type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--image",  action="store_true", help="Treat input as single image")
    args = parser.parse_args()

    run_prediction(args.input, args.config, args.conf, is_image=args.image)
