#!/usr/bin/env python
"""Evaluate YOLO mAP@0.5, or inspect model metadata when no dataset is supplied.

Usage
-----
    # Full validation (requires a YOLO-format data.yaml)
    python evaluation/eval_map.py --data path/to/data.yaml --model models/best.pt

    # Metadata-only inspection (never crashes when --data is omitted/missing)
    python evaluation/eval_map.py --model models/best.pt

The metadata path prints classes, parameter count, training image size, layer
summary and clear instructions for running validation later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_URL = (
    "https://universe.roboflow.com/roboflow-jvuqo/"
    "football-players-detection-3zvbc/dataset/1"
)


def load_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is required. Install it with: pip install ultralytics")
        raise SystemExit(1)

    try:
        return YOLO(model_path)
    except Exception as exc:
        print(f"Error loading model {model_path}: {exc}")
        raise SystemExit(1)


def inspect_model(model_path: str):
    """Print model metadata even when no validation dataset is available."""
    print("=" * 68)
    print("MODEL METADATA")
    print("=" * 68)

    model = load_model(model_path)
    names = getattr(model, "names", {}) or {}
    class_list = [f"{class_id}: {name}" for class_id, name in names.items()]

    parameters = sum(parameter.numel() for parameter in model.model.parameters())

    checkpoint = getattr(model, "ckpt", {}) or {}
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    model_args = getattr(model.model, "args", {}) or {}
    imgsz = train_args.get("imgsz", model_args.get("imgsz", 640))

    print(f"Model path        : {Path(model_path).resolve()}")
    print(f"Task              : {getattr(model, 'task', 'detect')}")
    print(f"Classes ({len(names)})       : {', '.join(class_list) if class_list else 'unknown'}")
    print(f"Parameters        : {parameters:,} ({parameters / 1e6:.2f}M)")
    print(f"Training imgsz    : {imgsz}")
    print(f"Modules           : {len(list(model.model.modules()))}")

    print("\nLayer summary:")
    try:
        model.model.info()
    except Exception as exc:  # pragma: no cover - depends on Ultralytics version
        print(f"  (layer summary unavailable: {exc})")

    print("=" * 68)
    return model


def print_dataset_instructions() -> None:
    print("\n" + "=" * 68)
    print("HOW TO RUN mAP@0.5 VALIDATION")
    print("=" * 68)
    print("1. Download a YOLO-format football dataset:")
    print(f"   {DATASET_URL}")
    print("\n2. Run this script with the data.yaml:")
    print(
        "   python evaluation/eval_map.py --data path/to/football-players-detection/data.yaml "
        "--model models/best.pt --split val --imgsz 640 --batch 16 --device 0"
    )
    print("\n   Or use the Ultralytics CLI directly:")
    print(
        "   yolo val model=models/best.pt data=path/to/data.yaml "
        "split=val imgsz=640 batch=16 device=0"
    )
    print(
        "\nThe data.yaml must contain the train/val/test image directories and "
        "the class names used during training."
    )
    print("=" * 68)


def run_validation(model, data_yaml: str, split: str, imgsz: int, batch: int, device: str):
    print("\n" + "=" * 68)
    print("YOLO VALIDATION")
    print("=" * 68)
    print(f"Dataset : {Path(data_yaml).resolve()}")
    print(f"Split   : {split}")
    print(f"Images  : {imgsz}  Batch: {batch}  Device: {device}")

    try:
        results = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=False,
        )
    except Exception as exc:
        print(f"\nValidation failed: {exc}")
        print_dataset_instructions()
        return None

    try:
        map50 = float(results.box.map50)
        map50_95 = float(results.box.map)
    except (AttributeError, TypeError):
        print("\nValidation completed but mAP metrics were not returned.")
        print(results)
        return None

    print("\n" + "=" * 68)
    print(f"mAP@0.5      : {map50:.4f}")
    print(f"mAP@0.5:0.95 : {map50_95:.4f}")
    print("=" * 68)
    if map50 >= 0.80:
        print("PASSES target (>= 0.80)")
    elif map50 >= 0.70:
        print("ACCEPTABLE (>= 0.70)")
    else:
        print("BELOW target (< 0.70)")
    return map50


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate mAP@0.5 for BallTracker, or inspect model metadata"
    )
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO model")
    parser.add_argument(
        "--data",
        default=None,
        help="Optional path to a YOLO data.yaml. Omitting it prints model metadata.",
    )
    parser.add_argument("--split", default="val", help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="GPU device (0, 1, ...) or cpu")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    model = inspect_model(args.model)

    if not args.data:
        print("\nNo --data supplied; skipped mAP validation.")
        print_dataset_instructions()
        return

    if not Path(args.data).exists():
        print(f"\nData YAML not found at {args.data}; skipped mAP validation.")
        print_dataset_instructions()
        return

    run_validation(model, args.data, args.split, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
