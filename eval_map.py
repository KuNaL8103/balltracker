#!/usr/bin/env python
"""
mAP@0.5 Evaluation Script for BallTracker

This script evaluates the detection accuracy of the YOLO model using mAP@0.5 metric.
Requires a validation dataset in YOLO format (images + labels).

Usage:
    python eval_map.py --data path/to/data.yaml --model models/best.pt

Or run directly with Ultralytics CLI:
    yolo val model=models/best.pt data=path/to/football-players-detection-3zvbc/data.yaml split=val imgsz=640 batch=16 device=0

Expected output: metrics/mAP50(B) — should be ~0.82
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_yolo_val(model_path, data_yaml, split='val', imgsz=640, batch=16, device=0):
    """Run YOLO validation and return mAP metrics."""
    cmd = [
        'yolo', 'val',
        f'model={model_path}',
        f'data={data_yaml}',
        f'split={split}',
        f'imgsz={imgsz}',
        f'batch={batch}',
        f'device={device}'
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running YOLO validation:")
        print(result.stderr)
        return None

    return result.stdout


def parse_map_output(output):
    """Parse mAP50 from YOLO validation output."""
    import re

    # Look for mAP50 pattern in output
    # Typical format: "metrics/mAP50(B): 0.823"
    patterns = [
        r'metrics/mAP50\(B\):\s*([\d.]+)',
        r'mAP50:\s*([\d.]+)',
        r'mAP@0\.5:\s*([\d.]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))

    # If not found, print full output for debugging
    print("Could not parse mAP50 from output. Full output:")
    print(output)
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate mAP@0.5 for BallTracker YOLO model')
    parser.add_argument('--model', default='models/best.pt', help='Path to YOLO model')
    parser.add_argument('--data', required=True, help='Path to data.yaml (validation dataset)')
    parser.add_argument('--split', default='val', help='Dataset split to evaluate (val/test)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='0', help='GPU device (0, 1, ...) or cpu')

    args = parser.parse_args()

    # Verify model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    if not Path(args.data).exists():
        print(f"Error: Data YAML not found at {args.data}")
        print("Download a Roboflow dataset in YOLO format first.")
        sys.exit(1)

    print(f"Evaluating mAP@0.5 for model: {args.model}")
    print(f"Using dataset: {args.data}")
    print(f"Split: {args.split}, Image size: {args.imgsz}, Batch: {args.batch}, Device: {args.device}")
    print("-" * 60)

    output = run_yolo_val(args.model, args.data, args.split, args.imgsz, args.batch, args.device)

    if output:
        map50 = parse_map_output(output)
        if map50 is not None:
            print(f"\n{'='*60}")
            print(f"mAP@0.5 (metrics/mAP50(B)): {map50:.4f}")
            print(f"{'='*60}")

            # Target benchmark
            if map50 >= 0.80:
                print("✓ PASSES target (>= 0.80)")
            elif map50 >= 0.70:
                print("~ ACCEPTABLE (>= 0.70)")
            else:
                print("✗ BELOW target (< 0.70)")


if __name__ == '__main__':
    main()