#!/usr/bin/env python
"""
Processing Speed (FPS) Benchmark Script for BallTracker

This script benchmarks the end-to-end processing speed of the tracking pipeline
on GPU with FP16 inference.

Usage:
    python benchmark_speed.py --video input_videos/08fd33_4.mp4 --model models/best.pt --frames 100

Expected output: FPS, elapsed time for 100 frames
"""

import argparse
import time
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))

from trackers.tracker import Tracker
from utils.video_utils import read_video


def benchmark_detection(tracker, frames, num_runs=3):
    """Benchmark just the detection step."""
    import numpy as np
    print("Benchmarking detection only...")

    # Warmup
    _ = tracker.detect_frames(frames[:5])

    times = []
    for run in range(num_runs):
        start = time.perf_counter()
        _ = tracker.detect_frames(frames)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({len(frames)/elapsed:.1f} FPS)")

    avg_time = np.mean(times)
    avg_fps = len(frames) / avg_time
    print(f"  Average: {avg_time:.3f}s ({avg_fps:.1f} FPS)")

    return avg_fps


def benchmark_full_tracking(tracker, frames, num_runs=3):
    """Benchmark full tracking pipeline (detection + tracking)."""
    import numpy as np
    print("\nBenchmarking full tracking (detection + ByteTrack)...")

    # Warmup
    _ = tracker.get_object_tracks(frames[:5])

    times = []
    for run in range(num_runs):
        start = time.perf_counter()
        _ = tracker.get_object_tracks(frames)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({len(frames)/elapsed:.1f} FPS)")

    avg_time = np.mean(times)
    avg_fps = len(frames) / avg_time
    print(f"  Average: {avg_time:.3f}s ({avg_fps:.1f} FPS)")

    return avg_fps


def print_gpu_info():
    """Print GPU information."""
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")

        # Memory info
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"VRAM Total: {total_mem:.1f} GB")
        print(f"VRAM Allocated: {allocated:.1f} GB")
        print(f"VRAM Reserved: {reserved:.1f} GB")

        # cuDNN settings
        print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    else:
        print("CUDA Available: No (running on CPU)")

    print("=" * 60)


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description='Benchmark processing speed (FPS)')
    parser.add_argument('--video', default='input_videos/08fd33_4.mp4', help='Input video path')
    parser.add_argument('--model', default='models/best.pt', help='YOLO model path')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to benchmark')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--detection-only', action='store_true', help='Benchmark detection only')

    args = parser.parse_args()

    print(f"Benchmarking BallTracker processing speed")
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Frames: {args.frames}")
    print(f"Runs: {args.runs}")
    print("-" * 60)

    # Print GPU info
    print_gpu_info()

    # Enable cuDNN benchmark and TF32 for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load video
    video_frames = read_video(args.video)
    if len(video_frames) < args.frames:
        print(f"Warning: Video has only {len(video_frames)} frames")
        args.frames = len(video_frames)

    benchmark_frames = video_frames[:args.frames]
    print(f"\nUsing {len(benchmark_frames)} frames for benchmark")
    print(f"Frame resolution: {benchmark_frames[0].shape[1]}x{benchmark_frames[0].shape[0]}")

    # Initialize tracker
    tracker = Tracker(args.model)

    if args.detection_only:
        benchmark_detection(tracker, benchmark_frames, args.runs)
    else:
        benchmark_full_tracking(tracker, benchmark_frames, args.runs)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()