#!/usr/bin/env python
"""
Ball Interpolation Recovery Evaluation Script for BallTracker

This script evaluates the ball position interpolation by comparing
missing ball detections before and after interpolation.

Usage:
    python eval_ball_interpolation.py --video input_videos/08fd33_4.mp4 --model models/best.pt

Expected output: Missing ball frames before/after, recovery rate percentage
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from trackers.tracker import Tracker
from utils.video_utils import read_video


def count_missing_ball_frames(ball_tracks):
    """Count frames where ball detection is missing."""
    missing = 0
    for frame_ball in ball_tracks:
        # Ball track uses track_id=1
        if not frame_ball or 1 not in frame_ball:
            missing += 1
        elif 'bbox' not in frame_ball[1] or not frame_ball[1]['bbox']:
            missing += 1
    return missing


def analyze_ball_track_quality(ball_tracks):
    """Analyze ball track quality in detail."""
    import numpy as np
    total_frames = len(ball_tracks)
    missing = 0
    present = 0
    bboxes = []

    for frame_ball in ball_tracks:
        if not frame_ball or 1 not in frame_ball or 'bbox' not in frame_ball[1] or not frame_ball[1]['bbox']:
            missing += 1
        else:
            present += 1
            bboxes.append(frame_ball[1]['bbox'])

    print(f"Total frames: {total_frames}")
    print(f"Frames with ball: {present} ({present/total_frames*100:.1f}%)")
    print(f"Frames missing ball: {missing} ({missing/total_frames*100:.1f}%)")

    if bboxes:
        bboxes = np.array(bboxes)
        print(f"\nBall bbox statistics:")
        print(f"  x1: mean={bboxes[:,0].mean():.1f}, std={bboxes[:,0].std():.1f}")
        print(f"  y1: mean={bboxes[:,1].mean():.1f}, std={bboxes[:,1].std():.1f}")
        print(f"  x2: mean={bboxes[:,2].mean():.1f}, std={bboxes[:,2].std():.1f}")
        print(f"  y2: mean={bboxes[:,3].mean():.1f}, std={bboxes[:,3].std():.1f}")

    # Find gaps in detection
    gaps = []
    current_gap = 0
    for frame_ball in ball_tracks:
        if not frame_ball or 1 not in frame_ball or 'bbox' not in frame_ball[1] or not frame_ball[1]['bbox']:
            current_gap += 1
        else:
            if current_gap > 0:
                gaps.append(current_gap)
                current_gap = 0
    if current_gap > 0:
        gaps.append(current_gap)

    if gaps:
        print(f"\nDetection gaps (consecutive missing frames):")
        print(f"  Number of gaps: {len(gaps)}")
        print(f"  Max gap: {max(gaps)} frames")
        print(f"  Mean gap: {np.mean(gaps):.1f} frames")
        print(f"  Gap distribution: {gaps}")

    return missing, present


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description='Evaluate ball interpolation recovery')
    parser.add_argument('--video', default='input_videos/08fd33_4.mp4', help='Input video path')
    parser.add_argument('--model', default='models/best.pt', help='YOLO model path')
    parser.add_argument('--stub', default='stubs/track_stubs.pkl', help='Tracker stub path')

    args = parser.parse_args()

    print(f"Evaluating ball interpolation recovery")
    print(f"Video: {args.video}")
    print("-" * 60)

    # Load video and tracks
    video_frames = read_video(args.video)

    tracker = Tracker(args.model)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=args.stub
    )

    # Analyze BEFORE interpolation
    print("\n--- BEFORE Interpolation ---")
    missing_before, present_before = analyze_ball_track_quality(tracks['ball'])

    # Interpolate
    print("\n--- Interpolating ---")
    interpolated = tracker.interpolate_ball_positions(tracks['ball'])

    # Analyze AFTER interpolation
    print("\n--- AFTER Interpolation ---")
    missing_after, present_after = analyze_ball_track_quality(interpolated)

    # Recovery rate
    recovered = missing_before - missing_after
    recovery_rate = (recovered / missing_before * 100) if missing_before > 0 else 100

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Missing before: {missing_before}")
    print(f"  Missing after:  {missing_after}")
    print(f"  Recovered:      {recovered}")
    print(f"  Recovery rate:  {recovery_rate:.1f}%")
    print("=" * 60)

    if recovery_rate >= 90:
        print("✓ EXCELLENT recovery (>= 90%)")
    elif recovery_rate >= 70:
        print("~ GOOD recovery (70-90%)")
    else:
        print("✗ POOR recovery (< 70%)")


if __name__ == '__main__':
    main()