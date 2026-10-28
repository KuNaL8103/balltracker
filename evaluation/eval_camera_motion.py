#!/usr/bin/env python
"""
Camera Motion / Trajectory Drift Evaluation Script for BallTracker

This script evaluates:
1. Raw camera movement (pixels/frame) from optical flow
2. Residual drift after homography correction (meters/frame)

Usage:
    python evaluation/eval_camera_motion.py --video input_videos/08fd33_4.mp4 --model models/best.pt --frames 500

Expected output: Raw camera drift (px/frame), Residual drift after homography (m/frame)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trackers.tracker import Tracker
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from utils.video_utils import read_video
from utils.bbox_utils import measure_distance


def main():
    parser = argparse.ArgumentParser(description='Evaluate camera motion and trajectory drift')
    parser.add_argument('--video', default='input_videos/08fd33_4.mp4', help='Input video path')
    parser.add_argument('--model', default='models/best.pt', help='YOLO model path')
    parser.add_argument('--frames', type=int, default=500, help='Number of frames to evaluate')
    parser.add_argument('--stub', default='stubs/track_stubs.pkl', help='Tracker stub path')
    parser.add_argument('--cam-stub', default='stubs/camera_movement_stub.pkl', help='Camera movement stub path')

    args = parser.parse_args()

    print(f"Evaluating camera motion / trajectory drift")
    print(f"Video: {args.video}")
    print(f"Frames to evaluate: {args.frames}")
    print("-" * 60)

    # Load video and tracks
    video_frames = read_video(args.video)
    if len(video_frames) < args.frames:
        args.frames = len(video_frames)

    tracker = Tracker(args.model)
    tracks = tracker.get_object_tracks(
        video_frames[:args.frames],
        read_from_stub=True,
        stub_path=args.stub
    )

    # Add positions
    tracker.add_position_to_tracks(tracks)

    # 1. Raw camera movement (pixels/frame)
    print("\n--- Raw Camera Movement ---")
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_movement = cam_est.get_camera_movement(
        video_frames[:args.frames],
        read_from_stub=True,
        stub_path=args.cam_stub
    )

    # Calculate raw drift statistics
    movements = [np.linalg.norm(m) for m in cam_movement[1:] if np.linalg.norm(m) > 0]
    if movements:
        raw_drift_mean = np.mean(movements)
        raw_drift_max = np.max(movements)
        raw_drift_std = np.std(movements)
        print(f"Raw camera drift (px/frame):")
        print(f"  Mean: {raw_drift_mean:.2f}")
        print(f"  Max:  {raw_drift_max:.2f}")
        print(f"  Std:  {raw_drift_std:.2f}")
        print(f"  Frames with movement: {len(movements)}/{args.frames-1}")
    else:
        print("No camera movement detected")
        raw_drift_mean = 0

    # 2. Apply camera movement adjustment
    cam_est.add_adjust_positions_to_tracks(tracks, cam_movement)

    # 3. Apply view transformation (homography)
    print("\n--- Homography Correction ---")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 4. Measure residual drift on "stationary" players
    print("\n--- Residual Drift After Homography ---")
    residuals = []

    # Sample frames for evaluation
    eval_frames = min(100, args.frames - 1)

    for frame_num in range(1, eval_frames):
        for track_id, track in tracks['players'][frame_num].items():
            if 'position_transformed' in track and track['position_transformed'] is not None:
                # Check if track exists in previous frame
                if track_id in tracks['players'][frame_num - 1]:
                    prev_track = tracks['players'][frame_num - 1][track_id]
                    if 'position_transformed' in prev_track and prev_track['position_transformed'] is not None:
                        prev_pos = prev_track['position_transformed']
                        curr_pos = track['position_transformed']
                        dist = measure_distance(prev_pos, curr_pos)

                        # Only consider "stationary" players (movement < 5 meters/frame)
                        if dist < 5:
                            residuals.append(dist)

    if residuals:
        residual_mean = np.mean(residuals)
        residual_median = np.median(residuals)
        residual_max = np.max(residuals)
        residual_std = np.std(residuals)

        print(f"Residual drift after homography (m/frame) - stationary players:")
        print(f"  Mean:   {residual_mean:.4f}")
        print(f"  Median: {residual_median:.4f}")
        print(f"  Max:    {residual_max:.4f}")
        print(f"  Std:    {residual_std:.4f}")
        print(f"  Samples: {len(residuals)}")
    else:
        print("No stationary players found for residual evaluation")
        residual_mean = 0

    # 5. Additional: Camera movement per axis
    print("\n--- Camera Movement by Axis ---")
    x_movements = [m[0] for m in cam_movement[1:]]
    y_movements = [m[1] for m in cam_movement[1:]]

    print(f"X-axis (px/frame): Mean={np.mean(np.abs(x_movements)):.2f}, Max={np.max(np.abs(x_movements)):.2f}")
    print(f"Y-axis (px/frame): Mean={np.mean(np.abs(y_movements)):.2f}, Max={np.max(np.abs(y_movements)):.2f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Raw camera drift:     {raw_drift_mean:.2f} px/frame (mean)")
    print(f"  Residual drift:       {residual_mean:.4f} m/frame (mean)")
    if residual_mean < 0.1:
        print("  ✓ EXCELLENT homography correction (<0.1 m/frame)")
    elif residual_mean < 0.5:
        print("  ~ GOOD homography correction (0.1-0.5 m/frame)")
    else:
        print("  ✗ POOR homography correction (>0.5 m/frame)")
    print("=" * 60)


if __name__ == '__main__':
    main()