#!/usr/bin/env python
"""
Output Frame-Drop Rate Evaluation Script for BallTracker

This script runs the full pipeline and verifies that the output video
has the same number of frames as the input (no frame drops).

Usage:
    python eval_output.py --video input_videos/08fd33_4.mp4 --model models/best.pt

Expected output: Input frames, Output frames, Frame drop rate percentage
"""

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))

from trackers.tracker import Tracker
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from utils.video_utils import read_video, save_video


def run_full_pipeline(video_frames, tracker, stub_path='stubs/track_stubs.pkl', cam_stub_path='stubs/camera_movement_stub.pkl'):
    """Run the complete BallTracker pipeline and return output frames."""
    print("Running full pipeline...")

    # 1. Get object tracks
    print("  1/8: Getting object tracks...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=stub_path
    )

    # 2. Add positions
    print("  2/8: Adding positions...")
    tracker.add_position_to_tracks(tracks)

    # 3. Camera movement estimation
    print("  3/8: Estimating camera movement...")
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_movement = cam_est.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=cam_stub_path
    )
    cam_est.add_adjust_positions_to_tracks(tracks, cam_movement)

    # 4. View transformation
    print("  4/8: Applying view transformation...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 5. Ball interpolation
    print("  5/8: Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 6. Speed and distance estimation
    print("  6/8: Estimating speed and distance...")
    speed_est = SpeedAndDistance_Estimator()
    speed_est.add_speed_and_distance_to_tracks(tracks)

    # 7. Team assignment
    print("  7/8: Assigning teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # 8. Ball possession assignment
    print("  8/8: Assigning ball possession...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned != -1:
            tracks['players'][frame_num][assigned]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

    team_ball_control = np.array(team_ball_control)

    # Draw annotations
    print("  Drawing annotations...")
    start = time.time()
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = cam_est.draw_camera_movement(output_frames, cam_movement)
    output_frames = speed_est.draw_speed_and_distance(output_frames, tracks)
    draw_time = time.time() - start

    return output_frames, draw_time, team_ball_control


def verify_output_video(output_path, expected_frames):
    """Verify output video frame count matches expected."""
    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        print(f"Error: Could not open output video {output_path}")
        return None

    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        'frames': actual_frames,
        'fps': fps,
        'width': width,
        'height': height
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate output frame-drop rate')
    parser.add_argument('--video', default='input_videos/08fd33_4.mp4', help='Input video path')
    parser.add_argument('--model', default='models/best.pt', help='YOLO model path')
    parser.add_argument('--output', default='output_videos/benchmark_output.avi', help='Output video path')
    parser.add_argument('--stub', default='stubs/track_stubs.pkl', help='Tracker stub path')
    parser.add_argument('--cam-stub', default='stubs/camera_movement_stub.pkl', help='Camera movement stub path')

    args = parser.parse_args()

    print(f"Evaluating output frame-drop rate")
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print("-" * 60)

    # Load video
    video_frames = read_video(args.video)
    input_fps = 24  # From save_video function
    expected_frames = len(video_frames)

    print(f"Input video: {expected_frames} frames @ {input_fps} FPS")
    print(f"Resolution: {video_frames[0].shape[1]}x{video_frames[0].shape[0]}")

    # Initialize tracker
    tracker = Tracker(args.model)

    # Run full pipeline
    start_total = time.time()
    output_frames, draw_time, team_ball_control = run_full_pipeline(
        video_frames, tracker, args.stub, args.cam_stub
    )
    total_time = time.time() - start_total

    print(f"\nPipeline completed in {total_time:.2f}s")
    print(f"  Drawing time: {draw_time:.2f}s")
    print(f"  Output frames generated: {len(output_frames)}")

    # Save output video
    print(f"\nSaving output video...")
    save_start = time.time()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_video(output_frames, args.output)
    save_time = time.time() - save_start
    print(f"  Save time: {save_time:.2f}s")

    # Verify output
    print(f"\nVerifying output video...")
    output_info = verify_output_video(args.output, expected_frames)

    if output_info:
        actual_frames = output_info['frames']
        drop_rate = (expected_frames - actual_frames) / expected_frames * 100

        print(f"\n" + "=" * 60)
        print("SUMMARY")
        print(f"  Input frames:     {expected_frames}")
        print(f"  Output frames:    {actual_frames}")
        print(f"  Frame drop rate:  {drop_rate:.2f}%")
        print(f"  Output FPS:       {output_info['fps']:.1f}")
        print(f"  Output resolution: {output_info['width']}x{output_info['height']}")
        print(f"  Total pipeline time: {total_time:.2f}s")
        print(f"  Effective FPS:    {expected_frames/total_time:.1f}")
        print("=" * 60)

        if drop_rate == 0:
            print("✓ PERFECT - No frame drops!")
        elif drop_rate < 0.1:
            print("✓ EXCELLENT - Negligible frame drops")
        elif drop_rate < 1:
            print("~ ACCEPTABLE - Minor frame drops")
        else:
            print("✗ POOR - Significant frame drops")
    else:
        print("Failed to verify output video")


if __name__ == '__main__':
    main()