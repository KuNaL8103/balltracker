#!/usr/bin/env python
"""
GPU-Optimized Main Entry Point for BallTracker

This is a GPU-accelerated version of main.py with:
- CUDA model initialization
- FP16 inference
- Larger batch size
- cuDNN benchmark & TF32 enabled
- Performance timing

Usage:
    python pipeline/main_gpu.py

Output: output_videos/output_video_gpu.avi
"""

import sys
import time
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enable cuDNN benchmark and TF32 for optimal GPU performance
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Import pipeline modules
from utils.video_utils import read_video, save_video, get_video_fps
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from analytics.analytics_exporter import export_analytics

import numpy as np


def print_gpu_status():
    """Print GPU configuration status."""
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"✓ CUDA Available: Yes")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")

        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {total_mem:.1f} GB")

        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    else:
        print("✗ CUDA Available: No (running on CPU)")
        print("  Install CUDA-enabled PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    print("=" * 60)


def main():
    print_gpu_status()

    # Paths
    video_path = 'input_videos/08fd33_4.mp4'
    output_path = 'output_videos/output_video_gpu.avi'
    model_path = 'models/best.pt'
    track_stub = 'stubs/track_stubs.pkl'
    cam_stub = 'stubs/camera_movement_stub.pkl'

    print(f"\nInput video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")

    source_fps = get_video_fps(video_path)

    # Load video
    print("\n[1/10] Loading video...")
    start = time.time()
    video_frames = read_video(video_path)
    load_time = time.time() - start
    print(f"  Loaded {len(video_frames)} frames in {load_time:.2f}s")
    print(f"  Resolution: {video_frames[0].shape[1]}x{video_frames[0].shape[0]}")

    # Initialize Tracker (GPU-optimized in __init__)
    print("\n[2/10] Initializing tracker (GPU)...")
    start = time.time()
    tracker = Tracker(model_path)
    init_time = time.time() - start
    print(f"  Initialized in {init_time:.2f}s")

    # Get object tracks
    print("\n[3/10] Running object tracking...")
    start = time.time()
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=track_stub
    )
    track_time = time.time() - start
    print(f"  Tracking completed in {track_time:.2f}s ({len(video_frames)/track_time:.1f} FPS)")

    # Add positions
    print("\n[4/10] Adding positions...")
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimation
    print("\n[5/10] Estimating camera movement...")
    start = time.time()
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_movement = cam_est.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=cam_stub
    )
    cam_est.add_adjust_positions_to_tracks(tracks, cam_movement)
    cam_time = time.time() - start
    print(f"  Camera movement completed in {cam_time:.2f}s")

    # View transformation
    print("\n[6/10] Applying view transformation...")
    start = time.time()
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    view_time = time.time() - start
    print(f"  View transform completed in {view_time:.2f}s")

    # Ball interpolation
    print("\n[7/10] Interpolating ball positions...")
    start = time.time()
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    interp_time = time.time() - start
    print(f"  Interpolation completed in {interp_time:.4f}s")

    # Speed and distance estimation
    print("\n[8/10] Estimating speed and distance...")
    start = time.time()
    speed_est = SpeedAndDistance_Estimator()
    speed_est.frame_rate = source_fps
    speed_est.add_speed_and_distance_to_tracks(tracks)
    speed_time = time.time() - start
    print(f"  Speed/distance completed in {speed_time:.2f}s")

    # Team assignment
    print("\n[9/10] Assigning teams...")
    start = time.time()
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
    team_time = time.time() - start
    print(f"  Team assignment completed in {team_time:.2f}s")

    # Ball possession assignment
    print("\n[10/10] Assigning ball possession...")
    start = time.time()
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_track = tracks['ball'][frame_num].get(1)
        assigned = -1
        if ball_track is not None:
            assigned = player_assigner.assign_ball_to_player(player_track, ball_track['bbox'])

        if assigned != -1:
            tracks['players'][frame_num][assigned]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)
    possess_time = time.time() - start
    print(f"  Ball possession completed in {possess_time:.2f}s")

    # Export structured tactical analytics (JSON + CSV)
    print("\nExporting tactical analytics...")
    start = time.time()
    _, analytics_json, analytics_csv = export_analytics(
        {
            'tracks': tracks,
            'team_ball_control': team_ball_control,
            'team_colors': team_assigner.team_colors,
            'fps': source_fps,
            'frames': len(video_frames),
            'width': video_frames[0].shape[1],
            'height': video_frames[0].shape[0],
            'video_path': video_path,
            'model_path': model_path,
            'used_stubs': True,
        },
        output_dir='outputs/analytics',
    )
    analytics_time = time.time() - start
    print(f"  Analytics exported in {analytics_time:.2f}s")
    print(f"    JSON: {analytics_json}")
    print(f"    CSV : {analytics_csv}")

    # Draw output
    print("\nDrawing output video...")
    start = time.time()
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = cam_est.draw_camera_movement(output_frames, cam_movement)
    output_frames = speed_est.draw_speed_and_distance(output_frames, tracks)
    draw_time = time.time() - start
    print(f"  Drawing completed in {draw_time:.2f}s")

    # Save video
    print("\nSaving output video...")
    start = time.time()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_video(output_frames, output_path, fps=source_fps)
    save_time = time.time() - start
    print(f"  Save completed in {save_time:.2f}s")

    # Summary
    total_time = load_time + init_time + track_time + cam_time + view_time + interp_time + speed_time + team_time + possess_time + analytics_time + draw_time + save_time

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY (GPU)")
    print("=" * 60)
    print(f"  Video loading:         {load_time:>6.2f}s")
    print(f"  Tracker init:          {init_time:>6.2f}s")
    print(f"  Object tracking:       {track_time:>6.2f}s  ({len(video_frames)/track_time:.1f} FPS)")
    print(f"  Camera movement:       {cam_time:>6.2f}s")
    print(f"  View transform:        {view_time:>6.2f}s")
    print(f"  Ball interpolation:    {interp_time:>6.4f}s")
    print(f"  Speed/distance:        {speed_time:>6.2f}s")
    print(f"  Team assignment:       {team_time:>6.2f}s")
    print(f"  Ball possession:       {possess_time:>6.2f}s")
    print(f"  Analytics export:      {analytics_time:>6.2f}s")
    print(f"  Drawing:               {draw_time:>6.2f}s")
    print(f"  Video saving:          {save_time:>6.2f}s")
    print(f"  {'─' * 20}")
    print(f"  TOTAL:                 {total_time:>6.2f}s  ({len(video_frames)/total_time:.1f} FPS)")
    print("=" * 60)
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()