#!/usr/bin/env python
"""
MOTA / ID Switches Evaluation Script for BallTracker

This script evaluates tracking quality by computing ID switches from tracker output.
Uses the existing stub files or runs tracking on the input video.

Usage:
    python eval_tracking.py --video input_videos/08fd33_4.mp4 --model models/best.pt --frames 1200

Expected output: Estimated ID switches, total unique tracks
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from trackers.tracker import Tracker
from utils.video_utils import read_video


def compute_id_switches(tracks, object_type='players'):
    """
    Compute estimated ID switches by analyzing track appearance/disappearance.

    Heuristic: When tracks disappear and new tracks appear in the same frame,
    count the minimum of lost and new tracks as potential ID switches.
    """
    if object_type not in tracks:
        print(f"Object type '{object_type}' not found in tracks")
        return 0, 0

    object_tracks = tracks[object_type]
    track_ids_per_frame = [set(t.keys()) for t in object_tracks]

    id_switches = 0
    for i in range(1, len(track_ids_per_frame)):
        lost = track_ids_per_frame[i-1] - track_ids_per_frame[i]
        new = track_ids_per_frame[i] - track_ids_per_frame[i-1]
        id_switches += min(len(lost), len(new))

    all_track_ids = set().union(*track_ids_per_frame)
    return id_switches, len(all_track_ids)


def compute_mota_approximation(tracks, object_type='players'):
    """
    Approximate MOTA (Multiple Object Tracking Accuracy) components.

    MOTA = 1 - (FP + FN + IDSW) / GT
    Where FP = False Positives, FN = False Negatives, IDSW = ID Switches

    Since we don't have ground truth, we estimate:
    - FP: detections without consistent tracks
    - FN: frames where expected tracks are missing
    - IDSW: from compute_id_switches
    """
    if object_type not in tracks:
        return None

    object_tracks = tracks[object_type]
    num_frames = len(object_tracks)

    # Count frames with detections vs expected
    total_detections = sum(len(t) for t in object_tracks)
    avg_detections_per_frame = total_detections / num_frames if num_frames > 0 else 0

    # Estimate track fragmentation
    track_lengths = {}
    for frame_idx, frame_tracks in enumerate(object_tracks):
        for track_id in frame_tracks:
            if track_id not in track_lengths:
                track_lengths[track_id] = []
            track_lengths[track_id].append(frame_idx)

    # Tracks that appear in non-consecutive frames
    fragmented_tracks = 0
    for track_id, frames in track_lengths.items():
        if len(frames) > 1:
            gaps = sum(1 for i in range(1, len(frames)) if frames[i] - frames[i-1] > 1)
            if gaps > 0:
                fragmented_tracks += 1

    return {
        'total_detections': total_detections,
        'avg_detections_per_frame': avg_detections_per_frame,
        'num_unique_tracks': len(track_lengths),
        'fragmented_tracks': fragmented_tracks,
        'total_frames': num_frames
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate tracking quality (MOTA/ID Switches)')
    parser.add_argument('--video', default='input_videos/08fd33_4.mp4', help='Input video path')
    parser.add_argument('--model', default='models/best.pt', help='YOLO model path')
    parser.add_argument('--frames', type=int, default=1200, help='Number of frames to evaluate')
    parser.add_argument('--stub', default='stubs/track_stubs.pkl', help='Tracker stub path')
    parser.add_argument('--read-stub', action='store_true', default=True, help='Read from stub')

    args = parser.parse_args()

    print(f"Evaluating tracking quality")
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Frames: {args.frames}")
    print("-" * 60)

    # Load or compute tracks
    if args.read_stub and Path(args.stub).exists():
        print(f"Loading tracks from stub: {args.stub}")
        with open(args.stub, 'rb') as f:
            tracks = pickle.load(f)
    else:
        print("Running tracker on video...")
        video_frames = read_video(args.video)
        if len(video_frames) < args.frames:
            print(f"Warning: Video has only {len(video_frames)} frames, using all")
            args.frames = len(video_frames)

        tracker = Tracker(args.model)
        tracks = tracker.get_object_tracks(video_frames[:args.frames])

    # Evaluate players
    print("\n--- Players Tracking ---")
    id_switches, num_tracks = compute_id_switches(tracks, 'players')
    print(f"Estimated ID switches (first {args.frames} frames): {id_switches}")
    print(f"Total unique player tracks: {num_tracks}")

    mota_info = compute_mota_approximation(tracks, 'players')
    if mota_info:
        print(f"Total detections: {mota_info['total_detections']}")
        print(f"Avg detections/frame: {mota_info['avg_detections_per_frame']:.2f}")
        print(f"Fragmented tracks: {mota_info['fragmented_tracks']}")

    # Evaluate ball
    print("\n--- Ball Tracking ---")
    id_switches_ball, num_tracks_ball = compute_id_switches(tracks, 'ball')
    print(f"Ball track consistency: {num_tracks_ball} track(s) over {args.frames} frames")

    # Evaluate referees
    print("\n--- Referees Tracking ---")
    id_switches_ref, num_tracks_ref = compute_id_switches(tracks, 'referees')
    print(f"Estimated ID switches: {id_switches_ref}")
    print(f"Total unique referee tracks: {num_tracks_ref}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Player ID switches: {id_switches} ({num_tracks} unique tracks)")
    print(f"  Referee ID switches: {id_switches_ref} ({num_tracks_ref} unique tracks)")
    print(f"  Ball tracks: {num_tracks_ball}")
    print("=" * 60)


if __name__ == '__main__':
    main()