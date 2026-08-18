#!/usr/bin/env python
"""
Team Assignment Accuracy Evaluation Script for BallTracker

This script evaluates the quality of team assignment by:
1. Sampling players and checking cluster separation (silhouette score)
2. Visualizing team color clusters

Usage:
    python eval_team_assignment.py --video input_videos/08fd33_4.mp4 --model models/best.pt --frames 100

Expected output: Silhouette score (>0.5 = good separation), team colors
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append(str(Path(__file__).parent))

from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from utils.video_utils import read_video


def evaluate_team_separation(video_frames, tracks, team_assigner, sample_players=10):
    """
    Evaluate team color separation using silhouette score.

    Silhouette score ranges from -1 to 1:
    - > 0.5: Good separation
    - 0.25 - 0.5: Reasonable separation
    - < 0.25: Poor separation
    """
    if 'players' not in tracks or len(tracks['players']) == 0:
        print("No player tracks found")
        return None

    frame_0_players = tracks['players'][0]
    player_colors = []

    print(f"Extracting colors from {min(sample_players, len(frame_0_players))} players in frame 0...")

    for i, (player_id, track) in enumerate(list(frame_0_players.items())[:sample_players]):
        bbox = track['bbox']
        img = video_frames[0][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        if img.size == 0:
            continue

        # Use top half of player bbox (jersey area)
        top_half = img[0:int(img.shape[0]/2), :]

        if top_half.size == 0:
            continue

        top_2d = top_half.reshape(-1, 3)

        # Cluster pixels in player region
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(top_2d)
        labels = kmeans.labels_

        clustered = labels.reshape(top_half.shape[0], top_half.shape[1])

        # Identify player cluster (non-background)
        corners = [clustered[0,0], clustered[0,-1], clustered[-1,0], clustered[-1,-1]]
        non_player = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player

        player_color = kmeans.cluster_centers_[player_cluster]
        player_colors.append(player_color)

        # Get assigned team
        team = team_assigner.get_player_team(video_frames[0], bbox, player_id)
        print(f"  Player {player_id}: Team {team}, Color: {team_assigner.team_colors.get(team, 'N/A')}")

    if len(player_colors) < 2:
        print("Not enough players for clustering evaluation")
        return None

    player_colors = np.array(player_colors)

    # Final clustering to evaluate separation
    kmeans_final = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
    kmeans_final.fit(player_colors)

    sil = silhouette_score(player_colors, kmeans_final.labels_)

    print(f"\nPlayer colors collected: {len(player_colors)}")
    print(f"Cluster centers: {kmeans_final.cluster_centers_}")
    print(f"Silhouette score: {sil:.4f}")

    # Also evaluate using the team_assigner's internal KMeans
    if hasattr(team_assigner, 'kmeans') and team_assigner.kmeans is not None:
        sil_assigner = silhouette_score(player_colors, team_assigner.kmeans.predict(player_colors))
        print(f"Silhouette (using TeamAssigner KMeans): {sil_assigner:.4f}")

    return sil


def visualize_team_colors(video_frame, tracks, team_assigner, output_path='output_videos/team_colors.jpg'):
    """Visualize team color assignments on frame."""
    frame = video_frame.copy()

    for player_id, track in tracks['players'][0].items():
        bbox = track['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        team = track.get('team', 0)
        color = team_assigner.team_colors.get(team, (255, 255, 255))
        color_int = tuple(map(int, color))

        # Draw bounding box with team color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_int, 2)

        # Draw team label
        cv2.putText(frame, f"Team {team}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_int, 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate team assignment accuracy')
    parser.add_argument('--video', default='input_videos/08fd33_4.mp4', help='Input video path')
    parser.add_argument('--model', default='models/best.pt', help='YOLO model path')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to process')
    parser.add_argument('--sample', type=int, default=10, help='Number of players to sample')
    parser.add_argument('--stub', default='stubs/track_stubs.pkl', help='Tracker stub path')

    args = parser.parse_args()

    print(f"Evaluating team assignment accuracy")
    print(f"Video: {args.video}")
    print(f"Frames to process: {args.frames}")
    print("-" * 60)

    # Load video and tracks
    video_frames = read_video(args.video)
    if len(video_frames) < args.frames:
        args.frames = len(video_frames)

    tracker = Tracker(args.model)
    tracks = tracker.get_object_tracks(video_frames[:args.frames])

    # Initialize and run team assigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    print(f"\nTeam colors assigned:")
    for team_id, color in team_assigner.team_colors.items():
        print(f"  Team {team_id}: RGB{tuple(map(int, color))}")

    # Assign teams to all players
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Evaluate separation
    print("\n--- Team Separation Evaluation ---")
    sil = evaluate_team_separation(video_frames, tracks, team_assigner, args.sample)

    # Visualize
    print("\n--- Visualization ---")
    visualize_team_colors(video_frames[0], tracks, team_assigner)

    print("\n" + "=" * 60)
    print("SUMMARY")
    if sil is not None:
        print(f"  Silhouette score: {sil:.4f}")
        if sil > 0.5:
            print("  ✓ EXCELLENT separation (>0.5)")
        elif sil > 0.25:
            print("  ~ GOOD separation (0.25-0.5)")
        else:
            print("  ✗ POOR separation (<0.25)")
    print("=" * 60)


if __name__ == '__main__':
    main()