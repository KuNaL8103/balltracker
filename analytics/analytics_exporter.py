#!/usr/bin/env python
"""Export structured tactical analytics from the BallTracker pipeline.

Produces two artefacts at the end of a run:

* ``match_summary.json`` - match-level context, team possession, per-team and
  per-player heatmaps.
* ``player_metrics.csv`` - one row per track ID with distance, sprint speed,
  average speed and heatmap peak coordinates.

All spatial values are expressed in the pitch coordinate system created by
``view_transformer.ViewTransformer``: x runs along the 23.32 m pitch length and
y runs across the 68 m pitch width.

Usage
-----
    # Fast path: reuse the cached tracking + camera-movement stubs
    python analytics/analytics_exporter.py --max-frames 750

    # Full detection path (no stubs)
    python analytics/analytics_exporter.py --no-stub --video input_videos/08fd33_4.mp4
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep these in sync with view_transformer/view_transformer.py.
PITCH_LENGTH_M = 23.32
PITCH_WIDTH_M = 68.0

# 12 x 32 gives cells of roughly 1.94 m x 2.13 m.
HEATMAP_COLS = 12
HEATMAP_ROWS = 32
SPRINT_WINDOW_FRAMES = 5
MAX_PLAUSIBLE_SPEED_MPS = 12.0  # ~43 km/h; rejects homography/ID-switch spikes


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _fit_sequence(sequence: Sequence[Any], length: int, factory) -> List[Any]:
    """Trim or pad a per-frame sequence so it matches the analysed frame count."""
    fitted = list(sequence)[:length]
    if len(fitted) < length:
        fitted.extend(factory() for _ in range(length - len(fitted)))
    return fitted


def _fit_tracks(tracks: Dict[str, List[Dict[int, Dict[str, Any]]]], length: int) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
    for object_name, object_tracks in tracks.items():
        tracks[object_name] = _fit_sequence(object_tracks, length, dict)
    return tracks


def read_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    resize_width: Optional[int] = None,
) -> Tuple[List[np.ndarray], float, int, int]:
    """Read frames, optionally downscaling to keep CPU/Streamlit RAM bounded.

    A 1080p BGR frame is ~6.2 MB. At 300 frames that is ~1.86 GB before any
    model/tracker allocations, so live CPU runs pass ``resize_width=960`` and
    a small frame cap.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames: List[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_width and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (resize_width, max(2, int(round(frame.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from: {video_path}")

    height, width = frames[0].shape[:2]
    if width <= 0 or height <= 0:  # defensive fallback
        width, height = source_width, source_height

    return frames, fps, width, height


def get_video_source_size(video_path: str) -> Tuple[int, int]:
    """Return the original (width, height) before any RAM-saving resize."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return width, height


def scale_tracks_to_frame_size(
    tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> None:
    """Scale cached-stub boxes from original pixels to downscaled frames."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if (
        source_width <= 0
        or source_height <= 0
        or (source_width, source_height) == (target_width, target_height)
    ):
        return

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    for object_tracks in tracks.values():
        for frame_track in object_tracks:
            for track_info in frame_track.values():
                bbox = track_info.get("bbox")
                if bbox is not None and len(bbox) == 4:
                    track_info["bbox"] = [
                        float(bbox[0]) * scale_x,
                        float(bbox[1]) * scale_y,
                        float(bbox[2]) * scale_x,
                        float(bbox[3]) * scale_y,
                    ]


def scale_camera_movement_to_frame_size(
    camera_movement: Sequence[Sequence[float]],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> List[List[float]]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if (
        source_width <= 0
        or source_height <= 0
        or (source_width, source_height) == (target_width, target_height)
    ):
        return [list(movement) for movement in camera_movement]
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return [
        [float(movement[0]) * scale_x, float(movement[1]) * scale_y]
        for movement in camera_movement
    ]


def interpolate_ball_positions(ball_positions: Sequence[Dict[int, Dict[str, Any]]]) -> List[Dict[int, Dict[str, Any]]]:
    """Linear interpolation + edge fill for missing ball boxes.

    This mirrors ``Tracker.interpolate_ball_positions`` but is dependency-free
    and safe when the ball was never detected.
    """
    frame_count = len(ball_positions)
    boxes = np.full((frame_count, 4), np.nan, dtype=np.float64)
    for frame_idx, track in enumerate(ball_positions):
        bbox = track.get(1, {}).get("bbox")
        if bbox is not None and len(bbox) == 4:
            boxes[frame_idx] = bbox

    if np.isnan(boxes).all():
        return [{} for _ in range(frame_count)]

    indices = np.arange(frame_count)
    for column in range(4):
        values = boxes[:, column]
        valid = ~np.isnan(values)
        if not valid.any():
            continue
        values = np.interp(indices, indices[valid], values[valid])
        first, last = int(indices[valid][0]), int(indices[valid][-1])
        values[:first] = values[first]
        values[last + 1:] = values[last]
        boxes[:, column] = values

    return [{1: {"bbox": boxes[frame_idx].tolist()}} for frame_idx in range(frame_count)]


def add_positions_to_tracks(tracks: Dict[str, List[Dict[int, Dict[str, Any]]]]) -> None:
    from utils import get_center_of_bbox, get_foot_position

    for object_name, object_tracks in tracks.items():
        for frame_track in object_tracks:
            for track_info in frame_track.values():
                bbox = track_info["bbox"]
                track_info["position"] = (
                    get_center_of_bbox(bbox) if object_name == "ball" else get_foot_position(bbox)
                )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    model_path: str = "models/best.pt",
    max_frames: Optional[int] = None,
    stub_path: Optional[str] = None,
    camera_stub_path: Optional[str] = None,
    use_stubs: bool = True,
    tracker: Any = None,
    resize_width: Optional[int] = 960,
    log=print,
) -> Dict[str, Any]:
    """Run detection/tracking + homography + team assignment for one clip.

    ``tracker`` may be a pre-loaded ``Tracker`` instance. The FastAPI service
    uses this to keep the YOLO weights warm across requests while getting a
    fresh ByteTrack state per clip.
    """
    stub_available = bool(
        use_stubs and stub_path and Path(stub_path).is_file()
    )
    source_size = get_video_source_size(video_path) if stub_available else (0, 0)

    frames, fps, width, height = read_video_frames(
        video_path,
        max_frames=max_frames,
        resize_width=resize_width,
    )
    frame_count = len(frames)
    log(f"Loaded {frame_count} frames at {fps:.1f} FPS ({width}x{height})")

    tracker_obj = None
    if stub_available:
        log(f"Loading cached tracks from {stub_path}")
        with open(stub_path, "rb") as handle:
            tracks = pickle.load(handle)
    else:
        from trackers.tracker import Tracker

        tracker_obj = tracker or Tracker(model_path)
        if hasattr(tracker_obj, "reset_tracker"):
            tracker_obj.reset_tracker()
        log("Running YOLO + ByteTrack ...")
        tracks = tracker_obj.get_object_tracks(frames)

    tracks = _fit_tracks(tracks, frame_count)
    if stub_available:
        # Stubs were generated at the source resolution; frames may have been
        # downscaled for Streamlit Cloud's RAM limit.
        scale_tracks_to_frame_size(tracks, source_size, (width, height))
    add_positions_to_tracks(tracks)
    gc.collect()

    from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator

    camera_estimator = CameraMovementEstimator(frames[0])
    camera_stub_available = bool(
        stub_available and camera_stub_path and Path(camera_stub_path).is_file()
    )
    camera_movement = camera_estimator.get_camera_movement(
        frames,
        read_from_stub=camera_stub_available,
        # Only pass a write path when we are intentionally in stub mode.
        # Otherwise a --no-stub run would overwrite the cached stub with a
        # different clip and silently corrupt later runs.
        stub_path=str(camera_stub_path) if camera_stub_available else None,
    )
    if camera_stub_available:
        camera_movement = scale_camera_movement_to_frame_size(
            camera_movement, source_size, (width, height)
        )
    camera_movement = _fit_sequence(camera_movement, frame_count, lambda: [0, 0])
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
    gc.collect()

    from view_transformer.view_transformer import ViewTransformer

    log("Projecting tracks onto the 23.32 m x 68 m pitch ...")
    view_transformer = ViewTransformer()
    if width != 1920 or height != 1080:
        # Keep the calibration valid when frames were downscaled for RAM.
        scale = np.array([width / 1920.0, height / 1080.0], dtype=np.float32)
        view_transformer.pixel_vertices = (
            view_transformer.pixel_vertices * scale
        ).astype(np.float32)
        view_transformer.persepctive_trasnformer = cv2.getPerspectiveTransform(
            view_transformer.pixel_vertices,
            view_transformer.target_vertices,
        )
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = interpolate_ball_positions(tracks.get("ball", []))

    team_colors = _assign_teams(tracks, frames, log)
    team_ball_control = _assign_possession(tracks)

    # Frames are no longer needed once positions, teams and possession are
    # computed. Release them before returning the potentially large track dict.
    del frames
    gc.collect()

    return {
        "tracks": tracks,
        "team_ball_control": team_ball_control,
        "team_colors": team_colors,
        "fps": fps,
        "frames": frame_count,
        "width": width,
        "height": height,
        "video_path": str(video_path),
        "model_path": model_path,
        "used_stubs": stub_available,
    }


def _assign_teams(tracks: Dict[str, Any], frames: Sequence[np.ndarray], log=print):
    from team_assigner.team_assigner import TeamAssigner

    team_assigner = TeamAssigner()
    seed_index = next(
        (index for index, players in enumerate(tracks["players"]) if len(players) >= 2),
        None,
    )
    if seed_index is None:
        log("Team assignment skipped: fewer than two players detected")
        return {}

    try:
        team_assigner.assign_team_color(frames[seed_index], tracks["players"][seed_index])
    except Exception as exc:  # pragma: no cover - defensive around tiny crops
        log(f"Team clustering failed on seed frame: {exc}")
        return {}

    assigned = 0
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            try:
                team = team_assigner.get_player_team(
                    frames[frame_num], track["bbox"], player_id
                )
            except Exception:
                continue
            track["team"] = int(team)
            if team in team_assigner.team_colors:
                track["team_color"] = team_assigner.team_colors[team]
            assigned += 1

    log(f"Assigned teams for {assigned} player observations")
    return {
        int(team_id): [int(channel) for channel in color]
        for team_id, color in team_assigner.team_colors.items()
    }


def _assign_possession(tracks: Dict[str, Any]) -> np.ndarray:
    from player_ball_assigner.player_ball_assigner import PlayerBallAssigner

    assigner = PlayerBallAssigner()
    control: List[int] = []
    last_team = 0

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_track = tracks["ball"][frame_num].get(1) if frame_num < len(tracks["ball"]) else None
        if ball_track is not None:
            assigned_player = assigner.assign_ball_to_player(player_track, ball_track["bbox"])
            if assigned_player != -1:
                team = int(tracks["players"][frame_num][assigned_player].get("team", 0))
                if team in (1, 2):
                    last_team = team
        control.append(last_team)

    return np.asarray(control, dtype=int)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def _heatmap_cell(x: float, y: float) -> Tuple[int, int]:
    column = min(HEATMAP_COLS - 1, max(0, int(x / PITCH_LENGTH_M * HEATMAP_COLS)))
    row = min(HEATMAP_ROWS - 1, max(0, int(y / PITCH_WIDTH_M * HEATMAP_ROWS)))
    return column, row


def _cell_center(column: int, row: int) -> Tuple[float, float]:
    x = (column + 0.5) / HEATMAP_COLS * PITCH_LENGTH_M
    y = (row + 0.5) / HEATMAP_ROWS * PITCH_WIDTH_M
    return _round(x, 2), _round(y, 2)


def _player_stats(tracks: Dict[str, Any], fps: float) -> List[Dict[str, Any]]:
    observations: Dict[int, Dict[str, Any]] = defaultdict(
        lambda: {"team_counts": Counter(), "frames_seen": 0, "entries": []}
    )

    for frame_num, player_track in enumerate(tracks["players"]):
        for raw_id, track in player_track.items():
            player_id = int(raw_id)
            state = observations[player_id]
            state["frames_seen"] += 1

            team = int(track.get("team", 0) or 0)
            if team in (1, 2):
                state["team_counts"][team] += 1

            position = track.get("position_transformed")
            if position is None:
                continue
            try:
                x, y = float(position[0]), float(position[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                state["entries"].append((frame_num, x, y))

    max_step_m = MAX_PLAUSIBLE_SPEED_MPS / max(fps, 1e-6)
    players: List[Dict[str, Any]] = []

    for player_id, state in observations.items():
        entries = state["entries"]
        total_distance = 0.0
        segments: List[List[Tuple[int, float, float]]] = []
        current: List[Tuple[int, float, float]] = []
        previous: Optional[Tuple[int, float, float]] = None

        for entry in entries:
            frame_num, x, y = entry
            if previous is not None and frame_num == previous[0] + 1:
                step = math.hypot(x - previous[1], y - previous[2])
                if step <= max_step_m:
                    total_distance += step
                    current.append(entry)
                else:
                    if current:
                        segments.append(current)
                    current = [entry]
            else:
                if current:
                    segments.append(current)
                current = [entry]
            previous = entry
        if current:
            segments.append(current)

        top_speed_mps = 0.0
        for segment in segments:
            if len(segment) < SPRINT_WINDOW_FRAMES:
                continue
            for start in range(len(segment) - SPRINT_WINDOW_FRAMES + 1):
                first = segment[start]
                last = segment[start + SPRINT_WINDOW_FRAMES - 1]
                elapsed = (last[0] - first[0]) / max(fps, 1e-6)
                if elapsed <= 0:
                    continue
                speed = math.hypot(last[1] - first[1], last[2] - first[2]) / elapsed
                if speed <= MAX_PLAUSIBLE_SPEED_MPS:
                    top_speed_mps = max(top_speed_mps, speed)

        heatmap: Counter = Counter()
        for _, x, y in entries:
            heatmap[_heatmap_cell(x, y)] += 1

        if heatmap:
            peak_cell, _ = heatmap.most_common(1)[0]
            peak_x, peak_y = _cell_center(*peak_cell)
        else:
            peak_cell, peak_x, peak_y = None, None, None

        team_counts = state["team_counts"]
        team = team_counts.most_common(1)[0][0] if team_counts else 0
        tracked_seconds = state["frames_seen"] / max(fps, 1e-6)
        average_speed_kmh = (
            total_distance / tracked_seconds * 3.6 if tracked_seconds > 0 else 0.0
        )

        players.append(
            {
                "track_id": player_id,
                "team_id": team,
                "frames_seen": state["frames_seen"],
                "distance_m": _round(total_distance, 2),
                "top_speed_kmh": _round(top_speed_mps * 3.6, 2),
                "avg_speed_kmh": _round(average_speed_kmh, 2),
                "heatmap": {
                    "grid": [HEATMAP_COLS, HEATMAP_ROWS],
                    "peak_cell": list(peak_cell) if peak_cell is not None else None,
                    "peak_position_m": [peak_x, peak_y] if peak_x is not None else None,
                    "cells": [
                        [column, row, count]
                        for (column, row), count in sorted(heatmap.items())
                    ],
                },
            }
        )

    players.sort(key=lambda player: (-player["distance_m"], player["track_id"]))
    return players


def _team_payload(team_id: int, players: List[Dict[str, Any]], colors: Dict[int, Any],
                  possession_pct: float) -> Dict[str, Any]:
    members = [player for player in players if player["team_id"] == team_id]
    color = colors.get(team_id)
    if color is not None:
        color = [int(channel) for channel in color]
    aggregate: Counter = Counter()
    for player in members:
        for column, row, count in player["heatmap"]["cells"]:
            aggregate[(column, row)] += count

    peak_cell, peak_position = None, None
    if aggregate:
        peak_cell, _ = aggregate.most_common(1)[0]
        peak_position = list(_cell_center(*peak_cell))

    return {
        "team_id": team_id,
        "color_bgr": color,
        "possession_pct": possession_pct,
        "distance_m": _round(sum(player["distance_m"] for player in members), 2),
        "player_ids": [player["track_id"] for player in members],
        "heatmap": {
            "grid": [HEATMAP_COLS, HEATMAP_ROWS],
            "cell_size_m": [
                _round(PITCH_LENGTH_M / HEATMAP_COLS, 3),
                _round(PITCH_WIDTH_M / HEATMAP_ROWS, 3),
            ],
            "peak_cell": list(peak_cell) if peak_cell is not None else None,
            "peak_position_m": peak_position,
            "cells": [
                [column, row, count]
                for (column, row), count in sorted(aggregate.items())
            ],
        },
    }


def build_match_summary(result: Dict[str, Any], source_name: Optional[str] = None) -> Dict[str, Any]:
    """Turn a ``process_video`` result into the public JSON payload."""
    tracks = result["tracks"]
    fps = result["fps"]
    frame_count = result["frames"]
    players = _player_stats(tracks, fps)

    control = np.asarray(result.get("team_ball_control", []), dtype=int)
    team1_frames = int((control == 1).sum())
    team2_frames = int((control == 2).sum())
    controlled_frames = team1_frames + team2_frames
    if controlled_frames:
        team1_pct = _round(team1_frames / controlled_frames * 100.0, 2)
        team2_pct = _round(100.0 - team1_pct, 2)
    else:
        team1_pct = team2_pct = 0.0

    colors = result.get("team_colors", {}) or {}
    teams = [
        _team_payload(1, players, colors, team1_pct),
        _team_payload(2, players, colors, team2_pct),
    ]

    top_player = max(players, key=lambda player: player["top_speed_kmh"], default=None)
    total_distance = _round(sum(player["distance_m"] for player in players), 2)

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "source": source_name or Path(result["video_path"]).name,
            "frames_analyzed": frame_count,
            "fps": _round(fps, 3),
            "duration_s": _round(frame_count / max(fps, 1e-6), 2),
            "resolution": [result["width"], result["height"]],
            "used_cached_tracks": bool(result.get("used_stubs", False)),
        },
        "pipeline": {
            "detector": "YOLOv5",
            "tracker": "ByteTrack",
            "team_clustering": "K-Means (2 teams)",
            "camera_motion": "Lucas-Kanade optical flow",
            "pitch_projection": "perspective transform (homography)",
            "model_path": result.get("model_path"),
        },
        "pitch": {
            "length_m": PITCH_LENGTH_M,
            "width_m": PITCH_WIDTH_M,
            "coordinate_system": (
                "x along pitch length (23.32 m), y across pitch width (68 m); "
                "origin at near-left pitch corner"
            ),
            "heatmap_grid": [HEATMAP_COLS, HEATMAP_ROWS],
            "heatmap_cell_m": [
                _round(PITCH_LENGTH_M / HEATMAP_COLS, 3),
                _round(PITCH_WIDTH_M / HEATMAP_ROWS, 3),
            ],
        },
        "summary": {
            "players_tracked": len(players),
            "total_distance_m": total_distance,
            "team_possession_pct": {"team_1": team1_pct, "team_2": team2_pct},
            "team_distance_m": {"team_1": teams[0]["distance_m"], "team_2": teams[1]["distance_m"]},
            "top_sprint_speed_kmh": (
                {
                    "track_id": top_player["track_id"],
                    "team_id": top_player["team_id"],
                    "value": top_player["top_speed_kmh"],
                }
                if top_player and top_player["top_speed_kmh"] > 0
                else None
            ),
        },
        "teams": teams,
        "players": players,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def export_analytics(
    result: Dict[str, Any],
    output_dir: str = "outputs/analytics",
    source_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path, Path]:
    payload = build_match_summary(result, source_name=source_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "match_summary.json"
    csv_path = output_path / "player_metrics.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    possession = payload["summary"]["team_possession_pct"]
    fieldnames = [
        "player_id",
        "team_id",
        "frames_seen",
        "distance_m",
        "top_speed_kmh",
        "avg_speed_kmh",
        "team_possession_pct",
        "heatmap_peak_x_m",
        "heatmap_peak_y_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for player in payload["players"]:
            team_key = f"team_{player['team_id']}"
            peak = player["heatmap"]["peak_position_m"] or [None, None]
            writer.writerow(
                {
                    "player_id": player["track_id"],
                    "team_id": player["team_id"],
                    "frames_seen": player["frames_seen"],
                    "distance_m": player["distance_m"],
                    "top_speed_kmh": player["top_speed_kmh"],
                    "avg_speed_kmh": player["avg_speed_kmh"],
                    "team_possession_pct": possession.get(team_key, 0.0),
                    "heatmap_peak_x_m": peak[0],
                    "heatmap_peak_y_m": peak[1],
                }
            )

    return payload, json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tactical analytics for a football clip")
    parser.add_argument("--video", default="input_videos/08fd33_4.mp4")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--stub", default="stubs/track_stubs.pkl")
    parser.add_argument("--camera-stub", default="stubs/camera_movement_stub.pkl")
    parser.add_argument("--no-stub", action="store_true", help="Run detection instead of cached tracks")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--resize-width",
        type=int,
        default=960,
        help="Downscale frames to this width to bound RAM (0 disables resizing)",
    )
    parser.add_argument("--output-dir", default="outputs/analytics")
    args = parser.parse_args()

    use_stubs = not args.no_stub
    result = process_video(
        video_path=args.video,
        model_path=args.model,
        max_frames=args.max_frames,
        stub_path=args.stub,
        camera_stub_path=args.camera_stub,
        use_stubs=use_stubs,
        resize_width=args.resize_width if args.resize_width > 0 else None,
    )
    payload, json_path, csv_path = export_analytics(result, output_dir=args.output_dir)

    summary = payload["summary"]
    print("\n" + "=" * 60)
    print("TACTICAL ANALYTICS EXPORT")
    print("=" * 60)
    print(f"Players tracked     : {summary['players_tracked']}")
    print(f"Frames analyzed     : {payload['video']['frames_analyzed']}")
    print(f"Total distance      : {summary['total_distance_m']:.1f} m")
    print(
        "Team possession     : "
        f"{summary['team_possession_pct']['team_1']:.1f}% / "
        f"{summary['team_possession_pct']['team_2']:.1f}%"
    )
    if summary["top_sprint_speed_kmh"]:
        sprint = summary["top_sprint_speed_kmh"]
        print(
            f"Top sprint          : {sprint['value']:.2f} km/h "
            f"(player {sprint['track_id']}, team {sprint['team_id']})"
        )
    print(f"JSON                : {json_path.resolve()}")
    print(f"CSV                 : {csv_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
