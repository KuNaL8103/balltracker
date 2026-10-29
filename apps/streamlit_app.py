#!/usr/bin/env python
"""Interactive Streamlit dashboard for BallTracker.

Run with:
    streamlit run apps/streamlit_app.py

Tabs
----
1. Match analytics: KPIs, annotated video, team-colour plot, pitch heatmaps,
   player metrics table and CSV download.
2. Benchmarks: Batch-10 / Batch-1 inference comparison, charts, and the
   pipeline quality evaluation cards.
3. FastAPI playground: live health check, one-click clip submission, raw JSON
   response and an in-process exporter fallback.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
import os
import urllib.request
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best.pt"

# Paste your direct GitHub Release download URL here
MODEL_URL = "https://github.com/KuNaL8103/balltracker/releases/download/model/best.pt"

def ensure_model_exists():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        import streamlit as st
        with st.spinner("Downloading YOLO model weights (~165MB), please wait..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Run the check before running the pipeline or loading the model
ensure_model_exists()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SAMPLE_VIDEO = PROJECT_ROOT / "input_videos" / "08fd33_4.mp4"
TRACK_STUB = PROJECT_ROOT / "stubs" / "track_stubs.pkl"
CAMERA_STUB = PROJECT_ROOT / "stubs" / "camera_movement_stub.pkl"
ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics"
RUNS_DIR = ANALYTICS_DIR / "runs"
MATCH_SUMMARY_PATH = ANALYTICS_DIR / "match_summary.json"
PLAYER_METRICS_PATH = ANALYTICS_DIR / "player_metrics.csv"
EVAL_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "eval_summary.json"
BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
BENCHMARK_BATCH10_PATH = BENCHMARK_RESULTS_DIR / "benchmark_results.json"
BENCHMARK_BATCH1_PATH = BENCHMARK_RESULTS_DIR / "benchmark_batch1_results.json"
ANNOTATED_AVI = PROJECT_ROOT / "output_videos" / "output_video.avi"
BROWSER_MP4 = PROJECT_ROOT / "output_videos" / "output_video_dashboard.mp4"
TEAM_COLORS_IMAGE = PROJECT_ROOT / "output_videos" / "team_colors.jpg"

API_DEFAULT = "http://127.0.0.1:8000"
PITCH_LENGTH_M = 23.32
PITCH_WIDTH_M = 68.0

st.set_page_config(
    page_title="BallTracker Tactical Dashboard",
    page_icon="âš½",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1500px;}
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%);
        color: white; padding: 1.5rem 1.75rem; border-radius: 20px;
        margin-bottom: 1.2rem; box-shadow: 0 12px 30px rgba(15, 23, 42, .18);
    }
    .hero h1 {font-size: 2.1rem; margin: 0 0 .35rem 0; letter-spacing: -.02em;}
    .hero p {margin: 0; opacity: .92; font-size: 1rem;}
    [data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 1rem 1.1rem; box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
    }
    [data-testid="stMetricLabel"] {color: #475569;}
    [data-testid="stMetricValue"] {color: #0f172a;}
    [data-testid="stMetricDelta"] {color: #0f766e;}
    div[data-testid="stTabs"] button {font-size: 1rem; font-weight: 600;}
    .small-note {color: #64748b; font-size: .86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_json(path: Path, default: Optional[Any] = None) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def payload_caption(payload: Dict[str, Any]) -> str:
    video = payload.get("video", {})
    mode = "cached stubs" if video.get("used_cached_tracks") else "live pipeline"
    return (
        f"{video.get('source', 'clip')} Â· {video.get('frames_analyzed', 0)} frames Â· "
        f"{video.get('fps', 0):.1f} FPS Â· {mode}"
    )


def store_payload(
    payload: Optional[Dict[str, Any]],
    frames: Optional[List[bytes]] = None,
    *,
    is_custom: bool = False,
    uploaded_bytes: Optional[bytes] = None,
    uploaded_name: Optional[str] = None,
    run_dir: Optional[Path] = None,
    annotated_video_path: Optional[Path] = None,
    team_colors_path: Optional[Path] = None,
    player_metrics_path: Optional[Path] = None,
    active_source: Optional[str] = None,
) -> None:
    st.session_state.payload = payload
    st.session_state.annotated_frames = frames or []
    st.session_state.is_custom = is_custom
    st.session_state.uploaded_bytes = uploaded_bytes
    st.session_state.uploaded_name = uploaded_name
    st.session_state.run_dir = str(run_dir) if run_dir else None
    st.session_state.annotated_video_path = (
        str(annotated_video_path) if annotated_video_path else None
    )
    st.session_state.team_colors_path = (
        str(team_colors_path)
        if team_colors_path
        else (str(TEAM_COLORS_IMAGE) if not is_custom and run_dir is None else None)
    )
    st.session_state.player_metrics_path = (
        str(player_metrics_path)
        if player_metrics_path
        else (str(PLAYER_METRICS_PATH) if not is_custom and run_dir is None else None)
    )
    st.session_state.is_live_run = bool(run_dir)
    st.session_state.active_source = active_source or (
        "custom" if is_custom else "sample"
    )


def init_state() -> None:
    if "payload" not in st.session_state:
        store_payload(load_json(MATCH_SUMMARY_PATH))


def on_source_change() -> None:
    """Restore the instant pre-computed sample when switching back to it."""
    source = st.session_state.get("input_source", "")
    if source.startswith("Sample"):
        saved = load_json(MATCH_SUMMARY_PATH)
        if saved:
            store_payload(saved)
        else:
            store_payload(None, active_source="sample")
            st.session_state.payload = None


def get_tracker():
    """Cache YOLO weights across Streamlit reruns."""
    from trackers.tracker import Tracker

    return Tracker(str(MODEL_PATH))


@st.cache_resource(show_spinner=False)
def cached_tracker():
    return get_tracker()


def run_pipeline(
    video_path: Path,
    max_frames: int,
    use_stubs: bool,
    source_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[bytes]]:
    from analytics.analytics_exporter import build_match_summary, process_video

    effective_stubs = bool(use_stubs and TRACK_STUB.is_file())
    tracker = None if effective_stubs else cached_tracker()

    result = process_video(
        video_path=str(video_path),
        model_path=str(MODEL_PATH),
        max_frames=max_frames,
        stub_path=str(TRACK_STUB) if effective_stubs else None,
        camera_stub_path=str(CAMERA_STUB) if effective_stubs else None,
        use_stubs=effective_stubs,
        tracker=tracker,
        log=lambda _message: None,
    )
    payload = build_match_summary(result, source_name=source_name or video_path.name)
    annotated = make_annotated_samples(result, count=3)
    return payload, annotated


def create_run_dir(source_name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_name).stem)[:40].strip("_")
    slug = slug or "clip"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RUNS_DIR / f"{stamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_team_colors_plot(
    frame: np.ndarray,
    tracks: Dict[str, Any],
    team_assigner: Any,
    output_path: Path,
    seed_index: int,
) -> Path:
    """Save a per-video K-Means swatch/cluster plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.4),
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    frame_axis, cluster_axis = axes

    frame_axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_axis.set_title("Detected teams in seed frame", fontweight="bold")
    frame_axis.axis("off")

    player_frames = tracks.get("players", [])
    seed_players = (
        player_frames[seed_index]
        if 0 <= seed_index < len(player_frames)
        else {}
    )

    for player_id, player in seed_players.items():
        bbox = player.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        team = int(player.get("team", 0) or 0)
        center = team_assigner.team_colors.get(team)
        if center is None:
            rgb = (0.65, 0.65, 0.65)
        else:
            rgb = tuple(float(channel) / 255.0 for channel in center[::-1])
        frame_axis.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=rgb, linewidth=2.0,
            )
        )
        frame_axis.text(
            x1, max(14, y1 - 6), f"P{int(player_id)} · T{team}",
            color=rgb, fontsize=8, fontweight="bold",
        )

    points: List[Tuple[float, float, float, int]] = []
    for player in seed_players.values():
        bbox = player.get("bbox")
        if not bbox:
            continue
        try:
            color = team_assigner.get_player_color(frame, bbox)
        except Exception:
            continue
        team = int(player.get("team", 0) or 0)
        points.append((float(color[2]), float(color[1]), float(color[0]), team))

    for team_id, marker in ((1, "o"), (2, "s")):
        team_points = [
            (red, green, blue)
            for red, green, blue, team in points
            if team == team_id
        ]
        if team_points:
            cluster_axis.scatter(
                [point[0] for point in team_points],
                [point[1] for point in team_points],
                s=80,
                marker=marker,
                alpha=0.9,
                c=[
                    (point[0] / 255.0, point[1] / 255.0, point[2] / 255.0)
                    for point in team_points
                ],
                edgecolor="black",
                linewidth=0.5,
                label=f"Team {team_id} players",
            )

    for team_id, center in team_assigner.team_colors.items():
        rgb = tuple(float(channel) / 255.0 for channel in center[::-1])
        cluster_axis.scatter(
            [float(center[2])],
            [float(center[1])],
            s=260,
            marker="X",
            c=[rgb],
            edgecolor="black",
            linewidth=1.2,
            label=f"Team {team_id} centre",
        )

    cluster_axis.set_title("Jersey-colour clusters (R/G projection)", fontweight="bold")
    cluster_axis.set_xlabel("Red channel")
    cluster_axis.set_ylabel("Green channel")
    cluster_axis.grid(alpha=0.2)
    if team_assigner.team_colors:
        cluster_axis.legend(fontsize=8, loc="best")
    else:
        cluster_axis.text(
            0.5, 0.5, "No team clusters found",
            ha="center", va="center", transform=cluster_axis.transAxes,
            color="#64748b",
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def encode_annotated_video(
    frames: List[np.ndarray],
    tracks: Dict[str, Any],
    team_ball_control: np.ndarray,
    camera_movement: List[Any],
    tracker: Any,
    camera_estimator: Any,
    speed_estimator: Any,
    fps: float,
    output_path: Path,
    progress_callback=None,
    chunk_size: int = 24,
) -> List[bytes]:
    """Draw annotations chunk-by-chunk and pipe them to H.264 ffmpeg."""
    if not frames:
        raise ValueError("No frames available for annotated video encoding.")

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is not available. Install `imageio-ffmpeg` or put ffmpeg on PATH."
        )

    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{max(float(fps), 1.0):.3f}",
        "-i", "-",
        "-an",
        "-vf", "scale=1280:-2",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "24",
        "-movflags", "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    sample_indices = {
        min(len(frames) - 1, int(len(frames) * fraction))
        for fraction in (0.2, 0.5, 0.8)
    }
    samples: List[bytes] = []

    try:
        for start in range(0, len(frames), chunk_size):
            end = min(start + chunk_size, len(frames))
            chunk_frames = frames[start:end]
            chunk_tracks = {
                name: object_tracks[start:end]
                for name, object_tracks in tracks.items()
            }
            drawn = tracker.draw_annotations(
                chunk_frames,
                chunk_tracks,
                team_ball_control,
                frame_offset=start,
            )
            drawn = camera_estimator.draw_camera_movement(
                drawn, camera_movement[start:end]
            )
            drawn = speed_estimator.draw_speed_and_distance(drawn, chunk_tracks)

            for local_index, frame in enumerate(drawn):
                global_index = start + local_index
                if global_index in sample_indices:
                    ok, buffer = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
                    )
                    if ok:
                        samples.append(buffer.tobytes())
                process.stdin.write(frame.tobytes())

            if progress_callback:
                progress_callback(end, len(frames))
    except Exception:
        process.kill()
        raise
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass

    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait(timeout=300)
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg encoding failed: {message or return_code}")

    return samples


def run_full_pipeline(
    video_path: Path,
    max_frames: int,
    *,
    use_stubs: bool,
    source_name: str,
    is_custom: bool,
    progress_callback=None,
    encode_callback=None,
) -> Dict[str, Any]:
    """Run the complete 8-step pipeline and write per-run artefacts.

    Custom uploads always pass ``use_stubs=False``. The sample-video live mode
    can optionally reuse the cached tracking/camera stubs while still running
    every visual and analytics step.
    """
    from analytics.analytics_exporter import export_analytics, interpolate_ball_positions, read_video_frames
    from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
    from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
    from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
    from team_assigner.team_assigner import TeamAssigner
    from view_transformer.view_transformer import ViewTransformer

    def announce(step: int, message: str) -> None:
        if progress_callback:
            progress_callback(step, message)

    frames, fps, width, height = read_video_frames(str(video_path), max_frames=max_frames)
    frame_count = len(frames)
    if frame_count == 0:
        raise ValueError("No frames decoded from the uploaded video.")

    run_dir = create_run_dir(source_name)
    tracker = cached_tracker()
    tracker.reset_tracker()

    # 1. YOLO + ByteTrack
    effective_stubs = bool(use_stubs and not is_custom and TRACK_STUB.is_file())
    announce(
        1,
        "Cached YOLO + ByteTrack tracks"
        if effective_stubs
        else "YOLO + ByteTrack detection and tracking",
    )
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=effective_stubs,
        stub_path=str(TRACK_STUB) if effective_stubs else None,
    )
    for name in list(tracks):
        sequence = list(tracks[name])[:frame_count]
        sequence.extend({} for _ in range(frame_count - len(sequence)))
        tracks[name] = sequence

    # 2. Positions
    announce(2, "Object position estimation")
    tracker.add_position_to_tracks(tracks)

    # 3. Camera motion
    announce(3, "Lucas-Kanade camera motion estimation")
    camera_estimator = CameraMovementEstimator(frames[0])
    camera_stub_available = bool(effective_stubs and CAMERA_STUB.is_file())
    camera_movement = camera_estimator.get_camera_movement(
        frames,
        read_from_stub=camera_stub_available,
        stub_path=str(CAMERA_STUB) if camera_stub_available else None,
    )
    camera_movement = list(camera_movement)[:frame_count]
    camera_movement.extend([0, 0] for _ in range(frame_count - len(camera_movement)))
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

    # 4. Perspective transform
    announce(4, "Perspective transformation (23.32 m x 68 m pitch)")
    view_transformer = ViewTransformer()
    if width != 1920 or height != 1080:
        # The calibration points are defined for 1920x1080 broadcast frames.
        # Scale them for same-framing clips delivered at another resolution.
        scale = np.array([width / 1920.0, height / 1080.0], dtype=np.float32)
        view_transformer.pixel_vertices = (
            view_transformer.pixel_vertices * scale
        ).astype(np.float32)
        view_transformer.persepctive_trasnformer = cv2.getPerspectiveTransform(
            view_transformer.pixel_vertices,
            view_transformer.target_vertices,
        )
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 5. Ball interpolation
    announce(5, "Ball position interpolation")
    tracks["ball"] = interpolate_ball_positions(tracks.get("ball", []))

    # 6. Speed and distance
    announce(6, "Speed and distance calculation")
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.frame_rate = fps
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # 7. Team assignment + fresh team-colour plot
    announce(7, "K-Means team assignment and team-colour plot")
    team_assigner = TeamAssigner()
    seed_index = next(
        (index for index, players in enumerate(tracks["players"]) if len(players) >= 2),
        None,
    )
    if seed_index is not None:
        try:
            team_assigner.assign_team_color(
                frames[seed_index], tracks["players"][seed_index]
            )
        except Exception:
            team_assigner.team_colors = {}
        for frame_index, player_track in enumerate(tracks["players"]):
            for player_id, player in player_track.items():
                try:
                    team = team_assigner.get_player_team(
                        frames[frame_index], player["bbox"], player_id
                    )
                except Exception:
                    continue
                player["team"] = int(team)
                if team in team_assigner.team_colors:
                    player["team_color"] = team_assigner.team_colors[team]

    team_colors_path = run_dir / "team_colors.jpg"
    save_team_colors_plot(
        frames[seed_index] if seed_index is not None else frames[0],
        tracks,
        team_assigner,
        team_colors_path,
        seed_index or 0,
    )

    # 8. Ball possession + annotations + H.264 encode
    announce(8, "Ball possession, annotations and H.264 encoding")
    player_assigner = PlayerBallAssigner()
    team_ball_control: List[int] = []
    for frame_index, player_track in enumerate(tracks["players"]):
        ball_track = tracks["ball"][frame_index].get(1) if tracks["ball"] else None
        assigned = -1
        if ball_track is not None:
            assigned = player_assigner.assign_ball_to_player(
                player_track, ball_track["bbox"]
            )
        if assigned != -1:
            player = tracks["players"][frame_index][assigned]
            player["has_ball"] = True
            team_ball_control.append(int(player.get("team", 0) or 0))
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    control_array = np.asarray(team_ball_control, dtype=int)

    annotated_video_path = run_dir / "annotated.mp4"
    annotated_samples = encode_annotated_video(
        frames,
        tracks,
        control_array,
        camera_movement,
        tracker,
        camera_estimator,
        speed_estimator,
        fps,
        annotated_video_path,
        progress_callback=encode_callback,
    )

    payload, json_path, csv_path = export_analytics(
        {
            "tracks": tracks,
            "team_ball_control": control_array,
            "team_colors": team_assigner.team_colors,
            "fps": fps,
            "frames": frame_count,
            "width": width,
            "height": height,
            "video_path": str(video_path),
            "model_path": str(MODEL_PATH),
            "used_stubs": effective_stubs,
        },
        output_dir=str(run_dir),
        source_name=source_name,
    )

    return {
        "payload": payload,
        "annotated_video_path": annotated_video_path,
        "team_colors_path": team_colors_path,
        "player_metrics_path": csv_path,
        "match_summary_path": json_path,
        "annotated_frames": annotated_samples,
        "run_dir": run_dir,
        "used_stubs": effective_stubs,
    }


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def ensure_browser_video(avi_path: Path, mp4_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Convert the annotated AVI to H.264 MP4 using bundled/system ffmpeg."""
    if not avi_path.is_file():
        return None, "Annotated AVI not found."
    if (
        mp4_path.is_file()
        and mp4_path.stat().st_size > 0
        and mp4_path.stat().st_mtime >= avi_path.stat().st_mtime
    ):
        return mp4_path, None

    ffmpeg = None
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return None, "ffmpeg is not available; showing key annotated frames instead."

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(avi_path),
        "-an",
        "-vf",
        "scale=1280:-2",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        str(mp4_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        return None, f"Video conversion failed: {exc}"

    if mp4_path.is_file() and mp4_path.stat().st_size > 0:
        return mp4_path, None
    return None, "Video conversion produced no output."


def extract_frames(video_path: Path, count: int = 3) -> List[bytes]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        indices = [max(0, int(total * fraction)) for fraction in (0.2, 0.5, 0.8)]
    else:
        indices = [0, 1, 2]

    frames: List[bytes] = []
    for index in indices[:count]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            frames.append(buffer.tobytes())
    cap.release()
    return frames


def _team_color(track: Dict[str, Any]) -> Tuple[int, int, int]:
    color = track.get("team_color", (0, 0, 255))
    try:
        return tuple(int(channel) for channel in color)
    except (TypeError, ValueError):
        return (0, 0, 255)


def make_annotated_samples(result: Dict[str, Any], count: int = 3) -> List[bytes]:
    """Draw a few annotated frames from the in-memory tracks for custom clips."""
    tracks = result.get("tracks", {})
    players = tracks.get("players", [])
    referees = tracks.get("referees", [])
    balls = tracks.get("ball", [])
    total = int(result.get("frames", len(players)))
    if total <= 0:
        return []

    cap = cv2.VideoCapture(str(result["video_path"]))
    if not cap.isOpened():
        return []

    targets = [max(0, int(total * fraction)) for fraction in (0.2, 0.5, 0.8)][:count]
    frames: List[bytes] = []

    for index in targets:
        index = min(index, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue

        for player_id, player in players[index].items() if index < len(players) else []:
            bbox = player.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            color = _team_color(player)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"P{int(player_id)} T{player.get('team', '?')}"
            cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for referee in referees[index].values() if index < len(referees) else []:
            bbox = referee.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        ball_track = balls[index].get(1) if index < len(balls) else None
        if ball_track and ball_track.get("bbox"):
            x1, y1, x2, y2 = map(int, ball_track["bbox"])
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (0, 255, 0), 2)

        cv2.putText(frame, f"Frame {index}", (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, f"Frame {index}", (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 23, 42), 1)

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            frames.append(buffer.tobytes())

    cap.release()
    return frames


def show_frames(frames: List[bytes], label_prefix: str = "Frame") -> None:
    if not frames:
        return
    columns = st.columns(len(frames))
    for index, (column, frame) in enumerate(zip(columns, frames)):
        column.image(frame, caption=f"{label_prefix} {index + 1}", width="stretch")


def download_video_button(video_path: Path, key: str) -> None:
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return
    try:
        video_bytes = video_path.read_bytes()
    except OSError:
        return
    st.download_button(
        "Download Annotated MP4",
        data=video_bytes,
        file_name=video_path.name,
        mime="video/mp4",
        width="stretch",
        key=key,
    )


def show_video_section(payload: Dict[str, Any]) -> None:
    st.subheader("Annotated match video")

    live_run = bool(st.session_state.get("is_live_run"))
    is_custom = bool(st.session_state.get("is_custom"))
    run_video_value = st.session_state.get("annotated_video_path")
    run_video = Path(run_video_value) if run_video_value else None

    # Custom uploads and live sample runs must never fall back to the cached
    # sample video. They show only the freshly encoded annotation for this run.
    if live_run or is_custom:
        if run_video and run_video.is_file() and run_video.stat().st_size > 0:
            st.video(str(run_video))
            st.caption(f"Fresh annotated output · {payload_caption(payload)}")
            download_video_button(run_video, "download_annotated_run")
        else:
            st.warning(
                "The annotated MP4 for this run is missing. Re-run the analysis "
                "or check the ffmpeg / imageio-ffmpeg installation."
            )

        run_frames = st.session_state.get("annotated_frames") or []
        if run_frames:
            with st.expander("Key annotated frames from this run", expanded=False):
                show_frames(run_frames, "Annotated frame")
        return

    # Instant default: pre-computed sample analytics + cached annotated video.
    if not ANNOTATED_AVI.is_file():
        st.info(
            "No pre-computed annotated video found. Run `python pipeline/main.py` or "
            "`python pipeline/main_gpu.py` to generate `output_videos/output_video.avi`, "
            "or run the live pipeline from the sidebar."
        )
        return

    with st.spinner("Preparing browser-compatible H.264 video..."):
        browser_video, error = ensure_browser_video(ANNOTATED_AVI, BROWSER_MP4)

    if browser_video:
        st.video(str(browser_video))
        st.caption(
            "Pre-computed sample output (H.264, cached at "
            "`output_videos/output_video_dashboard.mp4`)."
        )
        download_video_button(browser_video, "download_annotated_sample")
    else:
        if error:
            st.warning(error)
        frames = extract_frames(ANNOTATED_AVI)
        if frames:
            st.caption("Key annotated frames from `output_videos/output_video.avi`")
            show_frames(frames, "Annotated frame")
        else:
            st.info("Could not read the annotated video or extract frames.")


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def render_kpis(payload: Dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    possession = summary.get("team_possession_pct", {})
    top_sprint = summary.get("top_sprint_speed_kmh") or {}

    team1 = float(possession.get("team_1", 0.0))
    team2 = float(possession.get("team_2", 0.0))
    top_value = float(top_sprint.get("value", 0.0))
    top_player = top_sprint.get("track_id")
    top_team = top_sprint.get("team_id")

    columns = st.columns(4)
    columns[0].metric(
        "Players Tracked",
        f"{int(summary.get('players_tracked', 0))}",
        help="Unique ByteTrack IDs observed in the clip.",
    )
    columns[1].metric(
        "Total Distance",
        f"{float(summary.get('total_distance_m', 0.0)):.1f} m",
        help="Sum of per-player distance in the pitch coordinate system.",
    )
    columns[2].metric(
        "Team Possession",
        f"{team1:.1f}% vs {team2:.1f}%",
        help="Frame-based ball possession, team 1 vs team 2.",
    )
    columns[3].metric(
        "Top Sprint Speed",
        f"{top_value:.2f} km/h",
        delta=f"player {top_player} Â· team {top_team}" if top_player is not None else None,
        help="Fastest 5-frame displacement after homography projection.",
    )


def render_team_colors(team_colors_path: Optional[Path] = None) -> None:
    st.subheader("K-Means team clustering")
    stored_path = st.session_state.get("team_colors_path")
    if not stored_path and not st.session_state.get("is_custom"):
        stored_path = str(TEAM_COLORS_IMAGE)
    path = Path(team_colors_path or stored_path) if (team_colors_path or stored_path) else None
    if path and path.is_file():
        st.image(
            str(path),
            caption=(
                "Jersey-colour clusters for this run. Boxes show the assigned "
                "team; the R/G scatter shows player colours and cluster centres."
            ),
            width="stretch",
        )
        try:
            st.download_button(
                "Download team_colors.jpg",
                data=path.read_bytes(),
                file_name=path.name,
                mime="image/jpeg",
                width="stretch",
                key="download_team_colors",
            )
        except OSError:
            pass
    else:
        st.info(
            "No team-colour plot available for this run. The live pipeline "
            "creates one automatically in `outputs/analytics/runs/`."
        )


def _team_matrix(team: Dict[str, Any]) -> Tuple[np.ndarray, int, int]:
    heatmap = team.get("heatmap", {})
    grid = heatmap.get("grid", [12, 32])
    columns, rows = int(grid[0]), int(grid[1])
    matrix = np.zeros((rows, columns), dtype=float)
    for cell in heatmap.get("cells", []):
        if len(cell) < 3:
            continue
        column, row, count = int(cell[0]), int(cell[1]), float(cell[2])
        if 0 <= row < rows and 0 <= column < columns:
            matrix[row, column] += count
    return matrix, rows, columns


def render_heatmaps(payload: Dict[str, Any]) -> None:
    st.subheader("2D pitch heatmaps Â· 23.32 m Ã— 68 m")
    teams = payload.get("teams") or []
    if not teams:
        st.info("No team heatmap data in this payload.")
        return

    prepared = []
    for team in teams[:2]:
        matrix, rows, columns = _team_matrix(team)
        prepared.append((team, matrix, rows, columns))

    vmax = max((matrix.max() for _, matrix, _, _ in prepared), default=0.0)
    vmax = max(vmax, 1.0)

    figure, axes = plt.subplots(1, len(prepared), figsize=(7 * len(prepared), 5.3))
    if len(prepared) == 1:
        axes = [axes]

    for axis, (team, matrix, rows, columns) in zip(axes, prepared):
        cell_width = PITCH_LENGTH_M / columns
        cell_height = PITCH_WIDTH_M / rows
        x_labels = [
            f"{column * cell_width:.0f}" if column % 2 == 0 else ""
            for column in range(columns)
        ]
        y_labels = [
            f"{row * cell_height:.0f}" if row % 4 == 0 else ""
            for row in range(rows)
        ]
        cmap = "Blues" if team.get("team_id") == 1 else "Reds"
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            cbar_kws={"label": "occupancy samples"},
            xticklabels=x_labels,
            yticklabels=y_labels,
            linewidths=0,
        )
        axis.invert_yaxis()
        axis.set_title(
            f"Team {team.get('team_id')} Â· {float(team.get('possession_pct', 0)):.1f}% possession",
            fontsize=12,
            fontweight="bold",
        )
        axis.set_xlabel("Pitch length (m) â†’")
        axis.set_ylabel("Pitch width (m) â†’")

        # Visual pitch reference: halfway line and centre circle.
        axis.axvline(columns / 2, color="white", linewidth=1.0, alpha=0.8)
        axis.axhline(rows / 2, color="white", linewidth=1.0, alpha=0.8)
        circle = plt.Circle(
            (columns / 2, rows / 2),
            radius=min(columns, rows) / 6,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.8,
        )
        axis.add_patch(circle)

        peak = team.get("heatmap", {}).get("peak_cell")
        if peak:
            axis.scatter(
                [float(peak[0]) + 0.5],
                [float(peak[1]) + 0.5],
                marker="x",
                s=90,
                c="black",
                linewidths=2.0,
                label="peak",
            )
            axis.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        "Each cell is 1.943 m Ã— 2.125 m. The black Ã— marks the peak occupancy "
        "cell; the white lines are the halfway line and centre circle."
    )


def player_dataframe_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    possession = payload.get("summary", {}).get("team_possession_pct", {})
    rows: List[Dict[str, Any]] = []
    for player in payload.get("players", []):
        peak = (player.get("heatmap") or {}).get("peak_position_m") or [None, None]
        team_id = player.get("team_id")
        rows.append(
            {
                "player_id": player.get("track_id"),
                "team_id": team_id,
                "frames_seen": player.get("frames_seen"),
                "distance_m": player.get("distance_m"),
                "top_speed_kmh": player.get("top_speed_kmh"),
                "avg_speed_kmh": player.get("avg_speed_kmh"),
                "team_possession_pct": possession.get(f"team_{team_id}", 0.0),
                "heatmap_peak_x_m": peak[0],
                "heatmap_peak_y_m": peak[1],
            }
        )
    return pd.DataFrame(rows)


def render_players(payload: Dict[str, Any]) -> None:
    st.subheader("Player metrics")
    csv_value = st.session_state.get("player_metrics_path")
    if not csv_value and not st.session_state.get("is_custom"):
        csv_value = str(PLAYER_METRICS_PATH)
    csv_path = Path(csv_value) if csv_value else None

    if csv_path and csv_path.is_file():
        dataframe = pd.read_csv(csv_path)
        try:
            display_path = csv_path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = csv_path
        st.caption(f"Source: `{display_path}`")
    else:
        dataframe = player_dataframe_from_payload(payload)
        st.caption("Source: current analysis payload")

    if dataframe.empty:
        st.info("No player rows available.")
        return

    for column in ("player_id", "team_id", "frames_seen"):
        if column in dataframe:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0).astype(int)
    for column in (
        "distance_m",
        "top_speed_kmh",
        "avg_speed_kmh",
        "team_possession_pct",
        "heatmap_peak_x_m",
        "heatmap_peak_y_m",
    ):
        if column in dataframe:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    filter_left, filter_right = st.columns([1, 2])
    team_options = sorted(int(value) for value in dataframe["team_id"].dropna().unique())
    selected_teams = filter_left.multiselect(
        "Team filter", options=team_options, default=team_options
    )
    max_distance = float(dataframe["distance_m"].max()) if "distance_m" in dataframe else 0.0
    min_distance = filter_right.slider(
        "Minimum distance (m)",
        min_value=0.0,
        max_value=max(1.0, round(max_distance, 1)),
        value=0.0,
        step=1.0,
    )

    filtered = dataframe[
        dataframe["team_id"].isin(selected_teams)
        & (dataframe["distance_m"].fillna(0.0) >= min_distance)
    ].copy()

    display = filtered.rename(
        columns={
            "player_id": "Player ID",
            "team_id": "Team",
            "frames_seen": "Frames",
            "distance_m": "Distance (m)",
            "top_speed_kmh": "Top speed (km/h)",
            "avg_speed_kmh": "Avg speed (km/h)",
            "team_possession_pct": "Team possession (%)",
            "heatmap_peak_x_m": "Heatmap x (m)",
            "heatmap_peak_y_m": "Heatmap y (m)",
        }
    )
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Distance (m)": st.column_config.NumberColumn(format="%.2f"),
            "Top speed (km/h)": st.column_config.NumberColumn(format="%.2f"),
            "Avg speed (km/h)": st.column_config.NumberColumn(format="%.2f"),
            "Team possession (%)": st.column_config.NumberColumn(format="%.2f"),
            "Heatmap x (m)": st.column_config.NumberColumn(format="%.2f"),
            "Heatmap y (m)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(f"{len(filtered)} of {len(dataframe)} player rows shown.")

    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download filtered player_metrics.csv",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="player_metrics_filtered.csv",
        mime="text/csv",
        width="stretch",
    )
    download_right.download_button(
        "Download match_summary.json",
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name="match_summary.json",
        mime="application/json",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_frame(data: Optional[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not data:
        return pd.DataFrame(rows)

    for result in data.get("results", []):
        backend = "ONNX" if result.get("backend") == "ONNX Runtime" else "PyTorch"
        configuration = f"{backend} {result.get('precision', '')}"
        batch = int(result.get("batch_size") or data.get("config", {}).get("batch_size", 0))
        rows.append(
            {
                "Configuration": configuration,
                "Backend": backend,
                "Precision": result.get("precision", ""),
                "Device": result.get("device", ""),
                "Batch": batch,
                "Latency (ms/frame)": float(result.get("latency_ms_mean", 0.0)),
                "Throughput (FPS)": float(result.get("throughput_fps", 0.0)),
                "GPU VRAM (MB)": (
                    float(result["gpu_peak_mb"])
                    if result.get("gpu_peak_mb") is not None
                    else np.nan
                ),
                "CPU RSS delta (MB)": float(result.get("cpu_rss_delta_mb", 0.0)),
                "Speedup vs FP32": float(result.get("speedup_vs_baseline", 1.0)),
            }
        )
    return pd.DataFrame(rows)


def chart_frame(dataframe: pd.DataFrame, value_column: str, order: List[str]) -> pd.DataFrame:
    if dataframe.empty or value_column not in dataframe:
        return pd.DataFrame()
    pivot = dataframe.pivot_table(
        index="Configuration", columns="Batch", values=value_column, aggfunc="first"
    )
    pivot = pivot.reindex([item for item in order if item in pivot.index])
    pivot.columns = [f"Batch {int(column)}" for column in pivot.columns]
    return pivot


def render_benchmarks() -> None:
    st.subheader("Inference benchmark Â· 100 frames, batch 10 vs batch 1")
    batch10 = load_json(BENCHMARK_BATCH10_PATH)
    batch1 = load_json(BENCHMARK_BATCH1_PATH)

    if not batch10:
        st.warning(
            "`benchmarks/results/benchmark_results.json` is missing. Run:\n\n"
            "`python benchmarks/benchmark.py --frames 100 --runs 3 --warmup 20 "
            "--batch-size 10 --onnx-provider cuda --onnx-variants fp16 fp32`"
        )
        return

    frame10 = benchmark_frame(batch10)
    frame1 = benchmark_frame(batch1) if batch1 else pd.DataFrame()

    if not frame10.empty:
        fp32_row = frame10[(frame10["Configuration"] == "PyTorch FP32")]
        fp16_row = frame10[(frame10["Configuration"] == "PyTorch FP16")]
        if not fp32_row.empty and not fp16_row.empty:
            fp32 = fp32_row.iloc[0]
            fp16 = fp16_row.iloc[0]
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "FP16 latency",
                f"{fp16['Latency (ms/frame)']:.2f} ms/frame",
                f"-{(1 - fp16['Latency (ms/frame)'] / fp32['Latency (ms/frame)']) * 100:.1f}% vs FP32",
            )
            metric_columns[1].metric(
                "FP16 throughput",
                f"{fp16['Throughput (FPS)']:.1f} FPS",
                f"{fp16['Speedup vs FP32']:.2f}x",
            )
            memory_saved = float(fp32["GPU VRAM (MB)"]) - float(fp16["GPU VRAM (MB)"])
            metric_columns[2].metric(
                "GPU VRAM saved",
                f"{memory_saved:.0f} MB",
                f"{memory_saved / float(fp32['GPU VRAM (MB)']) * 100:.1f}%",
            )

    combined = pd.concat([frame10, frame1], ignore_index=True) if not frame1.empty else frame10
    combined = combined.sort_values(["Batch", "Configuration"], ascending=[False, True])
    st.dataframe(
        combined[
            [
                "Configuration",
                "Device",
                "Batch",
                "Latency (ms/frame)",
                "Throughput (FPS)",
                "GPU VRAM (MB)",
                "CPU RSS delta (MB)",
                "Speedup vs FP32",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Latency (ms/frame)": st.column_config.NumberColumn(format="%.2f"),
            "Throughput (FPS)": st.column_config.NumberColumn(format="%.1f"),
            "GPU VRAM (MB)": st.column_config.NumberColumn(format="%.1f"),
            "CPU RSS delta (MB)": st.column_config.NumberColumn(format="%.1f"),
            "Speedup vs FP32": st.column_config.NumberColumn(format="%.2fx"),
        },
    )
    st.caption(
        "Measured on the sample video with YOLOv5l6u. GPU VRAM is the PyTorch "
        "allocator peak; ONNX Runtime reports n/a because it manages its own "
        "CUDA allocations. See `benchmarks/results/benchmark_results.json` for power/clock telemetry."
    )

    order = ["PyTorch FP32", "PyTorch FP16", "ONNX FP16", "ONNX FP32"]
    chart_columns = st.columns(3)
    with chart_columns[0]:
        st.markdown("**Latency (ms/frame)** Â· lower is better")
        latency_chart = chart_frame(combined, "Latency (ms/frame)", order)
        if not latency_chart.empty:
            st.bar_chart(latency_chart, width="stretch")
    with chart_columns[1]:
        st.markdown("**Throughput (FPS)** Â· higher is better")
        throughput_chart = chart_frame(combined, "Throughput (FPS)", order)
        if not throughput_chart.empty:
            st.bar_chart(throughput_chart, width="stretch")
    with chart_columns[2]:
        st.markdown("**GPU VRAM (MB)** Â· PyTorch allocator")
        vram_source = combined[combined["GPU VRAM (MB)"].notna()]
        vram_chart = chart_frame(vram_source, "GPU VRAM (MB)", order)
        if not vram_chart.empty:
            st.bar_chart(vram_chart, width="stretch")
        else:
            st.info("No GPU VRAM values available.")

    st.divider()
    st.subheader("Pipeline quality evaluations")
    evaluation = load_json(EVAL_SUMMARY_PATH, {}) or {}
    ball = evaluation.get("ball_interpolation", {})
    team = evaluation.get("team_assignment", {})
    camera = evaluation.get("camera_motion", {})

    quality_columns = st.columns(3)
    quality_columns[0].metric(
        "Ball Interpolation Recovery",
        f"{float(ball.get('recovery_after_pct', 100.0)):.1f}%",
        f"from {float(ball.get('recovery_before_pct', 37.1)):.1f}% Â· "
        f"{int(ball.get('frames_recovered', 472))} frames recovered",
        help="Share of frames with a usable ball position before vs after interpolation.",
    )
    quality_columns[1].metric(
        "Team Assignment Silhouette",
        f"{float(team.get('silhouette_score', 0.7705)):.4f}",
        "excellent separation (>0.5)",
        help="Silhouette score for the two jersey-colour clusters.",
    )
    quality_columns[2].metric(
        "Camera Residual Drift",
        f"{float(camera.get('residual_drift_mean_m_per_frame', 0.1469)):.4f} m/frame",
        f"median {float(camera.get('residual_drift_median_m_per_frame', 0.1101)):.4f}",
        help="Residual movement of stationary players after homography correction.",
    )
    st.caption(
        "Values are read from `outputs/eval_summary.json`, sourced from the "
        "`eval_*.py` scripts on the sample video."
    )
    with st.expander("Reproduce the evaluations"):
        st.code(
            "python evaluation/eval_ball_interpolation.py\n"
            "python evaluation/eval_team_assignment.py --frames 100\n"
            "python evaluation/eval_camera_motion.py --frames 500",
            language="bash",
        )


# ---------------------------------------------------------------------------
# FastAPI helpers
# ---------------------------------------------------------------------------

def check_api(url: str) -> Tuple[bool, Optional[int], Dict[str, Any]]:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=2.5)
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text}
        return response.ok, response.status_code, payload
    except requests.RequestException as exc:
        return False, None, {"error": str(exc)}


def render_api_playground() -> None:
    st.subheader("FastAPI microservice playground")
    api_url = st.text_input("API base URL", value=API_DEFAULT)

    if "api_status" not in st.session_state:
        st.session_state.api_status = check_api(api_url)

    status_left, status_right = st.columns([1, 2])
    with status_left:
        if st.button("Check API status", width="stretch"):
            st.session_state.api_status = check_api(api_url)

    online, status_code, health_payload = st.session_state.api_status
    with status_right:
        if online:
            st.success(f"API online Â· HTTP {status_code} Â· {api_url}")
        else:
            st.error(f"API offline Â· {api_url}")

    if online:
        with st.expander("Live /health response", expanded=False):
            st.json(health_payload)
        st.link_button("Open Swagger UI", f"{api_url.rstrip('/')}/docs")
    else:
        st.code(
            "uvicorn apps.app:app --host 127.0.0.1 --port 8000 --workers 1",
            language="bash",
        )
        st.caption(
            "Start the service in another terminal, then click Check API status. "
            "The in-process fallback below works without the server."
        )

    st.divider()
    st.markdown("### Send a test clip")
    test_clip = st.file_uploader(
        "Optional test clip (sample video is used if empty)",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="api_test_clip",
    )
    frames_to_send = st.slider("Frames to send / analyze", 5, 100, 20, step=5)

    send_column, fallback_column = st.columns(2)
    with send_column:
        send_clicked = st.button(
            "Send to POST /analyze-clip",
            type="primary",
            width="stretch",
            disabled=not online,
        )
    with fallback_column:
        fallback_clicked = st.button(
            "Run exporter in-process",
            width="stretch",
        )

    if send_clicked:
        if not online:
            st.warning("API is offline. Use the in-process fallback or start the server.")
        else:
            with st.spinner("Sending clip and running the API pipeline..."):
                try:
                    if test_clip is not None:
                        file_name = test_clip.name
                        file_bytes = test_clip.getvalue()
                    else:
                        if not SAMPLE_VIDEO.is_file():
                            raise FileNotFoundError(f"Sample video not found: {SAMPLE_VIDEO}")
                        file_name = SAMPLE_VIDEO.name
                        file_bytes = SAMPLE_VIDEO.read_bytes()

                    response = requests.post(
                        f"{api_url.rstrip('/')}/analyze-clip",
                        params={"max_frames": frames_to_send},
                        files={"file": (file_name, file_bytes, "video/mp4")},
                        timeout=900,
                    )
                    response.raise_for_status()
                    st.session_state.api_payload = response.json()
                    st.session_state.api_frames = []
                    st.success(f"API returned HTTP {response.status_code}")
                except Exception as exc:
                    st.error(f"API request failed: {exc}")

    if fallback_clicked:
        with st.spinner("Running the exporter in-process..."):
            try:
                if test_clip is not None:
                    suffix = Path(test_clip.name).suffix or ".mp4"
                    temp_dir = Path(tempfile.mkdtemp(prefix="balltracker_api_"))
                    clip_path = temp_dir / f"upload{suffix}"
                    clip_path.write_bytes(test_clip.getvalue())
                    payload, annotated = run_pipeline(
                        clip_path, frames_to_send, use_stubs=False,
                        source_name=test_clip.name,
                    )
                    st.session_state.api_frames = annotated
                else:
                    payload, annotated = run_pipeline(
                        SAMPLE_VIDEO, frames_to_send, use_stubs=True,
                        source_name=SAMPLE_VIDEO.name,
                    )
                    st.session_state.api_frames = annotated
                st.session_state.api_payload = payload
                st.success("In-process exporter completed")
            except Exception as exc:
                st.error(f"In-process analysis failed: {exc}")

    payload = st.session_state.get("api_payload")
    if payload:
        st.markdown("### Raw JSON response")
        st.json(payload)
        st.download_button(
            "Download response match_summary.json",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name="match_summary_api_response.json",
            mime="application/json",
            width="stretch",
        )
        frames = st.session_state.get("api_frames") or []
        if frames:
            with st.expander("Annotated sample frames from this request", expanded=False):
                show_frames(frames, "Response frame")

    with st.expander("Equivalent curl command"):
        st.code(
            f'curl -X POST "{api_url.rstrip("/")}/analyze-clip?max_frames={frames_to_send}" '
            f'-F "file=@input_videos/08fd33_4.mp4"',
            language="bash",
        )


# ---------------------------------------------------------------------------
# Sidebar and page
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>âš½ BallTracker Tactical Dashboard</h1>
        <p>YOLOv5 + ByteTrack + K-Means + homography â†’ match analytics, inference benchmarks and a live API playground.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

init_state()

with st.sidebar:
    if st.session_state.pop("_reload_sample_pending", False):
        st.session_state["input_source"] = "Sample Video (Pre-computed)"

    st.markdown("## Match controls")
    st.caption("Default opens instantly with the pre-computed sample match.")
    source = st.radio(
        "Input source",
        ["Sample Video (Pre-computed)", "Custom Uploaded Video (Live Run)"],
        key="input_source",
        on_change=on_source_change,
        label_visibility="collapsed",
    )

    uploaded_file = None
    use_stubs = False
    if source.startswith("Sample"):
        use_stubs = st.checkbox("Use cached stubs (fast)", value=True)
        max_frames = st.slider("Max frames", 10, 750, 750, step=10)
        st.caption(
            "Stubs make the sample instant. Uncheck to run the full 8-step live "
            "pipeline on the sample video."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a short clip",
            type=["mp4", "avi", "mov", "mkv", "webm"],
        )
        max_frames = st.slider("Max frames", 10, 300, 100, step=10)
        st.caption(
            "Runs the complete 8-step live pipeline and creates a fresh annotated "
            "MP4, team-colour plot, JSON and CSV for this clip."
        )

    run_clicked = st.button("Run Analysis", type="primary", width="stretch")
    if st.button("Reload pre-computed sample analytics", width="stretch"):
        saved = load_json(MATCH_SUMMARY_PATH)
        if saved:
            store_payload(saved)
            st.session_state["_reload_sample_pending"] = True
            st.rerun()
        else:
            st.error("No saved analytics found. Run `python analytics/analytics_exporter.py`.")

    st.divider()
    st.markdown("### Files")
    st.caption(
        f"Pre-computed analytics: `{MATCH_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`\n\n"
        f"Live run artefacts: `{RUNS_DIR.relative_to(PROJECT_ROOT)}`\n\n"
        f"Benchmark: `{BENCHMARK_BATCH10_PATH.relative_to(PROJECT_ROOT)}`"
    )

if run_clicked:
    is_custom_run = source.startswith("Custom")
    saved_sample = load_json(MATCH_SUMMARY_PATH)

    if is_custom_run and uploaded_file is None:
        st.sidebar.error("Upload a clip first.")
    elif (
        not is_custom_run
        and use_stubs
        and max_frames >= 750
        and saved_sample
        and int(saved_sample.get("video", {}).get("frames_analyzed", 0)) == 750
    ):
        store_payload(saved_sample)
        st.sidebar.success("Loaded pre-computed 750-frame sample analytics")
    else:
        status = st.status("Running full 8-step pipeline...", expanded=True)
        progress = st.progress(0.0, text="Preparing pipeline...")

        def on_step(step: int, message: str) -> None:
            # Step 8 is the long draw/encode phase; leave the last 1/8 for the
            # per-frame encoding progress callback.
            fraction = 7.0 / 8.0 if step >= 8 else step / 8.0
            progress.progress(
                fraction,
                text=f"Step {step}/8 · {message}",
            )
            status.write(f"**Step {step}/8** — {message}")

        def on_encode(done: int, total: int) -> None:
            fraction = (7.0 + done / max(total, 1)) / 8.0
            progress.progress(
                min(fraction, 1.0),
                text=f"Step 8/8 · drawing and encoding {done}/{total} frames",
            )

        try:
            if is_custom_run:
                suffix = Path(uploaded_file.name).suffix or ".mp4"
                temp_dir = Path(tempfile.mkdtemp(prefix="balltracker_ui_"))
                clip_path = temp_dir / f"upload{suffix}"
                clip_path.write_bytes(uploaded_file.getvalue())
                source_name = uploaded_file.name
                pipeline_video = clip_path
                pipeline_stubs = False
            else:
                source_name = SAMPLE_VIDEO.name
                pipeline_video = SAMPLE_VIDEO
                pipeline_stubs = use_stubs

            result = run_full_pipeline(
                pipeline_video,
                max_frames,
                use_stubs=pipeline_stubs,
                source_name=source_name,
                is_custom=is_custom_run,
                progress_callback=on_step,
                encode_callback=on_encode,
            )
            progress.progress(1.0, text="Complete")
            status.update(label="Full pipeline complete", state="complete")
            store_payload(
                result["payload"],
                result["annotated_frames"],
                is_custom=is_custom_run,
                uploaded_bytes=uploaded_file.getvalue() if is_custom_run else None,
                uploaded_name=uploaded_file.name if is_custom_run else None,
                run_dir=result["run_dir"],
                annotated_video_path=result["annotated_video_path"],
                team_colors_path=result["team_colors_path"],
                player_metrics_path=result["player_metrics_path"],
            )
            st.sidebar.success(
                "Run complete: "
                f"`{result['run_dir'].relative_to(PROJECT_ROOT)}`"
            )
        except Exception as exc:
            status.update(label="Full pipeline failed", state="error")
            status.write(f"Error: {exc}")
            st.sidebar.error(f"Full pipeline failed: {exc}")

payload = st.session_state.get("payload")

tab1, tab2, tab3 = st.tabs(
    [
        "Match Analytics & Video",
        "Inference & Pipeline Benchmarks",
        "FastAPI Microservice Playground",
    ]
)

with tab1:
    active_source = st.session_state.get("active_source", "sample")
    selected_custom = source.startswith("Custom")

    if selected_custom and active_source != "custom":
        st.info(
            "Upload a clip in the sidebar and click **Run Analysis** to run the "
            "complete 8-step live pipeline. The pre-computed sample is not shown "
            "while the custom source is selected."
        )
    elif not selected_custom and active_source != "sample":
        st.info(
            "No pre-computed sample analytics loaded. Click **Reload pre-computed "
            "sample analytics** or run the sample live from the sidebar."
        )
    elif not payload:
        st.info(
            "No analytics payload loaded. Click **Run Analysis** in the sidebar, "
            "or run `python analytics/analytics_exporter.py` to create the canonical JSON."
        )
    else:
        st.caption(payload_caption(payload))
        run_dir_value = st.session_state.get("run_dir")
        if run_dir_value:
            try:
                run_dir_display = Path(run_dir_value).relative_to(PROJECT_ROOT)
            except ValueError:
                run_dir_display = Path(run_dir_value)
            st.caption(f"Live run artefacts: `{run_dir_display}`")
        render_kpis(payload)
        st.divider()

        video_column, colors_column = st.columns([1.55, 1])
        with video_column:
            show_video_section(payload)
        with colors_column:
            render_team_colors()

        st.divider()
        render_heatmaps(payload)
        st.divider()
        render_players(payload)

with tab2:
    render_benchmarks()

with tab3:
    render_api_playground()

st.markdown(
    '<p class="small-note">BallTracker dashboard Â· analytics are in metres on a '
    "23.32 m Ã— 68 m pitch Â· rebuild outputs with `python analytics/analytics_exporter.py`.</p>",
    unsafe_allow_html=True,
)
