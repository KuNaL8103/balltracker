# ⚽ BallTracker Project

## Introduction

This project aims to **detect and track football players, referees, and the ball** in match videos using **YOLO**, one of the leading object detection models. The system enhances detection accuracy through training and assigns players to teams based on **t-shirt color clustering via KMeans**.

Key features include:

- Team identification using **KMeans clustering** on pixel data.
- **Ball possession analytics** by tracking which team controls the ball.
- **Optical Flow** to measure **camera movement** across frames.
- **Perspective transformation** to calculate movement in **real-world units (meters)**.
- **Speed and distance tracking** for individual players.
- **Interactive Streamlit dashboard** for match analytics, inference benchmarks and a live API playground.

This project showcases a combination of computer vision and machine learning techniques, making it valuable for both beginners and experienced engineers.

---

## Modules Used

- **YOLO (v5)**: Object detection (players, referees, ball)
- **ByteTrack**: Multi-object tracking
- **KMeans**: T-shirt color clustering (team assignment)
- **Optical Flow**: Camera motion estimation
- **Perspective Transformation**: Real-world distance estimation
- **Speed/Distance Calculation**: Player movement tracking

---

## Trained Models

Download the trained YOLO model:

- [Trained YOLO Model](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)
  → Place it in the `models/` folder as `best.pt`.

---

## Sample Input Video

Download the sample match video:

- [Sample Input Video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)
  → Place it in the `input_videos/` folder.

---

## Dataset

Download the football players detection dataset:

- [Roboflow Football Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)
  → Use this dataset to train your own YOLO model for player detection.

---

## Requirements

To run this project, make sure you have the following installed:

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for fast inference)

### Install dependencies:

```bash
pip install -r requirements.txt
```

### For GPU acceleration (recommended):

```bash
# Install CUDA-enabled PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify GPU is detected
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Running the Project

### Standard CPU Version

```bash
python pipeline/main.py
```

Output: `output_videos/output_video.avi`

### GPU-Accelerated Version (Recommended)

```bash
python pipeline/main_gpu.py
```

Output: `output_videos/output_video_gpu.avi`

**GPU optimizations in `pipeline/main_gpu.py`:**

- CUDA model initialization (`model.to('cuda')`)
- FP16 half-precision inference (`model.half()`)
- Batch size 32 (vs 20 on CPU)
- cuDNN benchmark + TF32 enabled
- ~74 FPS tracking on RTX 4050 (vs ~10 FPS on CPU)

> Absolute FPS depends heavily on the GPU power limit and thermal state.
> Controlled 100-frame benchmark metadata is stored in `benchmarks/results/`.

---

## Metric Validation Scripts

Run evaluation scripts to validate tracking quality:

| Script | Metric | Description |
| --- | --- | --- |
| `evaluation/eval_map.py` | mAP@0.5 | Detection accuracy; without `--data` prints model metadata/instructions |
| `evaluation/eval_tracking.py` | MOTA / ID Switches | Tracking quality and ID switches |
| `evaluation/eval_team_assignment.py` | Silhouette Score | Team colour separation quality |
| `evaluation/eval_camera_motion.py` | Camera Drift | Raw and residual drift after homography |
| `benchmarks/benchmark_speed.py` | FPS | End-to-end processing speed |
| `evaluation/eval_ball_interpolation.py` | Recovery Rate | Ball position interpolation recovery |
| `evaluation/eval_output.py` | Frame Drop Rate | Output frame completeness |

### Run all evaluations:

```bash
# Tracking quality
python evaluation/eval_tracking.py --frames 500

# Team assignment (silhouette >0.5 = good)
python evaluation/eval_team_assignment.py --frames 100

# Camera motion drift
python evaluation/eval_camera_motion.py --frames 500

# Ball interpolation recovery
python evaluation/eval_ball_interpolation.py

# Output frame verification
python evaluation/eval_output.py

# FPS benchmark
python benchmarks/benchmark_speed.py --frames 100 --runs 3

# mAP@0.5 (requires validation dataset)
python evaluation/eval_map.py --data path/to/data.yaml

# Model metadata only (classes, parameters, imgsz, layer summary)
python evaluation/eval_map.py
```

### Expected Results (on sample video):

- **mAP@0.5**: ~0.82 (with Roboflow validation set)
- **ID Switches**: ~19 (39 unique tracks, 100 frames)
- **Silhouette Score**: 0.7705 ✓ EXCELLENT
- **Residual Camera Drift**: 0.1469 m/frame mean, 0.1101 median ~ GOOD
- **Ball Recovery**: 100% (472 frames recovered) ✓ EXCELLENT
- **Frame Drop Rate**: 0.00% ✓ PERFECT
- **Tracking FPS**: ~74 FPS (GPU, warm)

---

## Production Engineering Add-ons

These additions turn the tutorial pipeline into something that can be measured,
exported and served as an API. See `benchmarks/results/` for measured benchmark numbers and
the dashboard for an interactive view.

### 1. Inference benchmark (`benchmarks/benchmark.py`)

Compares PyTorch FP32, PyTorch FP16 and ONNX Runtime on the same 100 frames.
Timing covers the model forward pass only (decode / letterbox / NMS excluded)
so the backends are directly comparable.

```bash
# Three-way comparison (PyTorch FP32 / FP16 / ONNX FP32), batch 10
python benchmarks/benchmark.py --frames 100 --runs 3 --batch-size 10

# Full GPU matrix: ONNX FP32 + FP16 on CUDA
# --require-ac refuses to overwrite results if the laptop switches to battery
python benchmarks/benchmark.py --frames 100 --runs 3 --warmup 20 --batch-size 10 \
    --onnx-variants fp16 fp32 --onnx-provider cuda --require-ac

# Single-frame latency reference
python benchmarks/benchmark.py --frames 100 --runs 3 --batch-size 1 \
    --onnx-variants fp16 fp32 --onnx-provider cuda \
    --output-json benchmarks/results/benchmark_batch1_results.json
```

Outputs a latency/throughput/memory table and `benchmarks/results/benchmark_results.json`.

### 2. Tactical analytics export (`analytics/analytics_exporter.py`)

Converts perspective-transformed pitch coordinates into coach-readable output:

```bash
# Reuse cached tracking + camera stubs (fast)
python analytics/analytics_exporter.py

# End-to-end detection on a clip
python analytics/analytics_exporter.py --no-stub --video input_videos/08fd33_4.mp4 --max-frames 300
```

Writes `outputs/analytics/match_summary.json` and
`outputs/analytics/player_metrics.csv` with per-player distance, top sprint
speed, average speed, team possession and a 12 x 32 pitch heatmap. `pipeline/main.py`
and `pipeline/main_gpu.py` now call the exporter automatically at the end of a run.

### 3. FastAPI microservice (`apps/app.py`)

```bash
uvicorn apps.app:app --host 0.0.0.0 --port 8000 --workers 1

curl -X POST "http://localhost:8000/analyze-clip?max_frames=300" \
     -F "file=@input_videos/08fd33_4.mp4"
```

The endpoint returns the `match_summary.json` payload. YOLO weights are loaded
lazily and cached for the process lifetime; ByteTrack state is reset per clip
and inference is serialised with a lock so one GPU is not oversubscribed.

### 4. Docker

```bash
docker build -t balltracker-api .
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" balltracker-api
```

The image uses CPU-only PyTorch to stay portable. Mount the trained `best.pt`
at runtime; model weights and sample videos are excluded by `.dockerignore`.

Opening `http://localhost:8000/` redirects to Swagger UI at `/docs`.

### 5. Interactive Streamlit dashboard

```bash
streamlit run apps/streamlit_app.py
```

The dashboard has three tabs:

1. **Match Analytics & Video** — instant pre-computed sample or a full live
   run on a custom clip, `max_frames` slider, cached-stub toggle, KPI cards,
   annotated H.264 video with download, K-Means team-colour plot, Team 1 vs
   Team 2 pitch heatmaps, and a filterable `player_metrics.csv` with download.
   Live runs execute all 8 pipeline stages and write per-run artefacts under
   `outputs/analytics/runs/`.
2. **Inference & Pipeline Benchmarks** — Batch-10 and Batch-1 benchmark
   tables, latency/FPS/VRAM bar charts, and pipeline quality cards for ball
   interpolation, team silhouette, and camera residual drift.
3. **FastAPI Microservice Playground** — live `/health` status, one-click
   `POST /analyze-clip`, raw JSON download, and an in-process exporter fallback
   when the API server is not running.

## Project Structure

```
BallTracker/
├── pipeline/
│   ├── main.py                 # Standard CPU pipeline
│   └── main_gpu.py             # GPU-optimized pipeline
├── apps/
│   ├── app.py                  # FastAPI /analyze-clip service
│   └── streamlit_app.py        # Interactive 3-tab tactical dashboard
├── analytics/
│   └── analytics_exporter.py   # match_summary.json + player_metrics.csv
├── benchmarks/
│   ├── benchmark.py            # FP32 / FP16 / ONNX inference benchmark
│   ├── benchmark_speed.py      # End-to-end FPS benchmark
│   └── results/                # Benchmark JSON reports
├── evaluation/
│   └── eval_*.py               # Tracking, team, camera, mAP evaluations
├── scripts/
│   └── yolo_inference.py       # Minimal YOLO inference helper
├── camera_movement_estimator/
├── player_ball_assigner/
├── speed_and_distance_estimator/
├── team_assigner/
├── trackers/
├── utils/
├── view_transformer/
├── models/                     # best.pt (git-ignored)
├── input_videos/               # Sample video
├── output_videos/              # Generated annotated videos
├── outputs/
│   ├── analytics/              # match_summary.json + player_metrics.csv
│   │   └── runs/               # Live dashboard runs (git-ignored)
│   └── eval_summary.json       # Ball / team / camera quality metrics
├── stubs/                      # Cached tracking results (git-ignored)
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## GPU Configuration Checklist

| Setting         | Command/Code                                     |
| --------------- | ------------------------------------------------ |
| Force CUDA      | `export CUDA_VISIBLE_DEVICES=0`                |
| FP16 inference  | `model.predict(..., half=True, device=0)`      |
| Batch size      | `batch_size=32` (adjust for VRAM)              |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = True`        |
| TF32 (Ampere+)  | `torch.backends.cuda.matmul.allow_tf32 = True` |
