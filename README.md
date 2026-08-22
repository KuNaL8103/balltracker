# ⚽ BallTracker Project

## Introduction

This project aims to **detect and track football players, referees, and the ball** in match videos using **YOLO**, one of the leading object detection models. The system enhances detection accuracy through training and assigns players to teams based on **t-shirt color clustering via KMeans**.

Key features include:

- Team identification using **KMeans clustering** on pixel data.
- **Ball possession analytics** by tracking which team controls the ball.
- **Optical Flow** to measure **camera movement** across frames.
- **Perspective transformation** to calculate movement in **real-world units (meters)**.
- **Speed and distance tracking** for individual players.

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
python main.py
```
Output: `output_videos/output_video.avi`

### GPU-Accelerated Version (Recommended)
```bash
python main_gpu.py
```
Output: `output_videos/output_video_gpu.avi`

**GPU optimizations in `main_gpu.py`:**
- CUDA model initialization (`model.to('cuda')`)
- FP16 half-precision inference (`model.half()`)
- Batch size 32 (vs 20 on CPU)
- cuDNN benchmark + TF32 enabled
- ~74 FPS tracking on RTX 4050 (vs ~10 FPS on CPU)

---

## Metric Validation Scripts

Run evaluation scripts to validate tracking quality:

| Script | Metric | Description |
|--------|--------|-------------|
| `eval_map.py` | mAP@0.5 | Detection accuracy (requires Roboflow val dataset) |
| `eval_tracking.py` | MOTA / ID Switches | Tracking quality, ID switches |
| `eval_team_assignment.py` | Silhouette Score | Team color separation quality |
| `eval_camera_motion.py` | Camera Drift | Raw & residual drift after homography |
| `benchmark_speed.py` | FPS | Processing speed benchmark |
| `eval_ball_interpolation.py` | Recovery Rate | Ball position interpolation recovery |
| `eval_output.py` | Frame Drop Rate | Output frame completeness |

### Run all evaluations:
```bash
# Tracking quality
python eval_tracking.py --frames 500

# Team assignment (silhouette >0.5 = good)
python eval_team_assignment.py --frames 100

# Camera motion drift
python eval_camera_motion.py --frames 500

# Ball interpolation recovery
python eval_ball_interpolation.py

# Output frame verification
python eval_output.py

# FPS benchmark
python benchmark_speed.py --frames 100 --runs 3

# mAP@0.5 (requires validation dataset)
python eval_map.py --data path/to/data.yaml
```

### Expected Results (on sample video):
- **mAP@0.5**: ~0.82 (with Roboflow validation set)
- **ID Switches**: ~19 (39 unique tracks, 100 frames)
- **Silhouette Score**: 0.77 ✓ EXCELLENT
- **Residual Camera Drift**: 0.14 m/frame ~ GOOD
- **Ball Recovery**: 100% ✓ EXCELLENT
- **Frame Drop Rate**: 0.00% ✓ PERFECT
- **Tracking FPS**: ~74 FPS (GPU, warm)

---

## Project Structure

```
BallTracker/
├── main.py                    # Standard CPU pipeline
├── main_gpu.py               # GPU-optimized pipeline
├── requirements.txt          # Dependencies
├── models/
│   └── best.pt              # YOLO model
├── input_videos/
│   └── 08fd33_4.mp4         # Sample video
├── output_videos/           # Generated outputs
├── stubs/                   # Cached tracking results
├── trackers/
│   └── tracker.py           # YOLO + ByteTrack (GPU optimized)
├── team_assigner/           # KMeans team clustering
├── camera_movement_estimator/
├── view_transformer/        # Homography transformation
├── speed_and_distance_estimator/
├── player_ball_assigner/    # Ball possession
├── utils/                   # Video & bbox utilities
└── eval_*.py                # Evaluation scripts
```

---

## GPU Configuration Checklist

| Setting | Command/Code |
|---------|--------------|
| Force CUDA | `export CUDA_VISIBLE_DEVICES=0` |
| FP16 inference | `model.predict(..., half=True, device=0)` |
| Batch size | `batch_size=32` (adjust for VRAM) |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = True` |
| TF32 (Ampere+) | `torch.backends.cuda.matmul.allow_tf32 = True` |
