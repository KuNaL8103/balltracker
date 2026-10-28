#!/usr/bin/env python
"""YOLOv5 inference benchmark: PyTorch FP32 vs FP16 vs ONNX Runtime.

The benchmark measures *model forward latency only* on the same pre-processed
frames so that every backend is compared fairly. Video decoding, letterboxing
and post-processing are intentionally excluded.

Example
-------
    python benchmarks/benchmark.py --video input_videos/08fd33_4.mp4 \
        --model models/best.pt --frames 100 --runs 3

Outputs a summary table to stdout and a machine-readable JSON report under
``benchmarks/results/``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import psutil
import torch

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("ultralytics is required: pip install ultralytics") from exc

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def read_frames(video_path: str, max_frames: int) -> tuple[List[np.ndarray], float]:
    """Read at most ``max_frames`` frames and return them with the source FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: List[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise SystemExit(f"No frames decoded from: {video_path}")
    return frames, fps


def letterbox_uint8(image: np.ndarray, size: int = 640) -> np.ndarray:
    """Resize + pad to a square RGB image and return uint8 HWC.

    Keeping the pre-processed frames as uint8 avoids holding ~0.5 GB of float32
    tensors in RAM for 100 frames. Conversion to float happens outside the
    timed region.
    """
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (size - new_w) / 2
    pad_h = (size - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


def to_torch_batch(frame_u8: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frame_u8.transpose(2, 0, 1)))
    return tensor.unsqueeze(0).to(device=device, dtype=dtype).div_(255.0)


def to_numpy_batch(frame_u8: np.ndarray, half: bool = False) -> np.ndarray:
    dtype = np.float16 if half else np.float32
    array = np.ascontiguousarray(frame_u8.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return array.astype(dtype)[None, ...]


def make_batches(frames_u8: List[np.ndarray], batch_size: int) -> List[List[np.ndarray]]:
    """Split frames into fixed-size batches, padding the tail by repetition.

    Padding keeps a fixed-batch ONNX model happy and makes every measured batch
    directly comparable. Per-frame latency is normalised by the batch size.
    """
    batches: List[List[np.ndarray]] = []
    for index in range(0, len(frames_u8), batch_size):
        batch = frames_u8[index:index + batch_size]
        if len(batch) < batch_size:
            batch = batch + [batch[-1]] * (batch_size - len(batch))
        batches.append(batch)
    return batches


def prepare_onnx_cuda() -> None:
    """Make PyTorch's bundled CUDA/cuDNN DLLs visible to ONNX Runtime on Windows."""
    if os.name != "nt" or not torch.cuda.is_available():
        return
    torch_lib = Path(torch.__file__).parent / "lib"
    if not torch_lib.is_dir():
        return
    try:
        os.add_dll_directory(str(torch_lib))
    except (AttributeError, OSError):
        pass
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def power_snapshot() -> dict:
    """Capture the power state so a benchmark run can be checked for drift."""
    state = {"ac_power": None, "sm_clock_mhz": None}
    try:
        battery = psutil.sensors_battery()
        if battery is not None:
            state["ac_power"] = bool(battery.power_plugged)
    except (AttributeError, NotImplementedError):
        pass
    smi = nvidia_smi_info()
    state["sm_clock_mhz"] = smi.get("gpu_sm_clock_mhz")
    return state


def is_cuda(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def sync(device: torch.device) -> None:
    if is_cuda(device):
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    backend: str
    precision: str
    device: str
    provider: Optional[str]
    batch_size: int
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p95: float
    throughput_fps: float
    gpu_peak_mb: Optional[float]
    cpu_rss_delta_mb: float
    model_load_s: float
    speedup_vs_baseline: float = 1.0
    ac_power_start: Optional[bool] = None
    ac_power_end: Optional[bool] = None
    sm_clock_start_mhz: Optional[float] = None
    sm_clock_end_mhz: Optional[float] = None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def benchmark_torch(
    model_cpu: torch.nn.Module,
    frames_u8: List[np.ndarray],
    device: torch.device,
    half: bool,
    warmup: int,
    runs: int,
    batch_size: int = 1,
) -> BenchmarkResult:
    precision = "FP16" if half else "FP32"
    dtype = torch.float16 if half else torch.float32
    batches = make_batches(frames_u8, batch_size)

    gc.collect()
    if is_cuda(device):
        torch.cuda.empty_cache()
    rss_before = rss_mb()

    load_start = time.perf_counter()
    model = copy.deepcopy(model_cpu).to(device)
    model = model.half() if half else model.float()
    model.eval()
    sync(device)
    load_s = time.perf_counter() - load_start

    # Warm up kernels / cuDNN autotuning before measuring.
    with torch.inference_mode():
        for batch in batches[:min(warmup, len(batches))]:
            inputs = torch.cat([to_torch_batch(frame, device, dtype) for frame in batch], dim=0)
            model(inputs)
    sync(device)
    power_start = power_snapshot()

    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            for batch in batches:
                inputs = torch.cat([to_torch_batch(frame, device, dtype) for frame in batch], dim=0)
                start = time.perf_counter()
                model(inputs)
                sync(device)
                batch_latency_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(batch_latency_ms / batch_size)

    power_end = power_snapshot()
    gpu_peak = None
    if is_cuda(device):
        gpu_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    cpu_delta = max(0.0, rss_mb() - rss_before)

    del model
    gc.collect()
    if is_cuda(device):
        torch.cuda.empty_cache()

    return BenchmarkResult(
        backend="PyTorch",
        precision=precision,
        device=str(device),
        provider=None,
        batch_size=batch_size,
        latency_ms_mean=statistics.fmean(latencies_ms),
        latency_ms_std=statistics.pstdev(latencies_ms),
        latency_ms_p95=float(np.percentile(latencies_ms, 95)),
        throughput_fps=1000.0 / statistics.fmean(latencies_ms),
        gpu_peak_mb=gpu_peak,
        cpu_rss_delta_mb=cpu_delta,
        model_load_s=load_s,
        ac_power_start=power_start["ac_power"],
        ac_power_end=power_end["ac_power"],
        sm_clock_start_mhz=power_start["sm_clock_mhz"],
        sm_clock_end_mhz=power_end["sm_clock_mhz"],
    )


def export_onnx(
    model_path: str,
    imgsz: int,
    onnx_path: Path,
    re_export: bool,
    half: bool,
    batch_size: int = 1,
) -> Path:
    if onnx_path.exists() and not re_export:
        print(f"[onnx] Reusing cached model: {onnx_path}")
        return onnx_path

    if half and not torch.cuda.is_available():
        raise SystemExit("FP16 ONNX export requires CUDA (Ultralytics limitation)")

    print(f"[onnx] Exporting {model_path} -> {onnx_path} (half={half}, batch={batch_size}) ...")
    export_model = YOLO(model_path)
    exported = export_model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        dynamic=False,
        simplify=False,
        batch=batch_size,
        device=0 if half else "cpu",
    )
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve() and exported_path.exists():
        exported_path.replace(onnx_path)
    print(f"[onnx] Export complete ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def benchmark_onnx(
    onnx_path: Path,
    frames_u8: List[np.ndarray],
    warmup: int,
    runs: int,
    provider: str,
    half: bool,
    batch_size: int = 1,
) -> BenchmarkResult:
    if ort is None:
        raise SystemExit("onnxruntime is required: pip install onnxruntime")

    batches = make_batches(frames_u8, batch_size)
    provider_map = {
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    if provider not in provider_map:
        raise ValueError(f"Unknown ONNX provider: {provider}")

    prepare_onnx_cuda()
    requested = [provider_map[provider]]
    if requested[0] != "CPUExecutionProvider":
        requested.append("CPUExecutionProvider")

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, (os.cpu_count() or 1))
    options.log_severity_level = 3  # suppress provider fallback warnings

    gc.collect()
    rss_before = rss_mb()
    load_start = time.perf_counter()
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=requested)
    load_s = time.perf_counter() - load_start
    actual_provider = session.get_providers()[0]

    if actual_provider != requested[0]:
        print(f"[onnx] WARNING: requested {requested[0]}, running on {actual_provider}")

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    expected_shape = input_meta.shape
    if len(expected_shape) == 4:
        if isinstance(expected_shape[0], int) and expected_shape[0] != batch_size:
            raise SystemExit(
                f"ONNX model was exported for batch {expected_shape[0]}, "
                f"but --batch-size is {batch_size}. Re-run with --re-export."
            )
        if isinstance(expected_shape[2], int) and isinstance(expected_shape[3], int):
            expected_h, expected_w = expected_shape[2], expected_shape[3]
            if (expected_h, expected_w) != frames_u8[0].shape[:2]:
                raise SystemExit(
                    f"ONNX model expects {expected_w}x{expected_h}, "
                    f"pre-processed frames are {frames_u8[0].shape[1]}x{frames_u8[0].shape[0]}"
                )

    def run_batch(batch: List[np.ndarray]) -> None:
        inputs = np.concatenate([to_numpy_batch(frame, half) for frame in batch], axis=0)
        session.run(None, {input_name: inputs})

    for batch in batches[:min(warmup, len(batches))]:
        run_batch(batch)
    power_start = power_snapshot()

    latencies_ms: List[float] = []
    for _ in range(runs):
        for batch in batches:
            inputs = np.concatenate([to_numpy_batch(frame, half) for frame in batch], axis=0)
            start = time.perf_counter()
            session.run(None, {input_name: inputs})
            batch_latency_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(batch_latency_ms / batch_size)

    power_end = power_snapshot()
    cpu_delta = max(0.0, rss_mb() - rss_before)
    del session
    gc.collect()

    device_label = {
        "CUDAExecutionProvider": "CUDA",
        "CPUExecutionProvider": "CPU",
    }.get(actual_provider, actual_provider)

    return BenchmarkResult(
        backend="ONNX Runtime",
        precision="FP16" if half else "FP32",
        device=device_label,
        provider=actual_provider,
        batch_size=batch_size,
        latency_ms_mean=statistics.fmean(latencies_ms),
        latency_ms_std=statistics.pstdev(latencies_ms),
        latency_ms_p95=float(np.percentile(latencies_ms, 95)),
        throughput_fps=1000.0 / statistics.fmean(latencies_ms),
        gpu_peak_mb=None,
        cpu_rss_delta_mb=cpu_delta,
        model_load_s=load_s,
        ac_power_start=power_start["ac_power"],
        ac_power_end=power_end["ac_power"],
        sm_clock_start_mhz=power_start["sm_clock_mhz"],
        sm_clock_end_mhz=power_end["sm_clock_mhz"],
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: List[BenchmarkResult]) -> None:
    headers = [
        "Backend", "Precision", "Device", "Batch", "Latency (ms/frame)",
        "Throughput (FPS)", "GPU peak (MB)", "CPU RSS delta (MB)",
        "Load (s)", "Speedup",
    ]
    rows = []
    for result in results:
        rows.append([
            result.backend,
            result.precision,
            result.device,
            str(result.batch_size),
            f"{result.latency_ms_mean:.2f} +/- {result.latency_ms_std:.2f}",
            f"{result.throughput_fps:.1f}",
            f"{result.gpu_peak_mb:.1f}" if result.gpu_peak_mb is not None else "n/a",
            f"{result.cpu_rss_delta_mb:.1f}",
            f"{result.model_load_s:.2f}",
            f"{result.speedup_vs_baseline:.2f}x",
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    separator = "+".join("-" * (width + 2) for width in widths)
    print(separator)
    print("| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |")
    print(separator)


def nvidia_smi_info() -> dict:
    """Best-effort power/clock telemetry so benchmark numbers are interpretable."""
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,power.limit,clocks.sm,clocks.max.sm,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}

    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != 5:
        return {}

    keys = [
        "gpu_power_draw_w",
        "gpu_power_limit_w",
        "gpu_sm_clock_mhz",
        "gpu_max_sm_clock_mhz",
        "gpu_temperature_c",
    ]
    info = {}
    for key, value in zip(keys, values):
        try:
            info[key] = float(value)
        except ValueError:
            info[key] = None  # e.g. "[N/A]" when the driver hides the limit
    return info


def environment_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mb": (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            if torch.cuda.is_available() else None
        ),
        "onnxruntime": ort.__version__ if ort is not None else None,
        "onnxruntime_providers": ort.get_available_providers() if ort is not None else None,
    }
    info.update(nvidia_smi_info())
    try:
        battery = psutil.sensors_battery()
        if battery is not None:
            info["ac_power"] = bool(battery.power_plugged)
            info["battery_percent"] = round(float(battery.percent), 1)
    except (AttributeError, NotImplementedError):
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv5 inference backends")
    parser.add_argument("--video", default="input_videos/08fd33_4.mp4", help="Source video")
    parser.add_argument("--model", default="models/best.pt", help="YOLOv5 .pt weights")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Repeated passes over the frames")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations (batches) per backend")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Inference batch size (1 = single-request latency; 8 = throughput)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--onnx-path", default=None, help="Base path for the ONNX export(s)")
    parser.add_argument("--onnx-variants", nargs="+", choices=["fp32", "fp16"], default=["fp32"],
                        help="ONNX precision variants to benchmark")
    parser.add_argument("--onnx-provider", choices=["auto", "cpu", "cuda", "all"], default="auto",
                        help="ONNX Runtime execution provider ('all' = CUDA + CPU)")
    parser.add_argument("--re-export", action="store_true", help="Force a new ONNX export")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip the ONNX backend")
    parser.add_argument(
        "--require-ac",
        action="store_true",
        help="Refuse to save results if the laptop switches to battery power",
    )
    parser.add_argument(
        "--output-json",
        default="benchmarks/results/benchmark_results.json",
        help="JSON report path",
    )
    args = parser.parse_args()

    if not Path(args.model).is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.frames < 1 or args.runs < 1:
        raise SystemExit("--frames and --runs must be >= 1")

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    torch.backends.cudnn.benchmark = True
    if is_cuda(device):
        torch.backends.cuda.matmul.allow_tf32 = True

    # Collect this before touching CUDA/ONNX Runtime: some export paths can
    # leave the CUDA context in a state where device queries fail.
    env_info = environment_info()

    print("=" * 78)
    print("BallTracker - YOLOv5 inference benchmark")
    print("=" * 78)
    print(f"Model      : {args.model}")
    print(f"Video      : {args.video}")
    print(f"Device     : {device}")
    print(f"Frames     : {args.frames} x {args.runs} run(s), batch {args.batch_size}, "
          f"warm-up {args.warmup} batch(es)")
    print("Timing     : model forward only (decode/pre-process/post-process excluded)")
    print("=" * 78)

    frames, source_fps = read_frames(args.video, args.frames)
    print(f"Loaded {len(frames)} frames at {source_fps:.1f} FPS "
          f"({frames[0].shape[1]}x{frames[0].shape[0]})")

    print("Pre-processing frames (letterbox -> 640x640 RGB uint8) ...")
    frames_u8 = [letterbox_uint8(frame, args.imgsz) for frame in frames]

    print("Loading PyTorch model ...")
    yolo = YOLO(args.model)
    model_cpu = copy.deepcopy(yolo.model).float().eval()
    del yolo
    gc.collect()

    results: List[BenchmarkResult] = []

    print("\n[pytorch] FP32 ...")
    results.append(benchmark_torch(model_cpu, frames_u8, device, half=False,
                                   warmup=args.warmup, runs=args.runs,
                                   batch_size=args.batch_size))

    if is_cuda(device):
        print("[pytorch] FP16 ...")
        results.append(benchmark_torch(model_cpu, frames_u8, device, half=True,
                                       warmup=args.warmup, runs=args.runs,
                                       batch_size=args.batch_size))
    else:
        print("[pytorch] FP16 skipped (CUDA required for useful FP16 throughput)")

    del model_cpu
    gc.collect()

    if not args.skip_onnx and ort is not None:
        available = env_info.get("onnxruntime_providers") or []
        cuda_ep = "CUDAExecutionProvider" in available
        if args.onnx_provider == "all":
            onnx_providers = (["cuda"] if cuda_ep else []) + ["cpu"]
        elif args.onnx_provider == "auto":
            onnx_providers = ["cuda"] if cuda_ep else ["cpu"]
        else:
            onnx_providers = [args.onnx_provider]

        if "cuda" in onnx_providers and not cuda_ep:
            print("[onnx] CUDA provider not available; falling back to CPU")
            onnx_providers = [provider for provider in onnx_providers if provider != "cuda"] or ["cpu"]

        base_onnx = (
            Path(args.onnx_path) if args.onnx_path else Path(args.model).with_suffix(".onnx")
        )
        multi_variant = len(args.onnx_variants) > 1

        for variant in args.onnx_variants:
            half = variant == "fp16"
            if half and not torch.cuda.is_available():
                print(f"[onnx] Skipping {variant.upper()} (requires CUDA for export)")
                continue

            name_parts = []
            if half or multi_variant:
                name_parts.append(variant)
            if args.batch_size != 1:
                name_parts.append(f"bs{args.batch_size}")
            if name_parts:
                onnx_path = base_onnx.with_name(
                    f"{base_onnx.stem}_{'_'.join(name_parts)}{base_onnx.suffix}"
                )
            else:
                onnx_path = base_onnx

            export_onnx(
                args.model, args.imgsz, onnx_path, args.re_export, half,
                batch_size=args.batch_size,
            )

            for provider in onnx_providers:
                if half and provider == "cpu":
                    print("[onnx] Skipping FP16 on CPUExecutionProvider")
                    continue
                print(f"[onnx] {variant.upper()} on {provider.upper()} ...")
                results.append(
                    benchmark_onnx(
                        onnx_path, frames_u8, args.warmup, args.runs,
                        provider=provider, half=half, batch_size=args.batch_size,
                    )
                )
    elif args.skip_onnx:
        print("[onnx] Skipped (--skip-onnx)")
    else:
        print("[onnx] Skipped (onnxruntime is not installed)")

    baseline = results[0].throughput_fps
    for result in results:
        result.speedup_vs_baseline = result.throughput_fps / baseline
        if (
            result.ac_power_start is not None
            and result.ac_power_start != result.ac_power_end
        ):
            print(
                f"[power] WARNING: {result.backend} {result.precision} "
                f"switched AC power state during measurement "
                f"({result.ac_power_start} -> {result.ac_power_end})"
            )
        if (
            result.sm_clock_start_mhz
            and result.sm_clock_end_mhz
            and max(result.sm_clock_start_mhz, result.sm_clock_end_mhz)
            / max(1.0, min(result.sm_clock_start_mhz, result.sm_clock_end_mhz))
            > 1.3
        ):
            print(
                f"[power] WARNING: {result.backend} {result.precision} "
                f"SM clock drifted {result.sm_clock_start_mhz:.0f} -> "
                f"{result.sm_clock_end_mhz:.0f} MHz during measurement"
            )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print_table(results)

    if args.require_ac:
        unstable = [
            result
            for result in results
            if result.ac_power_start is False
            or result.ac_power_end is False
            or result.ac_power_start != result.ac_power_end
        ]
        if unstable:
            print(
                "\n[power] ERROR: --require-ac was set, but at least one backend "
                "was not on stable AC power. Results were NOT written."
            )
            for result in unstable:
                print(
                    f"  - {result.backend} {result.precision}: "
                    f"{result.ac_power_start} -> {result.ac_power_end}"
                )
            sys.exit(2)

    payload = {
        "environment": env_info,
        "config": {
            "model": str(Path(args.model).resolve()),
            "video": str(Path(args.video).resolve()),
            "source_fps": source_fps,
            "frames": len(frames),
            "runs": args.runs,
            "warmup": args.warmup,
            "batch_size": args.batch_size,
            "timed_slots": len(make_batches(frames_u8, args.batch_size)) * args.batch_size,
            "imgsz": args.imgsz,
            "onnx_variants": args.onnx_variants,
            "onnx_provider": args.onnx_provider,
            "timing_scope": "model forward only",
        },
        "results": [asdict(result) for result in results],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
