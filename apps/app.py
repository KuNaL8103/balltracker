#!/usr/bin/env python
"""FastAPI inference service for short football clips.

Endpoints
---------
GET  /                -> 307 redirect to Swagger UI at /docs
GET  /health          -> service + model readiness
POST /analyze-clip    -> multipart video upload, returns match_summary.json

Run locally:
    uvicorn apps.app:app --host 0.0.0.0 --port 8000 --workers 1

The YOLO weights are loaded lazily and cached for the process lifetime. A lock
serialises inference so a single GPU is never oversubscribed; use one worker
per GPU. ByteTrack state is reset for every clip.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse

from analytics.analytics_exporter import build_match_summary, process_video

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("balltracker.api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
DEFAULT_MAX_FRAMES = int(os.getenv("MAX_FRAMES", "300"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_tracker = None
_tracker_lock = threading.Lock()

app = FastAPI(
    title="BallTracker Inference API",
    description=(
        "Upload a short football clip and receive structured tactical analytics "
        "(possession, distance, sprint speed and pitch heatmaps)."
    ),
    version="1.0.0",
)


def get_tracker():
    """Lazily create and cache the YOLO + ByteTrack tracker."""
    global _tracker
    if _tracker is None:
        from trackers.tracker import Tracker

        if not Path(MODEL_PATH).is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        logger.info("Loading YOLO model from %s", MODEL_PATH)
        _tracker = Tracker(MODEL_PATH)
        logger.info("Model ready (device=%s, half=%s)", _tracker.device, _tracker.use_half)
    return _tracker


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Open Swagger UI when someone visits the service root in a browser."""
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health")
def health():
    model_present = Path(MODEL_PATH).is_file()
    payload = {
        "status": "ok" if model_present else "degraded",
        "model_path": MODEL_PATH,
        "model_present": model_present,
        "model_loaded": _tracker is not None,
        "max_frames": DEFAULT_MAX_FRAMES,
        "max_upload_mb": MAX_UPLOAD_MB,
    }
    if not model_present:
        return JSONResponse(status_code=503, content=payload)
    return payload


def _analyze_blocking(path: str, max_frames: int, source_name: str) -> dict:
    with _tracker_lock:
        tracker = get_tracker()
        result = process_video(
            video_path=path,
            model_path=MODEL_PATH,
            max_frames=max_frames,
            use_stubs=False,
            tracker=tracker,
            log=lambda message: logger.info(message),
        )
        return build_match_summary(result, source_name=source_name)


@app.post("/analyze-clip")
async def analyze_clip(
    file: UploadFile = File(...),
    max_frames: Optional[int] = Query(
        default=None,
        ge=1,
        le=1500,
        description="Optional frame cap for the uploaded clip.",
    ),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    frame_cap = max_frames or DEFAULT_MAX_FRAMES
    source_name = file.filename or f"upload{suffix}"

    with tempfile.TemporaryDirectory(prefix="balltracker_") as temp_dir:
        upload_path = Path(temp_dir) / f"upload{suffix}"
        size = 0
        try:
            with upload_path.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB * 1024 * 1024:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds {MAX_UPLOAD_MB} MB limit",
                        )
                    handle.write(chunk)
        finally:
            await file.close()

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        try:
            return await run_in_threadpool(
                _analyze_blocking, str(upload_path), frame_cap, source_name
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - depends on media/model
            logger.exception("Clip analysis failed")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
