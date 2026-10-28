FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MODEL_PATH=/app/models/best.pt \
    MAX_FRAMES=300 \
    MAX_UPLOAD_MB=100

WORKDIR /app

# OpenCV runtime libraries for slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first so ultralytics does not pull CUDA wheels.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY apps ./apps
COPY analytics ./analytics
COPY camera_movement_estimator ./camera_movement_estimator
COPY player_ball_assigner ./player_ball_assigner
COPY speed_and_distance_estimator ./speed_and_distance_estimator
COPY team_assigner ./team_assigner
COPY trackers ./trackers
COPY utils ./utils
COPY view_transformer ./view_transformer

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Mount your trained weights at runtime, e.g.
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" balltracker-api
# The model is intentionally not baked into the image (173 MB, git-ignored).
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "apps.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
