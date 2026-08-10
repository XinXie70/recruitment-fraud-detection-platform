# Build from the capstone project root:
#   docker build -f docker/Dockerfile -t fraud-detection-api .
#
# Or from this folder:
#   docker compose up --build

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5000 \
    HOST=0.0.0.0 \
    ALLOW_CPU=1 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=180 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && rm -rf /var/lib/apt/lists/*

# CPU PyTorch first (cloud-friendly; no NVIDIA runtime required), then API deps.
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 \
    && pip install -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

# --- Runtime layout (must match model_service/settings.py relative paths) ---
COPY model_service/ /app/model_service/
COPY docker/entrypoint.sh /app/model_service/entrypoint.sh

COPY model_algorithm/sprint3/LR/weight/ /app/model_algorithm/sprint3/LR/weight/
COPY model_algorithm/sprint3/LR/results/config.json /app/model_algorithm/sprint3/LR/results/config.json
COPY model_algorithm/sprint3/BERT/weight/best/ /app/model_algorithm/sprint3/BERT/weight/best/
COPY model_algorithm/sprint3/ensemble_BERT_FP/results/config.json /app/model_algorithm/sprint3/ensemble_BERT_FP/results/config.json
COPY model_algorithm/sprint3/risk_level/risk_boundary_config.json /app/model_algorithm/sprint3/risk_level/risk_boundary_config.json

WORKDIR /app/model_service

RUN sed -i 's/\r$//' entrypoint.sh \
    && chmod +x entrypoint.sh \
    && chown -R app:app /app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

USER app

ENTRYPOINT ["./entrypoint.sh"]
