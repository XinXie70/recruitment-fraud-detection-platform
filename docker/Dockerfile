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

# --- Runtime layout (must match api_flask/settings.py relative paths) ---
COPY api_flask/ /app/api_flask/
COPY docker/entrypoint.sh /app/api_flask/entrypoint.sh

COPY lr_none_bigram_no_cv_paper_aligned_seed42/artifacts/ \
    /app/lr_none_bigram_no_cv_paper_aligned_seed42/artifacts/
COPY lr_none_bigram_no_cv_paper_aligned_seed42/results/config.json \
    /app/lr_none_bigram_no_cv_paper_aligned_seed42/results/config.json

COPY retrain_paper_aligned_seed42_maxlen512/weights/bert_paper_protocol_maxlen512/best/ \
    /app/retrain_paper_aligned_seed42_maxlen512/weights/bert_paper_protocol_maxlen512/best/
COPY retrain_paper_aligned_seed42_maxlen512/code/ \
    /app/retrain_paper_aligned_seed42_maxlen512/code/

COPY ensemble_bert_fp_gate_lr_none_bigram_maxlen512/results/config.json \
    /app/ensemble_bert_fp_gate_lr_none_bigram_maxlen512/results/config.json
COPY ensemble_bert_fp_gate_lr_none_bigram_maxlen512/risk_level/risk_boundary_config.json \
    /app/ensemble_bert_fp_gate_lr_none_bigram_maxlen512/risk_level/risk_boundary_config.json

WORKDIR /app/api_flask

RUN sed -i 's/\r$//' entrypoint.sh \
    && chmod +x entrypoint.sh \
    && mkdir -p /app/retrain_paper_aligned_seed42_maxlen512/results/logs \
    && chown -R app:app /app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

USER app

ENTRYPOINT ["./entrypoint.sh"]
