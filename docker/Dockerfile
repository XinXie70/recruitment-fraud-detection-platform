# Build from the repository root:
#   docker build -f docker/Dockerfile -t model-service:latest .
#
# Or start with Compose:
#   docker compose -f docker/docker-compose.yml up --build -d

FROM python:3.12-slim-bookworm

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

# Install CPU PyTorch first, then the rest of model_service/requirements.txt.
COPY model_service/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 \
    && grep -Ev '^(torch|#)' /tmp/requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install -r /tmp/requirements-no-torch.txt \
    && rm /tmp/requirements.txt /tmp/requirements-no-torch.txt

COPY model_service/ /app/model_service/
COPY docker/entrypoint.sh /app/model_service/entrypoint.sh

WORKDIR /app/model_service

RUN sed -i 's/\r$//' entrypoint.sh \
    && chmod +x entrypoint.sh \
    && chown -R app:app /app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

USER app

ENTRYPOINT ["./entrypoint.sh"]
