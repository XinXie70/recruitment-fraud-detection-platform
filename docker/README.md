# Fraud Detection API — Docker

This directory contains the container configuration for the production FP-gate
model inference API in `../model_service/`. It is separate from the root `compose.yaml`,
which runs the FastAPI web application backend, PostgreSQL, and the React frontend.

## Directory contents

```text
docker/
  Dockerfile                 # CPU image (recommended for cloud deployments)
  Dockerfile.gpu             # Optional GPU image
  docker-compose.yml
  entrypoint.sh
  requirements-docker.txt
  README.md
```

The build context is the repository root, so the root `.dockerignore` applies.

## Quick start

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Alternatively, build and run the CPU image directly:

```bash
docker build -f docker/Dockerfile -t fraud-detection-api:latest .
docker run -d --name fraud-detection-api -p 127.0.0.1:5000:5000 \
  -e MODEL_API_KEY=replace-me \
  --restart unless-stopped fraud-detection-api:latest
```

- API: `http://127.0.0.1:5000`
- Swagger UI: `http://127.0.0.1:5000/apidocs/`
- Health check: `GET /health`

The first startup preloads LR and BERT and can take one or two minutes.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Container listening port |
| `HOST` | `0.0.0.0` | Bind address inside the container (required for Docker networking) |
| `MODEL_API_KEY` | _(empty)_ | Shared secret for `/predict/*` (`X-API-Key` / Bearer). Leave empty only for local debugging |
| `RATE_LIMIT_PREDICT` | `30/minute` | Per-IP rate limit for prediction endpoints |
| `MAX_CONTENT_LENGTH` | `1048576` | Max JSON request body size in bytes |
| `MAX_TEXT_CHARS` | `50000` | Max characters per resolved advertisement text |
| `MAX_BATCH_TOTAL_CHARS` | `500000` | Max total characters across a batch request |
| `ALLOW_CPU` | `1` | Allow CPU inference when no GPU is available |
| `GUNICORN_WORKERS` | `1` | Recommended worker count; keep at one to avoid duplicate model memory |
| `GUNICORN_THREADS` | `4` | Concurrent Gunicorn threads |
| `GUNICORN_TIMEOUT` | `180` | Request timeout in seconds |

Compose publishes the API on `127.0.0.1` only. For CPU deployment, use at least 2 vCPUs and 4 GiB of memory.

## Cloud deployment

```bash
docker tag fraud-detection-api:latest <registry>/fraud-detection-api:latest
docker push <registry>/fraud-detection-api:latest
```

Deploy the image to a container platform such as ECS, ACI, Cloud Run, or
Container Apps. Expose port `5000` (or the platform-provided `PORT`), configure
`GET /health` as the health check, and allocate at least 4 GiB of memory.

## Optional GPU image

```bash
docker build -f docker/Dockerfile.gpu -t fraud-detection-api:gpu .
docker run --gpus all -d -p 5000:5000 -e ALLOW_CPU=0 fraud-detection-api:gpu
```
