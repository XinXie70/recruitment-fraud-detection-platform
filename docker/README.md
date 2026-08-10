# model_service — Docker

This directory packages the FP-gate inference API from [`../model_service/`](../model_service/).
The application backend (`backend/`) calls this service through `MODEL_SERVER_URL`; it does
not load model weights locally.

## What gets containerised

```text
model_service/
  app.py                 # Flask API (/health, /predict/*)
  wsgi.py                # Gunicorn preload entrypoint
  settings.py            # Paths + runtime config
  services/              # LR, BERT, ensemble, risk logic
  models/                # Frozen weights shipped inside the image
    bert/best/
    lr/
    ensemble/
    risk/
```

Training code under `model_algorithm/` is **not** copied into the image.

## Directory contents

```text
docker/
  Dockerfile             # CPU image (recommended)
  Dockerfile.gpu         # Optional GPU image
  docker-compose.yml
  entrypoint.sh
  README.md
```

Build context is the repository root; see [`.dockerignore`](../.dockerignore).

## Quick start

### Full stack (model + app)

From the repository root, one command starts PostgreSQL, migrations, `model_service`,
FastAPI, and the frontend:

```bash
docker compose up --build -d
```

- Frontend: `http://localhost:5190`
- Backend: `http://localhost:8000`
- Model Swagger: `http://127.0.0.1:5000/apidocs/`

The backend reaches the model container at `http://model-service:5000` on the
Compose network. Set `MODEL_API_KEY` in your shell or a `.env` file if you want
prediction auth enabled; the backend receives the same value as `MODEL_SERVER_API_KEY`.

### Model service only

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Or build and run manually:

```bash
docker build -f docker/Dockerfile -t model-service:latest .
docker run -d --name model-service -p 127.0.0.1:5000:5000 \
  -e MODEL_API_KEY=replace-me \
  --restart unless-stopped model-service:latest
```

Endpoints:

- API: `http://127.0.0.1:5000`
- Swagger UI: `http://127.0.0.1:5000/apidocs/`
- Health check: `GET /health`

The first startup preloads LR and BERT and can take one or two minutes.

## Backend integration

When the root [`compose.yaml`](../compose.yaml) stack is used, the backend already
points at `http://model-service:5000` on the Compose network.

For a standalone model container consumed by a host-run backend:

```env
MODEL_SERVER_URL=http://127.0.0.1:5000
MODEL_SERVER_API_KEY=<same value as MODEL_API_KEY when auth is enabled>
```

When the app stack runs in Docker but the model runs on the host:

```bash
export MODEL_SERVER_URL=http://host.docker.internal:5000   # macOS / Windows Docker Desktop
docker compose up --build -d database migrate backend frontend
```

Supported model-service endpoints used by the backend:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health |
| `POST` | `/predict/all` | LR, BERT, FP-gate, and risk result |
| `POST` | `/predict/batch` | Batch scoring for SHAP / XAI |

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `5000` | Container listening port |
| `HOST` | `0.0.0.0` | Bind address inside the container |
| `MODEL_FILES_ROOT` | `/app/model_service/models` | Override model artifact directory |
| `APP_ENV` | `development` | Runtime mode; production requires `MODEL_API_KEY` |
| `MODEL_API_KEY` | _(empty)_ | Shared secret for `/predict/*` (`X-API-Key` / Bearer); mandatory in production |
| `MODEL_CORS_ORIGINS` | backend localhost origins | Comma-separated browser origins allowed to call the model API |
| `RATE_LIMIT_PREDICT` | `30/minute` | Per-IP rate limit for prediction endpoints |
| `MAX_CONTENT_LENGTH` | `1048576` | Max JSON request body size in bytes |
| `MAX_TEXT_CHARS` | `50000` | Max characters per resolved advertisement text |
| `MAX_BATCH_ITEMS` | `100` | Max items in a batch request |
| `MAX_BATCH_TOTAL_CHARS` | `500000` | Max total characters across a batch request |
| `ALLOW_CPU` | `1` | Allow CPU inference when no GPU is available |
| `GUNICORN_WORKERS` | `1` | Keep at one to avoid duplicate model memory |
| `GUNICORN_THREADS` | `4` | Concurrent Gunicorn threads |
| `GUNICORN_TIMEOUT` | `180` | Request timeout in seconds |

Compose publishes the API on `127.0.0.1` only. For CPU deployment, allocate at least
2 vCPUs and 4 GiB of memory.

## Cloud / VM deployment

```bash
docker tag model-service:latest <registry>/model-service:latest
docker push <registry>/model-service:latest
```

Deploy to ECS, ACI, Cloud Run, Container Apps, or a VM with Docker. Set
`APP_ENV=production` and provide a non-empty `MODEL_API_KEY` before deployment;
the service refuses to start without it. Prefer a private service endpoint and
restrict `MODEL_CORS_ORIGINS`. Configure `GET /health` as the health check and
allocate at least 4 GiB of memory for CPU inference.

## Optional GPU image

```bash
docker build -f docker/Dockerfile.gpu -t model-service:gpu .
docker run --gpus all -d -p 5000:5000 -e ALLOW_CPU=0 model-service:gpu
```

## Legacy image names

Older deployments may still reference `fraud-detection-api:latest`. Re-tag if needed:

```bash
docker tag model-service:latest fraud-detection-api:latest
```
