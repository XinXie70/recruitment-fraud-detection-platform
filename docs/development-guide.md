# Development Guide

This guide covers the local development workflow for the website backend, standalone
model backend, frontend, database, and tests.

## Prerequisites

- Python 3.12+
- Node.js 22.22+
- Docker (optional, for full-stack runs)
- Git LFS (required for model weights)

## Service layout

- `backend.main:app` is the website backend. It owns authentication, analysis workflows,
  history, administration, and database access.
- `src.api.main:app` is the standalone model backend. It exposes LR, BERT, and ensemble
  prediction endpoints.
- `frontend/` is the React web application and sends `/api` requests to the website backend.

The website backend can load models locally or call a standalone model backend through
`MODEL_SERVER_URL`.

## Python setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements-dev.txt
pip install -e model
```

This installs the sprint3 LR + BERT FP-gate adapters under `model/final_model_pipelines`.
Weights are resolved from `model_algorithm/sprint3/`.

Create the local environment file and replace its example secret:

```bash
cp backend/env.example .env
```

The `.env` file is ignored by Git and must not contain committed credentials.

## Recommended full-stack startup

The simplest path starts PostgreSQL, applies migrations, and launches the website backend
and frontend together:

```bash
docker compose up --build
```

The frontend is available at `http://localhost:5190`, and the website backend API docs are
available at `http://127.0.0.1:8000/docs`.

## Manual startup

### 1. PostgreSQL

Start PostgreSQL and ensure `DATABASE_URL` in `.env` points to it.
Apply migrations before starting the website backend:

```bash
python -m alembic -c backend/alembic.ini upgrade head
```

### 2. Model backend (optional)

Skip this step when the website backend loads models locally. To use a separate model
process, start it on port 8001:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Then set `MODEL_SERVER_URL=http://127.0.0.1:8001` in `.env`.

### 3. Website backend

Start the application-facing API:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Its API docs will be available at:

```text
http://127.0.0.1:8000/docs
```

### 4. Frontend

```bash
cd frontend
npm ci
npm run dev
```

The frontend will be served from the Vite development server at:

```text
http://localhost:5190
```

## Running tests

### Backend

```bash
pytest -q
```

### Frontend

```bash
cd frontend
npm run lint
npm test
npm run test:coverage
npm run test:e2e
npm run build
```

## Environment variables

The backend reads configuration from environment variables. A sample file is available at:

```text
backend/env.example
```

If a remote model server is used, set the following variable before starting the website
backend:

```bash
export MODEL_SERVER_URL=http://127.0.0.1:8001
```

## Troubleshooting

- If the backend cannot start, verify that the required environment variables are present.
- If the model service is unavailable, confirm that the remote endpoint is reachable and the `MODEL_SERVER_URL` value is correct.
- If the frontend cannot run, ensure the Node.js version satisfies the engine requirement in the frontend package configuration.
