# Development Guide

## Prerequisites

- Python 3.11+
- Node.js 22.22+
- PostgreSQL
- Docker, optionally

## Service layout

- `model_service.app:app`: BERT-primary + LR false-positive-gate model API.
- `backend.main:app`: authenticated application API.
- `frontend/`: React client.

The application API always calls the model API. Local model loading is not
supported.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp backend/env.example .env
```

This installs the lightweight application API plus test and lint tools. It does
not install the model runtime: the backend calls the model API over HTTP and
never imports TensorFlow, PyTorch, Transformers, or XGBoost. Those dependencies
belong only to `model_service/requirements.txt`.

Ensure the new LR artifact and BERT checkpoint exist, then start the model API:

```bash
pip install -r model_service/requirements.txt
python model_service/app.py
```

Set `MODEL_SERVER_URL=http://127.0.0.1:5000` in `.env`, apply migrations, and
start FastAPI:

```bash
python -m alembic -c backend/alembic.ini upgrade head
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

## Verification

```bash
pytest -q
cd frontend
npm run lint
npm test
npm run build
```

If analysis returns `503`, check the model API `/health` endpoint, verify
`MODEL_SERVER_URL`, and confirm `/predict/all` returns the documented FP-gate
contract.
