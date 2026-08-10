# Fake Job Advertisement Detection Platform

The supported product consists of a React frontend, an authenticated FastAPI
application backend, PostgreSQL, and one separately deployable model service.
The model service runs the paper-aligned BERT-primary + Logistic Regression
false-positive-gate model. There is no local or weighted-ensemble fallback.

## Supported services

| Service | Entry point | Responsibility |
| --- | --- | --- |
| Frontend | `frontend/` | Authentication, analysis, reports, history, education, and administration UI |
| Application API | `backend.main:app` | Users, JWT, validation, analysis orchestration, XAI, guidance, and persistence |
| FP-gate model API | `model_service.app:app` | BERT/LR inference, FP-gate decision, risk bands, and batch scoring |

The model API loads these frozen experiment assets from
`model_service/models/`:

- `model_service/models/bert/best/`
- `model_service/models/lr/`
- `model_service/models/ensemble/`
- `model_service/models/risk/`

## Local development

Requirements: Python 3.11+, Node.js 22.22+, PostgreSQL, and the new model
artifacts. Install the application backend and frontend dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp backend/env.example .env

cd frontend
npm ci
cd ..
```

`backend/requirements-dev.txt` installs only the lightweight application API
and its development tools. TensorFlow, PyTorch, Transformers, XGBoost, and
other model runtimes are not backend dependencies; model-specific packages are
isolated in `model_service/requirements.txt`.

Start the FP-gate model service first:

```bash
pip install -r model_service/requirements.txt
python model_service/app.py
```

For any production deployment, set `APP_ENV=production`, a non-empty
`MODEL_API_KEY`, and restrictive `MODEL_CORS_ORIGINS`. Use the same secret as
`MODEL_SERVER_API_KEY` in the application backend.

Set `MODEL_SERVER_URL=http://127.0.0.1:5000`, then start the application API:

```bash
python -m alembic -c backend/alembic.ini upgrade head
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend from `frontend/` with `npm run dev`. It is served at
`http://localhost:5190` and proxies `/api` to FastAPI.

## Model contract

FastAPI uses only the following model-service endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Model service health |
| `POST` | `/predict/all` | LR score, BERT score, FP-gate decision, and risk result |
| `POST` | `/predict/batch` | Batched final risk scores used by SHAP |

`MODEL_SERVER_URL` is required. If the service is unavailable or returns an
invalid contract, the application returns `503`; it never silently switches to
an older model.

## Tests

Application and model-service tests use **pytest**. Install dev dependencies first:

```bash
pip install -r backend/requirements-dev.txt
pip install -r model_service/requirements-dev.txt
```

Run the full suite from the repository root:

```bash
pytest -q
```

Run only the model service tests (fast, mocked integration layer):

```bash
pytest model_service/tests -q
```

Optional slow end-to-end tests that load real BERT/LR weights:

```bash
RUN_MODEL_SERVICE_E2E=1 pytest model_service/tests -m e2e -q
```

### model_service coverage

| Area | Tests | Notes |
| --- | --- | --- |
| Input validation | `test_text_utils.py` | Happy + sad cases for JSON/text limits |
| Business logic | `test_ensemble_service.py` | FP-gate and risk-band rules |
| Race conditions | `test_coalesce.py` | Concurrent identical inference coalescing |
| HTTP integration | `test_app_integration.py` | Auth, batch limits, sanitized 500 responses |
| Real weights (opt-in) | `test_app_e2e.py` | Skipped unless `RUN_MODEL_SERVICE_E2E=1` |

See [`model_service/tests/README.md`](model_service/tests/README.md) for the full testing
approach and mocking strategy.

Frontend checks:

```bash
cd frontend
npm run lint
npm test
npm run build
```

See [the development guide](docs/development-guide.md) and
[architecture documentation](docs/architecture/README.md) for more details.
