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

The checked-in Cloud Run workflow deploys the application backend and frontend;
the model service has a separate release lifecycle. See the
[deployment ownership and verification checklist](docs/architecture/deployment.md#production-ownership)
before releasing the application.

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

## Testing strategy and coverage

Testing is split into four deliberately named layers. The distinction matters: a test is
only called end-to-end when it crosses real process boundaries rather than intercepting
those boundaries with mocks.

| Layer | Scope | External dependencies |
| --- | --- | --- |
| Unit | Pure functions, services, validation, React components and rendering logic | Mocked or stubbed |
| Integration | FastAPI/Flask routes, database transactions, model-service HTTP contracts and component/API interaction | Test database or mocked model/network boundary |
| Mocked browser flow | Browser rendering, navigation, responsive layouts and user interactions across the frontend | API calls intercepted with Playwright `page.route()` |
| Real E2E / smoke | A real browser, Vite proxy, FastAPI process and isolated SQLite database; separately, the real BERT/LR model artifacts | No mock at the boundary under test |

Happy paths and sad paths are covered at every practical layer. Examples include valid
and invalid input, authentication failures, database rollback, model timeouts and
unavailability, sanitized `500` responses, concurrent cache misses, failed inference
coalescing, frontend submission recovery, and unavailable explanation data.

### Current coverage gates and verified baseline

The following baseline was verified locally on 11 August 2026. CI rejects regressions
below the configured gates.

| Area | CI gate | Verified baseline | Test focus |
| --- | --- | --- | --- |
| Backend | 85% total | 92.16% | Validation, auth, database behavior, business logic, error mapping, cache races |
| Model service | 90% total | 95.71% | Model loading/inference adapters, FP-gate rules, HTTP contracts, concurrent request coalescing |
| Frontend | 80% statements, 76% branches, 75% functions, 83% lines | 88.96% statements, 85.01% branches, 86.13% functions, 91.67% lines | Rendering, component behavior, user interaction, routing and API failures |

The verified default suites contain 185 passing Python tests and 65 passing frontend
tests. Coverage is a regression gate rather than the sole quality measure; assertions
also check observable behavior, error safety, persistence and concurrency.

### Install test dependencies

Application and model-service tests use **pytest**. Install their development
dependencies first:

```bash
pip install -r backend/requirements-dev.txt
pip install -r model_service/requirements-dev.txt
```

Run the full suite from the repository root:

```bash
pytest -q
```

Run the backend coverage gate:

```bash
cd backend
pytest tests -q --cov=. --cov-config=.coveragerc --cov-fail-under=85
```

Run only the model-service unit and mocked integration tests:

```bash
pytest model_service/tests -q
```

Run the frontend unit/component coverage gate:

```bash
cd frontend
npm ci
npm run test:coverage
```

### Browser tests

Install the pinned Chromium build once, then run the mocked browser flows:

```bash
cd frontend
npx playwright install chromium
npm run test:e2e
```

These flows intentionally intercept API requests. They test frontend navigation,
rendering, responsive desktop/tablet/mobile layouts, request payloads, successful user
journeys and recoverable service failures. They do **not** prove that the backend or
database is reachable.

Run the separate real-backend smoke test with:

```bash
cd frontend
npm run test:e2e:real
```

This test starts Vite on port `5191` and FastAPI on port `8001`, creates a disposable
SQLite database, registers a user through the real API, and reads that user's history
through the real API. It performs no `page.route()` interception. The database is
deleted when the backend test process exits. It deliberately stops at the model-service
boundary so the fast CI smoke test does not load multi-gigabyte model artifacts.

### Real model-service E2E

The model-service E2E test loads the committed BERT checkpoint and LR artifact and calls
the real `/predict/all` route without mocking model loading or inference:

```bash
RUN_MODEL_SERVICE_E2E=1 pytest model_service/tests -m e2e -q
```

It is excluded from the default suite because loading Transformer/PyTorch weights is
slow and memory intensive, and CPU-only runners can take substantially longer or run out
of memory. Run it on a machine with enough RAM and disk space; a GPU is optional. A
failure caused by unavailable memory, missing large artifacts, or runner time limits is
an environment limitation and must be recorded in the report/CI result rather than
silently treated as a product assertion failure. Without `RUN_MODEL_SERVICE_E2E=1`,
pytest reports the test as skipped with the reason.

### Detailed coverage map

| Area | Tests | Notes |
| --- | --- | --- |
| Input validation | `test_text_utils.py` | Happy + sad cases for JSON/text limits |
| Business logic | `test_ensemble_service.py` | FP-gate and risk-band rules |
| Race conditions | `test_coalesce.py` | Concurrent identical inference coalescing |
| HTTP integration | `test_app_integration.py` | Auth, batch limits, sanitized 500 responses |
| Real weights (opt-in) | `test_app_e2e.py` | Skipped unless `RUN_MODEL_SERVICE_E2E=1` |
| Backend routes and persistence | `backend/tests/test_api.py`, `test_analysis_router.py` | Real test database, auth, rollback and safe error mapping |
| Backend race conditions | `backend/tests/test_cache.py` | Concurrent misses compute once and share the result |
| Frontend unit/component | `frontend/src/**/*.test.{js,jsx}` | Rendering, interactions, storage and API happy/sad paths |
| Mocked browser flow | `frontend/e2e/core-flows.spec.js` | Browser journeys against intercepted API responses |
| Real-backend browser smoke | `frontend/e2e/real-backend-smoke.spec.js` | Browser → Vite proxy → FastAPI → disposable SQLite |

See [`model_service/tests/README.md`](model_service/tests/README.md) for the full testing
approach and mocking strategy.

Full CI-equivalent frontend checks:

```bash
cd frontend
npm run lint
npm run format:check
npm run test:coverage
npx playwright install chromium
npm run test:e2e
npm run test:e2e:real
npm run build
```

See [the development guide](docs/development-guide.md) and
[architecture documentation](docs/architecture/README.md) for more details.
