# Fake Job Advertisement Detection Platform

> **Product entry points:** use `frontend/` for the web client and
> `backend.main:app` for the application API. The code under `src/api/` is a
> reproducible reference model API, while `api_flask/` is a retained legacy
> implementation. See the [code ownership guide](docs/architecture/code-ownership.md)
> before adding cross-directory dependencies.

This project combines a React web application, an authenticated FastAPI backend,
and separately deployable model inference services. The application backend can
orchestrate eight model families, while the locked reference API retains the
three-model LR/BERT experiment:

| Model                     | Description                        |
| ------------------------- | ---------------------------------- |
| **Logistic Regression**   | TF-IDF + LR baseline               |
| **BERT (class-weighted)** | BERT fine-tuned with class weights |
| **LR + BERT Ensemble**    | Weighted ensemble with risk bands  |

## Application services

The repository contains three service implementations with different purposes:

| Service | Entry point | Responsibility |
| ------- | ----------- | -------------- |
| Web application backend | `backend.main:app` | Authentication, users, analysis workflow, history, administration, and database access |
| Locked reference model API | `src.api.main:app` | Standalone LR, BERT, weighted ensemble, and risk-band endpoints used by the reproducible reference experiment |
| Legacy FP-gate model API | `api_flask.app:app` | Flask deployment retained for the paper-aligned LR/BERT FP-gate pipeline |

The web application backend can load its configured model adapters locally or
call a compatible separately deployed model service through `MODEL_SERVER_URL`.
The inference services are not replacements for the application backend, and
their endpoint contracts are not interchangeable. New application features
should target `backend.main:app`; use the other entry points only when
reproducing or deploying their documented model pipelines.

## Project structure

```text
├── frontend/                 # React and Vite web client
├── backend/                  # Main FastAPI application backend
├── model/final_model_pipelines/ # Eight application model adapters and pipelines
├── api_flask/                # Legacy Flask FP-gate model API
├── src/api/                  # Locked LR/BERT FastAPI reference API
├── src/models/               # LR training, ensemble, and risk-band scripts
├── data/                     # Fixed splits and processed experiment data
├── model_code/               # Model training and inference code
├── model_weights/            # Versioned model artifacts (Git LFS)
├── model_results/            # Evaluation outputs
├── reports/models/           # Metrics, predictions, and frozen configurations
├── docs/                     # Architecture, ADRs, security, and development guides
└── scripts/                  # Data, evaluation, and API utility scripts
```

## Quick start

### 1. Environment

Run these commands from the repository root. Python 3.12, Node.js 22.22, and
Git LFS are required.

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r model_code/requirements-bert.txt
pip install -e model
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1`.

### 2. Train LR (if weights are unavailable)

```bash
python src/models/logistic_regression/train_baseline.py
```

Weights are written to `model_weights/logistic_regression/logistic_regression_baseline.joblib`.

### 3. Start the standalone model inference API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger documentation: http://127.0.0.1:8000/docs

### 4. Smoke test

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict/lr \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent work-from-home role. Send bank details to apply."}'
```

## Web application backend development

Install the website backend dependencies, create the local configuration, and
apply the database migrations before starting it:

```bash
pip install -r backend/requirements-dev.txt
pip install -e model
cp backend/env.example .env
python -m alembic -c backend/alembic.ini upgrade head
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Its Swagger documentation is available at http://127.0.0.1:8000/docs. This service
provides authentication, analysis history, administration, and the application-facing
analysis API. It requires the database and other settings documented in
`backend/env.example`.

Leave `MODEL_SERVER_URL` unset to use the application's local model adapters.
When using a remote model service, follow its API contract and set the variable
as described in [the development guide](docs/development-guide.md).

## Frontend development

The frontend requires Node.js 22.22 or a compatible newer release. With `nvm`:

```bash
nvm use
cd frontend
npm ci
npm run dev
```

Before opening a pull request, run:

```bash
npm run lint
npm test
npm run test:coverage
npm run test:e2e
npm run build
```

The Playwright suite covers authentication redirects, registration, successful
analysis with history persistence, and recoverable model-service failure. On a
new development machine, install its headless browser once with:

```bash
npx playwright install --only-shell chromium
```

Husky and lint-staged automatically format and lint staged frontend files before
each commit. GitHub Actions runs the backend tests, frontend linting, frontend
tests, and a production build.

For a concise onboarding checklist and local development workflow, see
[docs/development-guide.md](docs/development-guide.md).

## Full-stack Docker development

Start PostgreSQL, run the database migrations, and launch the backend and
frontend with one command:

```bash
docker compose up --build
```

The frontend is available at http://localhost:5190 and proxies `/api` requests
to the backend container. Set `MODEL_SERVER_URL` before starting Compose when a
standalone model inference service is required:

```bash
MODEL_SERVER_URL=https://model-api.example.com docker compose up --build
```

Do not expose an unauthenticated model server directly to the public internet.
The Final Demo evidence and rehearsal checklist are maintained in
[`docs/archive/final-demo-readiness.md`](docs/archive/final-demo-readiness.md).
Architecture diagrams and rationale are available in
[`docs/architecture/`](docs/architecture/README.md) and
[`docs/design-justification.md`](docs/design-justification.md).

## Model inference API endpoints

| Method | Path                     | Description                                            |
| ------ | ------------------------ | ------------------------------------------------------ |
| GET    | `/health`                | Health check                                           |
| POST   | `/predict/lr`            | Logistic Regression prediction                         |
| POST   | `/predict/bert`          | Class-weighted BERT prediction                         |
| POST   | `/predict/ensemble`      | LR+BERT ensemble prediction                            |
| POST   | `/predict/ensemble/risk` | Ensemble prediction with Low/Suspicious/High risk band |
| POST   | `/predict/lr/batch`      | Batch LR prediction (up to 100 records)                |

### Request example

```http
POST /predict/ensemble/risk
Content-Type: application/json

{
  "text": "Urgent work-from-home job. Send bank details to apply."
}
```

The equivalent JSON request body is:

```json
{
  "text": "Urgent work-from-home job. Send bank details to apply."
}
```

### Response example

```json
{
  "model": "ensemble_lr_bert_class_weighted",
  "fraud_score": 0.82,
  "threshold": 0.62,
  "prediction": 1,
  "predicted_label": "Fraudulent",
  "weights": {
    "logistic_regression_baseline": 0.6,
    "bert_class_weighted": 0.4
  },
  "lr_fraud_score": 0.75,
  "bert_fraud_score": 0.91,
  "risk_score": 82.0,
  "risk_level": "High",
  "binary_threshold": 0.62,
  "low_suspicious_threshold": 0.1567,
  "suspicious_high_threshold": 0.62
}
```

## Model API integration example (Python)

```python
import httpx

resp = httpx.post(
    "http://127.0.0.1:8000/predict/ensemble/risk",
    json={"text": job_ad_text},
    timeout=30.0,
)
result = resp.json()
risk_level = result["risk_level"]  # Low | Suspicious | High
risk_score = result["risk_score"]  # 0–100
```

## Dataset

The fixed splits are stored in `data/splits/` (70% Train / 15% Validation / 15% Test, seed 42).

Obtain the raw EMSCAD dataset separately, rename it to `data/raw/emscad_v1.csv`,
and see `DATA_CONTRACT_V1.md` for details.

Dataset source: Vidros et al. (2017), _Automatic Detection of Online Recruitment
Frauds: Characteristics, Methods, and a Public Dataset_.
https://doi.org/10.3390/fi9010006

## Model training and evaluation workflow

```bash
# LR baseline
python src/models/logistic_regression/train_baseline.py

# Evaluate BERT with existing weights
python model_code/bert/evaluate_bert.py \
  --checkpoint_dir model_weights/bert/bert_class_weighted/best

# Search ensemble weights on validation; apply the locked config to test
python src/models/build_ensemble.py --mode validation
python src/models/build_ensemble.py --mode test

# Risk bands
python src/models/build_risk_bands.py --mode validation
python src/models/build_risk_bands.py --mode test
```

## Risk-band configuration

Frozen configuration: `reports/models/risk_band_v1_config.json`

| Level          | Rule                        |
| -------------- | --------------------------- |
| **Low**        | fraud_score < 0.1567        |
| **Suspicious** | 0.1567 ≤ fraud_score < 0.62 |
| **High**       | fraud_score ≥ 0.62          |

`risk_score = fraud_score × 100`

## Retained result files

- `reports/models/logistic_regression/` — LR metrics and predictions
- `reports/models/bert/` — BERT metrics and predictions
- `reports/models/ensemble_lr_bert/` — Ensemble configuration and predictions
- `reports/models/risk_band_v1_*` — Risk-band configuration and results

See `MODEL_EXPERIMENT_CONTRACT_V1.md` for additional experiment conventions.
