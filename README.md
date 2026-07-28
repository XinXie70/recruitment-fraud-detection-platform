# Fake Job Advertisement Detection — LR + BERT + Ensemble

This fake job advertisement detection project retains three production models:

| Model | Description |
|---|---|
| **Logistic Regression** | TF-IDF + LR baseline |
| **BERT (class-weighted)** | BERT fine-tuned with class weights |
| **LR + BERT Ensemble** | Weighted ensemble with risk bands |

## Project structure

```text
├── data/splits/              # Fixed train / validation / test splits
├── model_code/bert/          # BERT training and inference code
├── model_weights/            # LR joblib + BERT safetensors
├── model_results/bert/       # BERT evaluation results
├── reports/models/           # Metrics, predictions, ensemble, and risk-band configs
├── src/
│   ├── data_pipeline/        # Shared data processing
│   ├── models/               # LR training, ensemble, and risk-band scripts
│   └── api/                  # FastAPI inference service
└── scripts/smoke_test_api.py # Smoke test for the E: drive environment
```

## Quick start

### 1. Environment (E: drive with CUDA)

```powershell
git lfs install
git lfs pull
. E:\ml\activate.ps1
cd f:\final-version2\capstone-project-26t2-9900-h09c-almond
pip install -r requirements.txt
pip install -r model_code/requirements-bert.txt
```

### 2. Train LR (if weights are unavailable)

```powershell
python src/models/logistic_regression/train_baseline.py
```

Weights are written to `model_weights/logistic_regression/logistic_regression_baseline.joblib`.

### 3. Start FastAPI

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger documentation: http://127.0.0.1:8000/docs

### 4. Smoke test (E: drive)

```powershell
python scripts/smoke_test_api.py
```

Results are written to `E:\ml\smoke-test-results\fake-job-api-smoke.json`.

## Frontend development

The frontend requires Node.js 20.19 or a compatible newer release. With `nvm`:

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
npm run build
```

Husky and lint-staged automatically format and lint staged frontend files before
each commit. GitHub Actions runs the backend tests, frontend linting, frontend
tests, and a production build.

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
[`docs/final-demo-readiness.md`](docs/final-demo-readiness.md).

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict/lr` | Logistic Regression prediction |
| POST | `/predict/bert` | Class-weighted BERT prediction |
| POST | `/predict/ensemble` | LR+BERT ensemble prediction |
| POST | `/predict/ensemble/risk` | Ensemble prediction with Low/Suspicious/High risk band |
| POST | `/predict/lr/batch` | Batch LR prediction (up to 100 records) |

### Request example

```json
POST /predict/ensemble/risk
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

## Backend integration example (Python)

```python
import httpx

resp = httpx.post(
    "http://127.0.0.1:8000/predict/ensemble/risk",
    json={"text": job_ad_text},
    timeout=30.0,
)
result = resp.json()
risk_level = result["risk_level"]      # Low | Suspicious | High
risk_score = result["risk_score"]      # 0–100
```

## Dataset

The fixed splits are stored in `data/splits/` (70% Train / 15% Validation / 15% Test, seed 42).

Obtain the raw EMSCAD dataset separately, rename it to `data/raw/emscad_v1.csv`,
and see `DATA_CONTRACT_V1.md` for details.

Dataset source: Vidros et al. (2017), *Automatic Detection of Online Recruitment
Frauds: Characteristics, Methods, and a Public Dataset*.
https://doi.org/10.3390/fi9010006

## Model training and evaluation workflow

```powershell
# LR baseline
python src/models/logistic_regression/train_baseline.py

# Evaluate BERT with existing weights
cd model_code/bert
python evaluate_bert.py --checkpoint_dir ..\..\model_weights\bert\bert_class_weighted\best

# Search ensemble weights on validation; apply the locked config to test
cd ..\..
python src/models/build_ensemble.py --mode validation
python src/models/build_ensemble.py --mode test

# Risk bands
python src/models/build_risk_bands.py --mode validation
python src/models/build_risk_bands.py --mode test
```

## Risk-band configuration

Frozen configuration: `reports/models/risk_band_v1_config.json`

| Level | Rule |
|---|---|
| **Low** | fraud_score < 0.1567 |
| **Suspicious** | 0.1567 ≤ fraud_score < 0.62 |
| **High** | fraud_score ≥ 0.62 |

`risk_score = fraud_score × 100`

## Retained result files

- `reports/models/logistic_regression/` — LR metrics and predictions
- `reports/models/bert/` — BERT metrics and predictions
- `reports/models/ensemble_lr_bert/` — Ensemble configuration and predictions
- `reports/models/risk_band_v1_*` — Risk-band configuration and results

See `MODEL_EXPERIMENT_CONTRACT_V1.md` for additional experiment conventions.
