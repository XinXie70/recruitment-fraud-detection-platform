# Fake Job Detection — Final Model Pipelines

Two **text-only** fake job detection models — Logistic Regression and DNN — with three-tier risk mapping post-processing.

> **Note:** Training, evaluation, and prediction are fully self-contained within `final_model_pipelines/` and **do not depend** on `data_cleaning/`. Only `DataSet.csv` is required (see "Data Preparation" below).

---

## Directory Structure

```text
final_model_pipelines/
├── README.md
├── prepare_data.py           ← data cleaning + splitting (self-contained)
├── data/                     ← data directory
│   ├── DataSet.csv           ← raw data (or place at repository root)
│   ├── cleaned_data.csv
│   └── splits/               ← train / val / test
├── shared_config.py
├── text_utils.py
├── data_split.py
├── predict_all.py
├── risk_mapping.py
├── lr_pipeline/
└── dnn_pipeline/
```

---

## Requirements

- **Python 3.10+** (verified on 3.10)
- Windows / Linux / macOS

---

## Installation

Run from the **repository root** (`datapreprocessing/`, the parent directory of `final_model_pipelines/`):

### Option A: Editable install (recommended, resolves import paths)

```bash
cd datapreprocessing
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -e ".[inference]"     # inference only (LR + DNN)
# pip install -e ".[full]"      # training + evaluation + inference
```

After installation, you can run from any directory:

```python
from final_model_pipelines.predict_all import predict_with_all_models
```

### Option B: Run directly from repository root without installing

```bash
cd datapreprocessing
python final_model_pipelines/predict_all.py
```

All scripts add the repository root to `sys.path` internally. **You must run from the root directory**, or set:

```bash
set PYTHONPATH=C:\path\to\datapreprocessing    # Windows
export PYTHONPATH=/path/to/datapreprocessing   # Linux / macOS
```

### Dependencies

| Scenario | Command |
|----------|---------|
| Inference only (LR + DNN) | `pip install -e ".[inference]"` |
| Training / evaluation | `pip install -e ".[full]"` + prepare `DataSet.csv` |

---

## Data Preparation (before training, self-contained)

Place `DataSet.csv` in **either** location:

- `final_model_pipelines/data/DataSet.csv` (recommended)
- Repository root `datapreprocessing/DataSet.csv`

Then run:

```bash
cd datapreprocessing
python final_model_pipelines/prepare_data.py
```

This automatically performs: HTML cleaning → `combined_text` → save `data/cleaned_data.csv` → stratified split to `data/splits/`.

---

## Quick Prediction

### Python API (backend integration)

```python
from final_model_pipelines.lr_pipeline.predict import predict_job_posting
from final_model_pipelines.dnn_pipeline.predict import predict_job_posting as dnn_predict
from final_model_pipelines.predict_all import predict_with_all_models

# Single model
result = predict_job_posting("Software engineer at Google. Bachelor degree required...")
print(result)

# Both models
results = predict_with_all_models("URGENT! Work from home, wire transfer required...")
print(results)
```

### Command-line smoke test

```bash
cd datapreprocessing
python final_model_pipelines/predict_all.py
```

---

## API Response Fields (frontend/backend contract)

### Single-model response

```json
{
  "model": "Logistic Regression",
  "risk_score": 0.7694,
  "classification_label": "Suspicious",
  "prediction": "fake",
  "recommended_action": "Review Required"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | `"Logistic Regression"` or `"DNN"` |
| `risk_score` | float | Predicted fake job probability, 0~1; higher means more suspicious |
| `classification_label` | string | `Likely Legitimate` / `Suspicious` / `Likely Deceptive` |
| `prediction` | string | Backend binary label: `real` (prob < LOW) or `fake` (prob ≥ LOW) |
| `recommended_action` | string | `Safe` / `Review Required` / `High Risk Warning` |

### Dual-model response (`predict_all.py`)

```json
{
  "logistic_regression": {
    "risk_score": 0.91,
    "classification_label": "Likely Deceptive",
    "prediction": "fake",
    "recommended_action": "High Risk Warning"
  },
  "dnn": {
    "risk_score": 0.77,
    "classification_label": "Suspicious",
    "prediction": "fake",
    "recommended_action": "Review Required"
  }
}
```

### Risk Mapping Rules (post-processing, not training classes)

Models are trained as **real / fake binary classifiers**; three-tier labels are mapped from `risk_score` via `risk_mapping.py`:

| risk_score | classification_label | prediction | recommended_action |
|------------|---------------------|------------|-------------------|
| < LOW (0.19 LR / 0.33 DNN) | Likely Legitimate | **real** | Safe |
| LOW ~ HIGH (0.91) | Suspicious | **fake** | Review Required |
| ≥ HIGH | Likely Deceptive | **fake** | High Risk Warning |

Thresholds are saved in each pipeline's `saved_model/threshold.json` (tuned on validation set).

---

## Input Requirements

- **Pass a single job posting text string**
- No metadata required (location, salary, etc.)
- If you have multiple fields, concatenate them on the backend before passing; training uses `title + company_profile + description + requirements + benefits`

---

## Training and Evaluation (ML team, self-contained)

```bash
cd datapreprocessing
python final_model_pipelines/prepare_data.py                 # first run: clean + split

python final_model_pipelines/lr_pipeline/train_model.py
python final_model_pipelines/lr_pipeline/evaluate_model.py

python final_model_pipelines/dnn_pipeline/train_model.py
python final_model_pipelines/dnn_pipeline/evaluate_model.py
```

Data and splits are stored under `final_model_pipelines/data/`; no `data_cleaning/` dependency.

---

## Model Selection Guide

| Scenario | Recommendation |
|----------|----------------|
| Production API / low latency | **Logistic Regression** (~93 KB, millisecond-level) |
| High-recall screening | **DNN** (Test Recall ≈ 0.86) |
| Dashboard side-by-side comparison | `predict_with_all_models` |

See [`model_comparison_summary.md`](model_comparison_summary.md) for details.

---

## FAQ

**Q: `ModuleNotFoundError: No module named 'final_model_pipelines'`**  
A: Run `pip install -e ".[inference]"` from the repository root, or set `PYTHONPATH` to the root.

**Q: `FileNotFoundError: model not found`**  
A: Ensure `lr_pipeline/saved_model/` and `dnn_pipeline/saved_model/` are present (including `model.joblib` / `model.keras`, etc.).

**Q: I only want to deploy LR without TensorFlow**  
A: You can call `lr_pipeline/predict.py` alone; however, `predict_all.py` loads both models and requires TensorFlow.

**Q: Is low precision expected?**  
A: The current threshold strategy favors **high recall (fewer missed fake jobs)**, so the Suspicious tier may be large and should be paired with manual review. See each pipeline's `outputs/evaluation_results.csv`.

---

## Submodule Documentation

- [LR Pipeline README](lr_pipeline/README.md)
- [DNN Pipeline README](dnn_pipeline/README.md)
