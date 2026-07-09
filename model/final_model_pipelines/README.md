# Fake Job Detection — Final Model Pipelines

Six **text-only** fake job detection models — Logistic Regression, SVM, XGBoost, DNN, RNN, and Bi-LSTM — with input validation and three-tier risk mapping post-processing.

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
├── compare_all_models.py
├── data_diagnostics.py       ← imbalance / missingness / shortcut-risk report
├── input_validator.py        ← layer 1: basic text validity
├── job_description_filter.py ← layer 2: job-posting relevance scoring
├── validation_pipeline.py    ← pre-prediction validation + API response shaping
├── risk_mapping.py
├── lr_pipeline/
├── svm_pipeline/
├── xgboost_pipeline/
├── dnn_pipeline/
├── rnn_pipeline/
└── bilstm_pipeline/
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

Install the aggregate pipeline requirements:

```bash
pip install -r final_model_pipelines/requirements.txt
```

For a lighter deployment, install only the selected model's requirements, for example `lr_pipeline/requirements.txt` or `svm_pipeline/requirements.txt`.

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

### Dataset Diagnostics and Imbalance

The EMSCAD dataset is highly imbalanced. In the full dataset used during development:

- Legitimate postings: `17,014` (`95.16%`)
- Fraudulent postings: `866` (`4.84%`)
- Approximate class ratio: `19.6 : 1`

The column `in_balanced_dataset` is a dataset marker, not a target label or model feature. It should not be used for training features because it leaks how the dataset subset was constructed.

Run the lightweight diagnostics script before retraining:

```bash
python final_model_pipelines/data_diagnostics.py --csv /path/to/DataSet.csv
```

Training uses stratified train / validation / test splits so validation and test keep the original fraud ratio. Compare models using fraud-class precision, recall, F1, and PR-AUC rather than accuracy alone.

---

## Quick Prediction

### Python API (backend integration)

```python
from final_model_pipelines.lr_pipeline.predict import predict_job_posting
from final_model_pipelines.predict_all import predict_with_all_models

# Single model
result = predict_job_posting("Software engineer at Google. Bachelor degree required...")
print(result)

# All models
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

Every prediction first passes the validation layer:

1. `input_validator.py` rejects empty text, URL-only input, gibberish, code snippets, and very short casual text.
2. `job_description_filter.py` checks whether valid text looks like a job posting.
3. The selected model runs only when validation passes.

Rejected inputs return `status` values such as `invalid_input` or `not_job_related` with `risk_score`, `classification_label`, and `prediction` set to `null`.

### Single-model response

```json
{
  "status": "success",
  "job_relevance_score": 0.8117,
  "model": "Logistic Regression",
  "risk_score": 0.7694,
  "classification_label": "Suspicious",
  "prediction": "fake",
  "recommended_action": "Review Required"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `success`, `success_with_warning`, `invalid_input`, or `not_job_related` |
| `message` | string/null | Warning or rejection reason when applicable |
| `job_relevance_score` | float | Job-posting relevance score from the validation layer |
| `model` | string | Model display name |
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

python final_model_pipelines/svm_pipeline/train_model.py
python final_model_pipelines/svm_pipeline/evaluate_model.py

python final_model_pipelines/xgboost_pipeline/train_model.py
python final_model_pipelines/xgboost_pipeline/evaluate_model.py

python final_model_pipelines/dnn_pipeline/train_model.py
python final_model_pipelines/dnn_pipeline/evaluate_model.py

python final_model_pipelines/rnn_pipeline/train_model.py
python final_model_pipelines/rnn_pipeline/evaluate_model.py

python final_model_pipelines/bilstm_pipeline/train_model.py
python final_model_pipelines/bilstm_pipeline/evaluate_model.py

python final_model_pipelines/compare_all_models.py
```

Data and splits are stored under `final_model_pipelines/data/`; no `data_cleaning/` dependency.

---

## Model Selection Guide

| Scenario | Recommendation |
|----------|----------------|
| Production API / low latency | **SVM** or **Logistic Regression** |
| Strongest current test F1 | **Bi-LSTM** |
| High-recall screening | **Logistic Regression**, **DNN**, **XGBoost**, or **Bi-LSTM** depending on the precision trade-off |
| Dashboard side-by-side comparison | `predict_with_all_models` |

See [`model_comparison_summary.md`](model_comparison_summary.md) for details.

---

## FAQ

**Q: `ModuleNotFoundError: No module named 'final_model_pipelines'`**
A: Run `pip install -e ".[inference]"` from the repository root, or set `PYTHONPATH` to the root.

**Q: `FileNotFoundError: model not found`**
A: Ensure each selected pipeline has its `saved_model/` artifacts present (`model.joblib` or `model.keras`, vectorizer/tokenizer, threshold files, etc.).

**Q: I only want to deploy LR/SVM/XGBoost without TensorFlow**
A: Call the selected non-neural pipeline directly. `predict_all.py` imports the neural models too and therefore requires TensorFlow.

**Q: Is low precision expected?**
A: The current threshold strategy favors **high recall (fewer missed fake jobs)**, so the Suspicious tier may be large and should be paired with manual review. See each pipeline's `outputs/evaluation_results.csv`.

---

## Submodule Documentation

- [LR Pipeline README](lr_pipeline/README.md)
- [DNN Pipeline README](dnn_pipeline/README.md)
