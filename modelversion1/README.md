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
├── input_validator.py        ← layer 1: basic text validity
├── job_description_filter.py ← layer 2/3: job-posting relevance scoring
├── validation_pipeline.py    ← orchestrates pre-prediction checks + API responses
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

Every call to `predict_job_posting()` returns **one of three outcome types**. Check the `status` field first before reading model prediction fields.

### Outcome type overview

| Outcome type | `status` value | Model called? | When |
|--------------|----------------|---------------|------|
| **1. Success** | `success` | Yes | Input is valid and clearly job-related |
| **2. Success with warning** | `success_with_warning` | Yes | Input may be a job posting but lacks common fields (short or incomplete); prediction is less reliable |
| **3. Rejection** | `invalid_input` or `not_job_related` | **No** | Input failed pre-prediction validation |

> The pre-prediction filter judges whether input **is a job posting** — not whether it is fake or real. Fake/real detection is performed by LR / DNN only after validation passes.

---

### Type 1 — Success (`status: "success"`)

Input passed both validation stages. Model prediction and risk mapping are included.

```json
{
  "status": "success",
  "job_relevance_score": 0.8117,
  "model": "Logistic Regression",
  "risk_score": 0.1196,
  "classification_label": "Likely Legitimate",
  "prediction": "real",
  "recommended_action": "Safe"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"success"` |
| `job_relevance_score` | float | Job-description relevance score (0~1) from `job_description_filter.py` |
| `model` | string | `"Logistic Regression"` or `"DNN"` |
| `risk_score` | float | Predicted fake job probability, 0~1; higher means more suspicious |
| `classification_label` | string | `Likely Legitimate` / `Suspicious` / `Likely Deceptive` |
| `prediction` | string | Backend binary label: `real` (prob < LOW) or `fake` (prob ≥ LOW) |
| `recommended_action` | string | `Safe` / `Review Required` / `High Risk Warning` |

---

### Type 2 — Success with warning (`status: "success_with_warning"`)

Input passed validation but may be incomplete (e.g. short posting, missing responsibilities/requirements). Model is still invoked; frontend should show `message` to the user.

```json
{
  "status": "success_with_warning",
  "message": "Input may be a job posting, but it lacks common job description fields. Prediction may be less reliable.",
  "job_relevance_score": 0.42,
  "model": "Logistic Regression",
  "risk_score": 0.2303,
  "classification_label": "Suspicious",
  "prediction": "fake",
  "recommended_action": "Review Required"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"success_with_warning"` |
| `message` | string | Human-readable warning explaining why prediction may be unreliable |
| `job_relevance_score` | float | Job-description relevance score (0~1); typically below 0.55 |
| Other fields | — | Same as Type 1 success response |

---

### Type 3 — Rejection (model not called)

When validation fails, **LR / DNN are not invoked**. `risk_score`, `classification_label`, and `prediction` are always `null`.

Two rejection sub-types:

#### 3a. Invalid input (`status: "invalid_input"`)

Failed **basic text validity** checks (`input_validator.py`): empty text, too short (< 8 English words), URL-only, gibberish, code snippet, casual chat, etc.

```json
{
  "status": "invalid_input",
  "message": "Input cannot be empty.",
  "model": "Logistic Regression",
  "classification_label": null,
  "risk_score": null,
  "prediction": null,
  "recommended_action": "Please enter a valid job description.",
  "job_relevance_score": 0.0
}
```

#### 3b. Not job-related (`status: "not_job_related"`)

Passed basic validity but failed **job-description relevance** filter (`job_description_filter.py`): text is valid English but does not look like a job posting (e.g. news article, recipe, academic abstract).

```json
{
  "status": "not_job_related",
  "message": "Input text is valid English text but does not appear to be a job posting or job description.",
  "model": "Logistic Regression",
  "classification_label": null,
  "risk_score": null,
  "prediction": null,
  "recommended_action": "Please enter a valid job posting or job description.",
  "job_relevance_score": 0.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"invalid_input"` or `"not_job_related"` |
| `message` | string | Reason the input was rejected |
| `model` | string | Model name (for logging; model was not run) |
| `classification_label` | null | Not available |
| `risk_score` | null | Not available |
| `prediction` | null | Not available |
| `recommended_action` | string | User-facing guidance |
| `job_relevance_score` | float \| null | Relevance score; `0.0` for `invalid_input`, computed value for `not_job_related` |

---

### Pre-prediction validation pipeline

Before LR / DNN inference, `validation_pipeline.py` runs two independent stages:

```text
User input
  → input_validator.validate_input_text()           # layer 1: basic text validity
  → job_description_filter.check_job_description_relevance()  # layer 2/3: job relevance
  → (only if both pass) text preprocessing → model → risk_mapping
```

| Module | Responsibility | Key function |
|--------|----------------|--------------|
| `input_validator.py` | Is the input valid text? | `validate_input_text()` |
| `job_description_filter.py` | Is the text job-related? | `check_job_description_relevance()` |
| `validation_pipeline.py` | Orchestration + response wrapping | `validate_job_input()` |

`job_description_filter.py` uses a rule-based weighted scoring algorithm (`job_relevance_score`, threshold 0.4) combining job keywords, structure fields, job titles, hiring phrases, and non-job topic penalties. It does **not** use LR / DNN.

---

### Dual-model response (`predict_all.py`)

```json
{
  "logistic_regression": {
    "status": "success",
    "job_relevance_score": 0.81,
    "risk_score": 0.91,
    "classification_label": "Likely Deceptive",
    "prediction": "fake",
    "recommended_action": "High Risk Warning"
  },
  "dnn": {
    "status": "success_with_warning",
    "message": "Input may be a job posting, but it lacks common job description fields. Prediction may be less reliable.",
    "job_relevance_score": 0.42,
    "risk_score": 0.77,
    "classification_label": "Suspicious",
    "prediction": "fake",
    "recommended_action": "Review Required"
  }
}
```

Each sub-object follows the same three outcome types above (`status` field). The `model` key is omitted in dual-model responses.

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
- Input is checked by the pre-prediction validation pipeline before model inference (see [API Response Fields](#api-response-fields-frontendbackend-contract))
- Minimum **8 English words**; non-job content (news, recipes, essays, etc.) is rejected even if long enough

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

**Q: Why did I get `not_job_related` instead of a prediction?**  
A: The input passed basic validity but was judged unrelated to job postings by `job_description_filter.py` (e.g. news article, recipe). Enter an actual job description text.

**Q: What is the difference between `invalid_input` and `not_job_related`?**  
A: `invalid_input` means the text itself is invalid (empty, too short, URL, gibberish, code). `not_job_related` means the text is valid English but does not look like a job posting.

---

## Submodule Documentation

- [LR Pipeline README](lr_pipeline/README.md)
- [DNN Pipeline README](dnn_pipeline/README.md)
