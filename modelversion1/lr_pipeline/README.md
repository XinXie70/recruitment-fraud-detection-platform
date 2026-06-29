# Logistic Regression Final Pipeline

**Text-only** fake job detection pipeline based on the `lr_comparison` experiments, outputting structured JSON results.

## Final Configuration

| Setting | Choice | Experiment Test Performance |
|---------|--------|----------------------------|
| Text preprocessing | HTML cleaning + five fields → `combined_text` | — |
| Feature extraction | **TF-IDF** (2000 features, 1-2 gram) | — |
| Class imbalance | **Class Weighting** `{0:1, 1:6}` | F1=0.791, Recall=0.786 |
| LR parameters | lbfgs, max_iter=1000 | — |
| Risk tiers | validation dual-threshold tuning | LOW / HIGH → three-tier labels |

## Training Pipeline

```text
DataSet.csv → prepare_data.py → data/cleaned_data.csv + data/splits/
→ train-only fit vectorizer → Class Weighting training
→ evaluate (threshold tuning + binary metrics)
```

## How to Run

```bash
python final_model_pipelines/prepare_data.py
python final_model_pipelines/lr_pipeline/train_model.py
python final_model_pipelines/lr_pipeline/evaluate_model.py
```

## Prediction Pipeline

```text
validation_pipeline (input validity + job relevance)
→ combined_text preprocessing
→ LR binary classification
→ risk_score
→ risk_mapping layer
→ structured JSON (status: success / success_with_warning / rejection)
```

The model is **not trained as a three-class classifier**; Likely Legitimate / Suspicious / Likely Deceptive are post-processing display labels only.

Pre-prediction validation is handled by `validation_pipeline.py` (see [main README](../README.md#pre-prediction-validation-pipeline)). LR / DNN are only called when input passes both `input_validator.py` and `job_description_filter.py`.

## Structured Output

Three outcome types — check `status` first:

| `status` | Model called? | Description |
|----------|---------------|-------------|
| `success` | Yes | Valid job-related input; normal prediction |
| `success_with_warning` | Yes | Incomplete job posting; prediction included with `message` warning |
| `invalid_input` / `not_job_related` | No | Rejected before inference; prediction fields are `null` |

Example success response:

```json
{
  "status": "success",
  "job_relevance_score": 0.68,
  "model": "Logistic Regression",
  "risk_score": 0.68,
  "classification_label": "Suspicious",
  "prediction": "fake",
  "recommended_action": "Review Required"
}
```

`risk_score` is the model's fake job probability (0~1); higher values indicate greater suspicion.

## Risk Mapping Rules

| risk_score | classification_label | prediction | recommended_action |
|-------------------|---------------------|------------|-------------------|
| < LOW | Likely Legitimate | real | Safe |
| LOW ~ HIGH | Suspicious | fake | Review Required |
| ≥ HIGH | Likely Deceptive | fake | High Risk Warning |

## Backend Integration

```python
from final_model_pipelines.lr_pipeline.predict import predict_job_posting
result = predict_job_posting("Job posting text...")
```

Only text input is required; no metadata needed.
