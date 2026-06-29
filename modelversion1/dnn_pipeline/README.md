# DNN Final Pipeline

**Text-only** fake job detection pipeline based on the `dnn_comparison` experiments (TensorFlow/Keras).

## Final Configuration

| Setting | Choice | Experiment Test Performance |
|---------|--------|----------------------------|
| Text preprocessing | HTML cleaning + five fields → `combined_text` | — |
| Vectorization | **BoW** (2000 features, 1-2 gram) | — |
| Feature scaling | StandardScaler (train-only fit) | — |
| Class imbalance | **Class Weighting** `{0:1, 1:6}` | Recall=0.861, F1=0.821 |
| Network architecture | Dense 512 → 256, ReLU, Dropout 0.3, Sigmoid | — |
| Training | Adam 1e-3, batch=128, early stopping | — |

## Training Pipeline

```text
DataSet.csv → prepare_data.py → train-only BoW + DNN → evaluate
```

```bash
python final_model_pipelines/prepare_data.py
python final_model_pipelines/dnn_pipeline/train_model.py
python final_model_pipelines/dnn_pipeline/evaluate_model.py
```

## Prediction Pipeline

```text
validation_pipeline (input validity + job relevance)
→ combined_text preprocessing
→ DNN binary classification (Sigmoid)
→ risk_score
→ risk_mapping layer
→ structured JSON (status: success / success_with_warning / rejection)
```

Three-tier risk labels are post-processing results, not training classes.

Pre-prediction validation is handled by `validation_pipeline.py` (see [main README](../README.md#pre-prediction-validation-pipeline)). LR / DNN are only called when input passes both `input_validator.py` and `job_description_filter.py`.

## Structured Output

Three outcome types — check `status` first:

| `status` | Model called? | Description |
|----------|---------------|-------------|
| `success` | Yes | Valid job-related input; normal prediction |
| `success_with_warning` | Yes | Incomplete job posting; prediction included with `message` warning |
| `invalid_input` / `not_job_related` | No | Rejected before inference; prediction fields are `null` |

Same response schema as LR; `model` is `"DNN"`, and `risk_score` is the Sigmoid output fake job probability.

## Backend Integration

```python
from final_model_pipelines.dnn_pipeline.predict import predict_job_posting
result = predict_job_posting("Job posting text...")
```
