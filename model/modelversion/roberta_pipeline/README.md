# RoBERTa Final Pipeline

**Text-only** fake job detection via fine-tuned `roberta-base` (Hugging Face Transformers + PyTorch).

## Final Configuration

| Setting | Choice |
|---------|--------|
| Text preprocessing | HTML cleaning + five fields → `combined_text` |
| Encoder | **roberta-base** |
| Max sequence length | 256 |
| Class imbalance | **Class Weighting** `{0:1, 1:6}` |
| Classifier head | 2-way softmax (`real` / `fake`) |
| Training | AdamW 2e-5, batch=8, up to 3 epochs, early stopping |

## Training Pipeline

```bash
python final_model_pipelines/prepare_data.py
python final_model_pipelines/roberta_pipeline/train_model.py
python final_model_pipelines/roberta_pipeline/evaluate_model.py
```

Install deps first:

```bash
pip install -r final_model_pipelines/roberta_pipeline/requirements.txt
```

## Prediction Pipeline

```text
combined_text → RoBERTa logits → softmax P(fake)=risk_score → risk_mapping → structured JSON
```

## Backend Integration

```python
from final_model_pipelines.roberta_pipeline.predict import predict_job_posting
result = predict_job_posting("Job posting text...")
```
