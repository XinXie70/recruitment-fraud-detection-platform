# Model Comparison Summary

## Architecture

```
Job posting text (combined_text)
    ↓
Binary classifier (LR / SVM / XGBoost / DNN / RNN / Bi-LSTM / BERT / RoBERTa)  ——  training target: real vs fake
    ↓
risk_score                  ——  raw model output (fake job probability)
    ↓
risk_mapping layer          ——  post-processing, not a training class
    ↓
Likely Legitimate / Suspicious / Likely Deceptive  ——  frontend display
```

- **Primary evaluation metrics**: binary fraud class Precision / Recall / F1 (decision boundary = LOW_THRESHOLD)
- **Three-tier statistics**: supplementary analysis; see each pipeline's `outputs/tier_statistics.json`
- **Unified comparison**: run `compare_all_models.py` after evaluating all pipelines

## Model Configuration

| Model | Features | Imbalance handling | Architecture |
|-------|----------|-------------------|--------------|
| Logistic Regression | TF-IDF | Class Weighting | lbfgs LR |
| SVM | TF-IDF | Class Weighting | Linear SVC (probability=True) |
| XGBoost | TF-IDF | scale_pos_weight=6 | Gradient boosted trees |
| DNN | BoW + StandardScaler | Class Weighting | Dense 512→256, Sigmoid |
| RNN | Tokenized Sequences | Class Weighting | Embedding → LSTM → Sigmoid |
| Bi-LSTM | Tokenized Sequences | Class Weighting | Embedding → Bi-LSTM → Sigmoid |
| BERT | WordPiece (max_len=256) | Class Weighting | bert-base-uncased fine-tune → Softmax |
| RoBERTa | BPE (max_len=256) | Class Weighting | roberta-base fine-tune → Softmax |

All models share:
- Shared text cleaning (`text_utils.py`) and stratified train/val/test split (`data_split.py`)
- Validation-set dual-threshold tuning (`structured_output.tune_dual_thresholds`)
- Same binary metrics and risk mapping post-processing (`evaluation_utils.run_full_evaluation`)

## How to Run

```bash
python final_model_pipelines/prepare_data.py   # clean + split (self-contained)

# Train and evaluate each model
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
python final_model_pipelines/bert_pipeline/train_model.py
python final_model_pipelines/bert_pipeline/evaluate_model.py
python final_model_pipelines/roberta_pipeline/train_model.py
python final_model_pipelines/roberta_pipeline/evaluate_model.py

# Aggregate Test-set comparison
python final_model_pipelines/compare_all_models.py
```

```python
from final_model_pipelines.predict_all import predict_with_all_models
```

## Deployment Recommendations

- **Low-latency API**: Logistic Regression or SVM
- **Tabular / tree-based baseline**: XGBoost
- **High-recall screening**: DNN, sequence models (RNN / Bi-LSTM), or Transformers (BERT / RoBERTa)
- **Transformer baselines** (GPU preferred): BERT / RoBERTa
- **Dashboard side-by-side comparison**: `predict_all.py` or `compare_all_models.py`
