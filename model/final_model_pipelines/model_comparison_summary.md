# LR and DNN Final Pipeline Summary

## Architecture

```
Job posting text (combined_text)
    ↓
Binary classifier (LR / DNN)  ——  training target: real vs fake
    ↓
risk_score                  ——  raw model output (fake job probability)
    ↓
risk_mapping layer          ——  post-processing, not a training class
    ↓
Likely Legitimate / Suspicious / Likely Deceptive  ——  frontend display
```

- **Primary evaluation metrics**: binary fraud class Precision / Recall / F1 (decision boundary = LOW_THRESHOLD)
- **Three-tier statistics**: supplementary analysis; see `tier_statistics.json`

## Final Configuration

| | Logistic Regression | DNN |
|--|---------------------|-----|
| Features | TF-IDF | BoW + StandardScaler |
| Imbalance handling | Class Weighting | Class Weighting |
| Architecture | lbfgs LR | Dense 512→256, Sigmoid |
| Experiment Test Recall | 0.786 | **0.861** |
| Experiment Test F1 | 0.791 | **0.821** |

## Deployment Recommendations

- **Low-latency API**: LR
- **High-recall screening**: DNN
- **Dashboard**: `predict_all.py`

## How to Run

```bash
python final_model_pipelines/prepare_data.py   # clean + split (self-contained)
python final_model_pipelines/lr_pipeline/train_model.py
python final_model_pipelines/lr_pipeline/evaluate_model.py
python final_model_pipelines/dnn_pipeline/train_model.py
python final_model_pipelines/dnn_pipeline/evaluate_model.py
```

```python
from final_model_pipelines.predict_all import predict_with_all_models
```
