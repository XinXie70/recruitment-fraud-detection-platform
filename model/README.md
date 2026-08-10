# Model inference package (sprint3 LR + BERT FP-gate)

Installable package used by the backend:

```bash
pip install -e model
```

## Layout

- `final_model_pipelines.lr_pipeline.predict` — sprint3 LR `_predict_risk_score`
- `final_model_pipelines.bert_pipeline.predict` — sprint3 BERT `_predict_risk_score`
- `final_model_pipelines.ensemble_pipeline.fp_gate` — BERT primary + LR false-positive gate
- `final_model_pipelines.validation_pipeline` / `risk_mapping` — backend pre/post processing

Weights live under `model_algorithm/sprint3/` and are resolved at runtime.
