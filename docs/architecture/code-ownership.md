# Code ownership and runtime boundaries

## Product runtime

- `frontend/`: React web client.
- `backend/`: authenticated FastAPI application and shared input validation.
- `api_flask/`: production BERT + LR FP-gate inference service.
- `data/`: fixed experiment inputs.

## Production model assets

- `retrain_paper_aligned_seed42_maxlen512/`: BERT max-length-512 checkpoint and reproducibility material.
- `lr_none_bigram_no_cv_paper_aligned_seed42/`: LR bigram configuration and artifact location.
- `ensemble_bert_fp_gate_lr_none_bigram_maxlen512/`: FP-gate and risk-boundary configuration.

The dependency direction is:

```text
frontend -> backend -> HTTPS model API -> frozen FP-gate assets
                    -> PostgreSQL
```

FastAPI must not import model training code or load model weights. The model
service boundary is mandatory and configured through `MODEL_SERVER_URL`.
