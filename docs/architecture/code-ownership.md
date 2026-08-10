# Code ownership and runtime boundaries

## Product runtime

- `frontend/`: React web client.
- `backend/`: authenticated FastAPI application and shared input validation.
- `model_service/`: production BERT + LR FP-gate inference service.
- `data/`: fixed experiment inputs.

## Production model assets

- `model_algorithm/sprint3/BERT/`: BERT max-length-512 checkpoint and reproducibility material.
- `model_algorithm/sprint3/LR/`: LR bigram configuration and artifact location.
- `model_algorithm/sprint3/ensemble_BERT_FP/`: FP-gate configuration.
- `model_algorithm/sprint3/risk_level/`: frozen risk-boundary configuration.

The dependency direction is:

```text
frontend -> backend -> HTTPS model API -> frozen FP-gate assets
                    -> PostgreSQL
```

FastAPI must not import model training code or load model weights. The model
service boundary is mandatory and configured through `MODEL_SERVER_URL`.
