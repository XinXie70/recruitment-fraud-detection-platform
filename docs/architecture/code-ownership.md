# Code ownership and runtime boundaries

This repository contains product code, reproducible research code, and retained
legacy implementations. Keeping those responsibilities explicit prevents a
deployment from accidentally using an experiment or an obsolete API.

## Product runtime

These directories form the supported web application:

- `frontend/`: React web client.
- `backend/`: authenticated FastAPI application, persistence, administration,
  analysis orchestration, XAI, and educational guidance.
- `model/final_model_pipelines/`: model adapters and artifacts used by the
  application backend when local inference is enabled.

The supported product backend entry point is `backend.main:app`.

## Reproducible reference implementation

- `src/api/`: locked LR/BERT model API.
- `src/models/`: training and evaluation scripts for the locked reference API.
- `data/`, `model_weights/`, `model_results/`, and `reports/models/`: fixed
  experiment inputs and outputs.

The reference API is maintained for reproducibility. It is not interchangeable
with the authenticated application backend.

## Legacy implementation

- `api_flask/`: retained FP-gate Flask model service.
- `XAI&Gentle_AI/`: historical integration snapshot pending archive after its
  remaining differences have been audited against the product directories.

Product code must not add imports from these legacy directories.

## Offline experiments

The following root directories are self-contained experiment snapshots and are
not product runtime dependencies:

- `lr_none_bigram_no_cv_paper_aligned_seed42/`
- `retrain_paper_aligned_seed42_maxlen512/`
- `ensemble_bert_fp_gate_lr_none_bigram_maxlen512/`

They will be moved under `experiments/` only after their scripts no longer rely
on repository-relative paths. Runtime logs, PID files, caches, and virtual
environments must not be committed. Metrics, configurations, predictions, and
figures required to reproduce reported results may remain versioned.

## Dependency rule

The intended dependency direction is:

```text
frontend -> backend -> model/final_model_pipelines
                    -> backend/xai_gentle
                    -> database
```

Product runtime code must not import from `api_flask/`, `XAI&Gentle_AI/`, or an
offline experiment directory. Moving a directory does not change that rule.
