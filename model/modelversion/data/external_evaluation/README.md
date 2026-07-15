# External Evaluation Dataset

This directory contains a client-requested evaluation-only dataset:

- 50 current legitimate job advertisements collected across SEEK, Lever,
  SmartRecruiters, and Greenhouse-hosted company careers pages.
- 50 AI-synthetic fraudulent job advertisements covering distinct scam scenarios.

It must never be merged into EMSCAD or used for training, feature selection, threshold tuning, prompt refinement, or model selection. It is only for a final out-of-dataset behavioural check.

`external_evaluation.csv` records a stable sample ID, binary label, source type, source URL or generator, collection date, scenario, and the exact text passed to the model. Real samples retain their public job URL. Synthetic samples use non-working `.invalid` contact domains.

Because live job pages can close or behave differently across browser sessions, `source_manifest.csv` also records each source domain, source-site URL, and a SHA-256 hash. `source_snapshots/` stores the exact real-ad text captured on the collection date. A snapshot proves what was evaluated, while the URL documents where it came from; neither should be treated as evidence that the vacancy remains open indefinitely.

The real-data source quotas are 10 SEEK, 10 Lever, 20 SmartRecruiters, and 10
Greenhouse. Collection also caps the number selected from one employer. The source
mix is intentional: it reduces dependence on the wording and page conventions of
any one job platform. LinkedIn and Indeed pages are not used when their full job
description requires a signed-in session or blocks reproducible collection.

Evaluation command (run from the repository root):

```bash
python3 model/modelversion/evaluate_external_dataset.py
```

The evaluator validates required columns, binary labels, unique sample IDs, and
non-empty text before running inference. It writes per-sample predictions and
model-level metrics to `model/modelversion/external_evaluation_outputs/`.
