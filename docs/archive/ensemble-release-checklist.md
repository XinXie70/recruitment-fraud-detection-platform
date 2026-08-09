# Ensemble release checklist (Archived)

The checked-in ensemble configuration is currently an explicitly unfitted
development fallback. Do not present its equal weights as validation-optimised.

## Why the existing prediction cache is not accepted

The legacy `_val_predictions.npz` contains 1,788 rows and has no `record_id` or
validation-file fingerprint. The current leakage-resistant fixed validation
split contains 2,371 rows. Joining or truncating by row position would violate
the data contract and could silently associate scores with the wrong labels.

The fitting script now rejects any cache unless all of these match exactly:

- validation CSV SHA-256;
- ordered `record_id` values;
- ordered labels;
- one valid probability per row for every configured model.

## Generate and fit on the model host

Run this on the Compute Engine model VM, where the complete local model
artifacts are installed. Do not set `MODEL_SERVER_URL`; the public interactive
API rejects inputs that do not pass product-level relevance validation, while
offline evaluation must score every row in the fixed validation split.

```bash
git fetch origin
git switch improve/final-demo-readiness
git pull --ff-only
unset MODEL_SERVER_URL
python model/final_model_pipelines/ensemble_pipeline/fit_ensemble_safe.py \
  --refresh-predictions --batch-size 64
```

The command processes models sequentially, caches predictions with provenance,
creates a group-disjoint calibration/tuning split, fits calibrators and weights,
and writes `ensemble_config.json`.

## Review before committing

- [ ] `fitted` is `true`.
- [ ] `validation_split_sha256` matches `data/splits/validation.csv`.
- [ ] Model weights sum to 1 within floating-point tolerance.
- [ ] Low threshold is below high threshold.
- [ ] No test-set file was read during fitting.
- [ ] Calibration and tuning subsets share no `group_id`.
- [ ] Record the command, commit, random seed, row counts, and metrics.
- [ ] Run backend tests with the new configuration.
- [ ] Perform a deployment smoke test and retain the result as Demo evidence.

## Production gate

The fitted artifact is necessary but not sufficient for release. Before calling
it production-ready, compare it with the equal-weight baseline on the held-out
test set exactly once, publish precision/recall/F1/PR-AUC and tier statistics,
and retain the predictions joined by `record_id` for auditability.
