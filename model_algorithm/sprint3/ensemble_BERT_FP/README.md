# BERT + LR (no class weight, bigrams, no CV) FP-gate

BERT with `max_length=512` is the primary model. When BERT predicts fraud but
the LR score is below the gate threshold, the ensemble changes the decision to
legitimate to reduce false positives.

## Components

| Branch | Path |
|---|---|
| LR | `../LR` (`class_weight=None`, bigrams, no CV) |
| BERT | `../BERT` |
| Ensemble | This directory (FP-gate) |

The test report compares only these three branches: LR, BERT, and the ensemble.

## Reproduce the experiment

From the repository root, activate the project environment and run the steps in
order. Step 1 may be skipped when the frozen BERT validation predictions are
already present.

```bash
# 1. Export BERT validation scores
python model_algorithm/sprint3/BERT/code/bert.py export-val

# 2. Reproduce LR predictions and metrics
python model_algorithm/sprint3/LR/code/train_lr_none_bigram_no_cv.py

# 3. Reproduce the FP-gate ensemble
python model_algorithm/sprint3/ensemble_BERT_FP/code/run_fp_gate_ensemble.py

# 4. Select and apply risk boundaries
python model_algorithm/sprint3/risk_level/select_risk_boundaries.py \
  --mode select
python model_algorithm/sprint3/risk_level/select_risk_boundaries.py \
  --mode apply-test
```

Download the frozen dataset splits first with
`model_algorithm/sprint3/data/download_data.sh`. The LR and BERT commands above
regenerate the prediction artifacts consumed by the ensemble step, including:

- `../BERT/results/bert_validation_predictions.csv`
  (the export script also refreshes `results/bert_validation_predictions.csv` here)
- `../BERT/results/predictions_bert_paper_protocol_maxlen512.csv`
- BERT weights under
  `../BERT/weight/best/`

See [Ensemble Risk Score and Three-Level Output](../risk_score/RISK_SCORE_AND_LEVEL.md)
for the public risk-output contract.

## Three-level risk boundaries

The original FP-gate validation search determines the High parameters. A
validation trade-off search determines the Low parameter. Test data is not used
for parameter selection.

The frozen rule is:

```text
High: BERT score >= 0.30 and LR score >= 0.06
Low:  BERT score < 0.0024
Otherwise: Suspicious
```

BERT is the primary risk-scoring model. LR is used only as a false-positive
gate for High candidates and does not participate in the Low boundary.

The operational risk score is:

```text
Normally:       risk_score = BERT score
If gate fires:  risk_score = LR score
Display:        risk_score_100 = risk_score * 100
```

The risk level must still be calculated with the original BERT–LR gate rule; it
cannot be reconstructed from the mixed-source risk score alone. The operational
score is not a calibrated fraud probability. The original BERT and LR scores
are retained in the output.

Risk-score outputs under `risk_score/`:

- `RISK_SCORE_AND_LEVEL.md`
- `RISK_SCORE_REPORT.md`
- `risk_score_metrics.csv`

Risk-level outputs under `risk_level/`:

- `select_risk_boundaries.py`
- `risk_boundary_config.json`
- `RISK_BOUNDARY_REPORT.md`
- `low_boundary_tradeoff.csv`
- `low_boundary_target_comparison.csv`
- `validation_risk_levels.csv`
- `test_risk_levels.csv`
- `test_risk_level_summary.json`
