# BERT + LR(None/bigram/no-CV) FP-gate

Optimized BERT (`max_length=512`) is primary. When BERT predicts fraud and the LR
score is below the gate threshold, the decision is flipped to legitimate (FP gate).

## Components

| Branch | Path |
|---|---|
| LR | `../LR` (`class_weight=None`, bigram, no CV) |
| BERT | `../BERT` (Optimized BERT) |
| Ensemble | This directory (FP-gate) |

The test report compares only these three arms: LR / BERT / Ensemble.

## Reproduce

From `sprint3/`, activate the CUDA environment and run:

```powershell
. E:\ml\activate.ps1

# 1) BERT: export validation scores (skip if frozen files already exist)
python BERT/code/run.py export-val

# 2) LR: reproduce predictions + metrics (skip if already present)
python LR/code/train_lr_none_bigram_no_cv.py

# 3) FP-gate ensemble
python ensemble_BERT_FP/code/run_fp_gate_ensemble.py
```

Required artifacts (short names preferred; legacy long names still accepted):

- `../BERT/results/validation_predictions.csv` (or `bert_validation_predictions.csv`)
- `../BERT/results/predictions_test.csv`
- `../BERT/results/metrics_test.json`
- `../LR/results/validation_predictions.csv`
- `../LR/results/test_predictions.csv`
- `../LR/results/test_metrics.json`

Risk-score / risk-level docs live under sibling `../risk_score/` and `../risk_level/` when present.
