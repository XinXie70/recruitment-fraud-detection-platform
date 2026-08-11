# BERT + LR  FP-gate

Optimized BERT is primary. When BERT predicts fraud and the LR
score is below the gate threshold, the decision is flipped to legitimate (FP gate).

The test-set three-model Macro comparison table is in `test_macro_comparison.csv`.

## Components

| Branch | Path |
|---|---|
| LR | `../LR` |
| BERT | `../BERT` |
| Ensemble | This directory (FP-gate) |

The test report compares only these three arms: LR / BERT / Ensemble.

## Code files

| Script | Purpose |
|---|---|
| `code/run_fp_gate_ensemble.py` | Main ensemble runner: search FP-gate thresholds on validation, evaluate on test, write metrics/predictions |
| `code/show_test_macro_table.py` | Read existing LR / BERT / Ensemble test metrics and print a Macro P / R / F1 comparison table |

## Prerequisites

Before running the ensemble scripts, make sure these artifacts exist
(short names preferred; legacy long names still accepted):

- `../BERT/results/validation_predictions.csv` (or `bert_validation_predictions.csv`)
- `../BERT/results/predictions_test.csv`
- `../BERT/results/metrics_test.json`
- `../LR/results/validation_predictions.csv`
- `../LR/results/test_predictions.csv`
- `../LR/results/test_metrics.json`

If they are missing, generate them first from `sprint3/`:

```powershell
. E:\ml\activate.ps1

# BERT validation scores + test evaluation
python BERT/code/run.py export-val
python BERT/code/run.py evaluate --split test

# LR predictions + metrics
python LR/code/train_lr_none_bigram_no_cv.py
```

## How to run

Activate the environment, then work from `sprint3/`:

```powershell
. E:\ml\activate.ps1
cd model_algorithm/sprint3
```

### 1) `run_fp_gate_ensemble.py` — build / refresh the FP-gate ensemble

```powershell
python ensemble_BERT_FP/code/run_fp_gate_ensemble.py
```

What it does:
1. Loads BERT and LR validation/test prediction files
2. Searches BERT threshold + LR gate on validation (FP-gate rule)
3. Applies the selected gate on the test set once
4. Writes ensemble outputs under `ensemble_BERT_FP/results/`

Main outputs:
- `results/config.json`
- `results/validation_metrics.json` / `results/test_metrics.json`
- `results/validation_predictions.csv` / `results/test_predictions.csv`
- `results/validation_sweep.csv`
- `results/RESULTS.md`

### 2) `show_test_macro_table.py` — show the three-way Macro comparison

Run this after `run_fp_gate_ensemble.py` (and after LR/BERT metrics exist):

```powershell
python ensemble_BERT_FP/code/show_test_macro_table.py
```

What it does:
- Reads:
  - `../LR/results/test_metrics.json`
  - `../BERT/results/metrics_test.json`
  - `./results/test_metrics.json`
- Prints a terminal table of Macro Precision / Recall / F1 for LR, Optimized BERT, and FP-gate Ensemble

This script does **not** retrain or re-sweep thresholds; it only displays existing results.  


## Suggested full workflow

```powershell
. E:\ml\activate.ps1
cd model_algorithm/sprint3

# if artifacts are already present
python BERT/code/run.py export-val
python BERT/code/run.py evaluate --split test
python LR/code/train_lr_none_bigram_no_cv.py

# Ensemble
python ensemble_BERT_FP/code/run_fp_gate_ensemble.py

# print Macro comparison table
python ensemble_BERT_FP/code/show_test_macro_table.py
```

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
