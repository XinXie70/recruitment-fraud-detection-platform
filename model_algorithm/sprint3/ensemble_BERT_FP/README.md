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

