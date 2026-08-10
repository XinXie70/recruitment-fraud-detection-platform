# Optimized BERT

## Directory Overview

### `code/` — Source Code

- **`run.py` (main entry point)**: Unified launcher for the BERT experiments; running it directly without arguments defaults to `train`, and its subcommands also cover training, evaluation, inference, and validation-score export.
- **`config.py`**: Centralises paths, constants, hyper-parameter configs, and runtime utilities (random seed, CUDA checks, logging, etc.).
- **`data_metrics.py`**: Handles text preprocessing, dataset loading, fraud-metric computation, threshold selection, and result visualisation.
- **`training.py`**: Implements the BERT model wrapper, train/eval loops, and the business logic behind the four `run.py` subcommands.

#### Four `run.py` Commands

Activate the CUDA environment first, then enter the `code/` directory:

```powershell
. E:\ml\activate.ps1
cd model_algorithm/sprint3/BERT/code
```

| Command | Purpose |
|---|---|
| `python run.py train` | Fine-tune BERT on the fixed train/val/test splits and save weights plus test metrics |
| `python run.py evaluate` | Load existing weights and evaluate on validation and/or test (common usage: `evaluate --split test`) |
| `python run.py predict --text "..."` | Run inference on a single job-ad text (or batch prediction with `--csv`) |
| `python run.py export-val` | Export validation fraud scores for the downstream ensemble / FP-gate |

Running `python run.py` with no arguments defaults to `python run.py train`.

Common examples:

```bash
python run.py train
python run.py evaluate --split test
python run.py predict --text "Urgent hiring, work from home, send fee first."
python run.py export-val
```

### `results/` — Experiment Outputs

`results/` stores experiment outputs such as evaluation metrics, predictions, error analysis, and figures (model weights are not stored here).

### `weight/` — Model Weights

`weight/` stores the trained model weights and threshold files (the formal checkpoint lives in `weight/best/`).

### `comparison_summary.csv` — Comparison Summary

`comparison_summary.csv` summarises the gains of our **Optimized BERT** over the paper-reported **Paper BERT** on key test metrics (e.g. test Fraud F1: 0.9053 vs 0.8802); see `comparison_summary.md` for a readable version.  
Paper reference: https://doi.org/10.1007/s10791-025-09502-8
