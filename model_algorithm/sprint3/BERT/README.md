# Optimized BERT — seed 42 with `max_length=512`

Settings:

- Model label: **Optimized BERT**
- 17,880 EMSCAD records with seed 42
- Original inverse-frequency class weights
- Learning rate `5e-5` and 3 epochs
- Threshold selected by maximising Fraud F0.5 on validation data
- Effective batch size 16; batch size 8 with two accumulation steps on an RTX 3060 to avoid out-of-memory errors

Shared splits: `../data/splits/`  
Checkpoint: `weight/best/`  
Summary: `comparison_summary.csv` / `comparison_summary.md`

## Run

Code is under `code/`:

- `run.py` — VS Code / Cursor “Run Python File” entry (defaults to `train`)
- `bert.py` — CLI entrypoint
- `config.py` — paths (`weight/`, `results/`), constants, utilities
- `data_metrics.py` — preprocessing, datasets, metrics, plots
- `training.py` — model, train/eval loops, CLI command handlers
- `smoke_train.py` — short train smoke test (tiny subset, does not overwrite `weight/`)

Artifacts write to `weight/` and `results/` (including `results/figures/`).

From `sprint3/`, activate the CUDA environment and run:

```bash
python BERT/code/run.py train
python BERT/code/run.py evaluate
python BERT/code/run.py predict --text "..."
python BERT/code/run.py export-val
python BERT/code/smoke_train.py
```

Equivalent: `python BERT/code/bert.py <subcommand> ...`

| Subcommand | Purpose |
|---|---|
| `train` | Fine-tune on fixed train/val/test splits; select threshold on validation (`max_fbeta`, β=0.5) |
| `evaluate` | Score validation and/or test using `weight/best` (default `--split both`, `max_length` 512) |
| `predict` | Single-text or CSV inference |
| `export-val` | Write validation fraud scores for the FP-gate ensemble |

## Results naming

Evaluate writes short, stable filenames, for example:

- `results/metrics_validation.json` / `results/metrics_test.json` / `results/metrics_summary.json`
- `results/predictions_validation.csv` / `results/predictions_test.csv`
- `results/error_analysis_validation.csv` / `results/error_analysis_test.csv`
- `results/validation_predictions.csv` (plus legacy alias `bert_validation_predictions.csv`)

Weights go to `weight/`, metrics/plots to `results/`.

`export-val` refreshes the ensemble copy under `../ensemble_BERT_FP/results/` when that directory exists.
