# BERT — paper-aligned seed 42 with `max_length=512`

Settings:

- 17,880 EMSCAD records with seed 42
- Original inverse-frequency class weights
- Learning rate `5e-5` and 3 epochs
- Threshold selected by maximising Fraud F0.5 on validation data
- Effective batch size 16; batch size 8 with two accumulation steps on an
  RTX 3060 to avoid out-of-memory errors

Shared splits: `../data/splits/`  
Checkpoint: `weight/best/`

## Run

All entrypoints live in a single file. From `sprint3/`, activate the project
environment and run:

```bash
python BERT/code/bert.py train
python BERT/code/bert.py evaluate
python BERT/code/bert.py predict --text "..."
python BERT/code/bert.py export-val
```

| Subcommand | Purpose |
|---|---|
| `train` | Fine-tune on fixed train/val/test splits; select threshold on validation (`max_fbeta`, β=0.5) |
| `evaluate` | Score the test set using `weight/best` (`max_length` default 512) |
| `predict` | Single-text or CSV inference |
| `export-val` | Write validation fraud scores for the FP-gate ensemble |

Weights go to `weight/`, metrics to `results/`.

`export-val` writes `results/bert_validation_predictions.csv` and refreshes the
copy under `../ensemble_BERT_FP/results/` when that directory exists.
