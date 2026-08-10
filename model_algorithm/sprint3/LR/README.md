# LR (no class weight, bigrams, no CV) — paper-aligned seed 42

| Setting | Value |
|---|---|
| Bigrams | Fixed `ngram_range=(1, 2)` |
| Class weight | `None` |
| Three-fold CV within training data | Disabled; fixed `C=1.0` |
| Threshold | Maximise Fraud F1 on validation; evaluate test once |

This experiment uses the same seed-42 split and fits TF-IDF on training data only.

## Data

- Source: `../data/splits/`
- Seed: `42`
- Size: 12,873 training / 1,431 validation / 3,576 test records
- Text field: `combined_text`

## Reproducibility

The following inputs and settings are fixed:

- Split assignments and input text files
- `random_state=42` and `PYTHONHASHSEED=42`
- `ngram_range=(1, 2)`, `C=1.0`, and `class_weight=None`
- Threshold selection on validation data and a single final test evaluation

Package versions are recorded under `reproducibility.package_versions` in
`results/config.json`.

The corresponding ensemble is in `../ensemble_BERT_FP`.

## Run

From `sprint3/`, activate the project environment and run:

```bash
python LR/code/train_lr_none_bigram_no_cv.py
```

The script writes:
- model weights to `LR/weight/`
- metrics and predictions to `LR/results/`
