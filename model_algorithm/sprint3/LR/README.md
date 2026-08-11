# LR  —  seed 42
-The LR model referenced here, specifically our team's Optimized LR, will be compared against the Literature LR from the paper.
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

## Comparison with Literature LR

`comparison_summary.csv` summarises the gains of our **Optimized LR** over the literature-reported **Literature LR** on the test set. Optimized LR trades a small drop in Macro Precision for large gains in Macro Recall, Macro F1, and Accuracy.

| Model / Comparison | Macro Precision | Macro Recall | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|
| Literature LR | 98.00% | 70.00% | 78.00% | 97.00% |
| **Optimized LR** | 96.22% | 90.60% | 93.21% | 98.83% |

Paper reference: https://doi.org/10.1007/s10791-025-09502-8
