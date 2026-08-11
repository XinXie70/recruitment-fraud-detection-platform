# Data Contract v1

This document records the shared rules for our six models. It is intentionally
short and can be updated when the team agrees on a change.

## 1. Task

- Input: one complete English job advertisement
- Output label: `0 = legitimate`, `1 = fraudulent`
- Dataset: EMSCAD (`model_algorithm/sprint3/data/raw/emscad_v1.csv`), restored
  locally with `model_algorithm/sprint3/data/download_data.sh`
- Source labels: map `f` to `0` and `t` to `1`

The column `in_balanced_dataset` is not a prediction feature. It is only a marker
from the source dataset.

## 2. Text used by the models

For EMSCAD, combine these fields in the same order:

1. `title`
2. `company_profile`
3. `description`
4. `requirements`
5. `benefits`

Missing fields become empty strings. Join the non-empty fields with a newline.
Each processed row receives an ID based on its original row order, for example
`emscad_00001`. The ID is used for tracking only and is not a model feature.

## 3. Basic cleaning

The shared cleaning step will:

- decode HTML entities;
- remove HTML tags;
- standardise Unicode text;
- standardise line breaks and repeated whitespace;
- remove whitespace at the beginning and end.

The shared step will not remove punctuation, numbers, URLs, email addresses or
stop words. A model can apply additional preprocessing after the shared step, but
the model-specific steps must be documented.

## 4. Data splitting

Use three splits: **70% Train, 15% Validation, and 15% Test**. This ratio was
selected through the group-aware feasibility analysis before model training.
Use random seed `42` for the final reproducible allocation. Keep the
fraud ratio approximately equal across the three sets. Exact duplicates and
identified near-duplicates must be grouped before the split so that related
advertisements cannot appear in different sets.

For exact duplicates, keep the first record in the original source order and
remove the remaining identical copies from the modelling dataset. Keep different
near-duplicate texts, but assign them the same `group_id`.

If a duplicate group contains conflicting labels, review it before splitting.

The fixed files are generated locally under `model_algorithm/sprint3/data/splits/`.
All six models must use these same files. Full datasets and splits are excluded
from Git; their known SHA-256 digests are recorded in
`model_algorithm/sprint3/data/checksums.sha256`.

## 5. Avoiding data leakage

- Fit TF-IDF, vocabulary, scalers and feature selection using train only.
- Use validation for model and hyperparameter selection.
- Use validation for model selection, early stopping, and threshold selection.
- Use test once for final evaluation.
- Do not use the external dataset for training or tuning.

The current synthetic external dataset is only a preliminary stress test because
its length, templates and sources differ from EMSCAD.

## 6. Shared prediction output

Each model should produce these fields:

| Field | Meaning |
|---|---|
| `record_id` | ID of the advertisement |
| `model_name` | Model used for prediction |
| `fraud_score` | Score between 0 and 1; higher means more likely fraudulent |
| `threshold` | Threshold chosen on the validation set |
| `prediction` | `0` or `1` |
| `true_label` | Actual label for offline evaluation |

Ensemble results must be joined using `record_id`, not CSV row position.

## 7. Evaluation

Report at least precision, recall, F1, PR-AUC, ROC-AUC and a confusion matrix.
Accuracy should not be used alone because the dataset is highly imbalanced.
