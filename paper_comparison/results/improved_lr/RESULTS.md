# Improved LR result on the paper-aligned split

## Experiment setup

- Dataset: original EMSCAD, 17,880 advertisements
- Input: the 12 fields used by Taneja et al. (2025)
- Outer split: stratified random 80% Train / 20% Test
- Paper split seed: `12342`
- Development split: 64% internal Train / 16% Validation
- Model selection: three-fold CV on internal Train using PR-AUC
- Threshold selection: maximum Fraud F1 on Validation
- Final training: refit on the complete outer 80% Train
- Test: evaluated after the configuration was selected

The selected configuration was TF-IDF unigram plus bigram, `C=1.0`, and
balanced class weights. The selected Validation threshold was `0.5665`.

## Test comparison

| Model | Macro Precision | Macro Recall | Macro F1 | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---:|---:|---:|---:|---:|---:|
| Paper LR | **0.980** | 0.700 | 0.780 | **0.986** | 0.399 | 0.568 |
| Improved LR | 0.909 | **0.929** | **0.918** | 0.824 | **0.867** | **0.845** |

The paper directly reports the three Macro values, not the three fraud-class
values. The Paper LR fraud-class values are calculated separately from its
published confusion matrix: TN=3402, FP=1, FN=104, TP=69.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Paper LR | 0.981 | Not reported |
| Improved LR | **0.991** | 0.922 |

Our LR detected 150 of the 173 fraudulent advertisements and missed 23. The
paper LR detected 69 and missed 104. The improved model therefore achieved much
higher Fraud Recall and Fraud F1, while producing more false positives and lower
Fraud Precision than the paper LR.

This is a paper-aligned comparison rather than a claim that every preprocessing
detail is identical. The paper fitted TF-IDF before its split, whereas this
experiment correctly fits TF-IDF using training data only. Exact sample identity
also depends on using the same source-row order as the authors.

## Files for the later ensemble

- `split_assignments.csv`: fixed Train, Validation and Test membership
- `validation_predictions.csv`: LR Validation fraud scores
- `test_predictions.csv`: LR frozen-Test fraud scores
- `config.json`: split and selected model configuration
- `validation_metrics.json` and `test_metrics.json`: full metrics

BERT should reuse `split_assignments.csv` and output scores with the same
`record_id` values. Ensemble weights and threshold must be selected using the
Validation scores only.
