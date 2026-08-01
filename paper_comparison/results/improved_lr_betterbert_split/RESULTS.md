# Improved LR on the betterBERT split

## Purpose

This run retrains the existing improved Logistic Regression using the exact
Train, Validation and Test assignments from the team `betterBERT` branch at
commit `4c3b2b2`. It creates aligned LR scores for later ensembling with BERT.

## Protocol

- Dataset: original EMSCAD, 17,880 advertisements
- Assignment source: `betterBERT` saved `assignments.csv.gz`
- Split seed: `42`
- Train: 12,873 rows, including 624 fraud
- Validation: 1,431 rows, including 69 fraud
- Test: 3,576 rows, including 173 fraud
- LR input: shared five-field `combined_text`
- Configuration selection: three-fold CV inside Train using PR-AUC
- Threshold selection: maximum Fraud F1 on Validation
- Validation is not added back to Train, preserving clean scores for ensemble

The selected configuration was unigram plus bigram TF-IDF, `C=1.0`, balanced
class weights, and a Validation-selected threshold of `0.6395`.

## Comparison with the paper LR

| Model | Macro Precision | Macro Recall | Macro F1 | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---:|---:|---:|---:|---:|---:|
| Paper LR | **0.980** | 0.700 | 0.780 | **0.986** | 0.399 | 0.568 |
| Improved LR | 0.977 | **0.927** | **0.950** | 0.961 | **0.855** | **0.905** |

| Model | Accuracy | ROC-AUC | PR-AUC | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Paper LR | 0.971 | 0.981 | Not reported | 69 | **1** | 104 |
| Improved LR | **0.991** | **0.992** | 0.945 | **148** | 6 | **25** |

The improved LR detected 148 of 173 fraudulent advertisements, compared with
69 of 173 in the paper LR result. Fraud F1 increased from approximately 0.568
to 0.905, while Fraud Precision remained high at 0.961.

## Comparability note

This is not the paper's exact Test allocation. The published code uses split
seed `12342`; the team betterBERT split uses seed `42`. The comparison therefore
uses the same EMSCAD dataset and the same stratified 80/20 protocol, but not the
same Test advertisements. The earlier `improved_lr` experiment retains seed
`12342` for the closer paper-aligned comparison. This seed-42 run is the correct
LR component for an ensemble with betterBERT.

## Alignment with betterBERT

All 3,576 LR Test IDs match the betterBERT Test prediction IDs, and all labels
match. All 1,431 Validation IDs also match the betterBERT Validation split.

| Model on the common split | Fraud Precision | Fraud Recall | Fraud F1 | Macro F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Improved LR | **0.961** | 0.855 | **0.905** | **0.950** | **0.945** |
| Improved BERT | 0.943 | **0.861** | 0.900 | 0.948 | 0.934 |

The models are close in overall performance. Their Validation fraud scores must
be aligned by `record_id` before ensemble weights and the ensemble threshold are
selected. Test must not be used for that selection.
