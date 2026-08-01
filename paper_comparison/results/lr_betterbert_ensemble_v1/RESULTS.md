# LR + BetterBERT ensemble v1

## Method

The LR and BetterBERT scores were aligned using `record_id`. Five simple
weight combinations were compared on Validation. The tested LR weights were
0.25, 0.40, 0.50, 0.60 and 0.75. The BERT weight was the remaining proportion.
Thresholds from 0.01 to 0.99 were tested in steps of 0.01.

Fraud F1 on Validation was used to select the configuration. Test was not used
to choose the weight or threshold.

## Selected configuration

- LR weight: 0.25
- BetterBERT weight: 0.75
- Ensemble threshold: 0.87
- Validation Fraud F1: 0.8730
- Validation PR-AUC: 0.9214

## Test comparison

| Model | Fraud Precision | Fraud Recall | Fraud F1 | Macro F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Improved LR | **0.9610** | 0.8555 | **0.9052** | **0.9503** | 0.9454 | 0.9922 |
| BetterBERT | 0.9430 | **0.8613** | 0.9003 | 0.9477 | 0.9341 | 0.9921 |
| LR 25% + BetterBERT 75% | **0.9610** | 0.8555 | **0.9052** | **0.9503** | **0.9563** | **0.9937** |

The ensemble improved PR-AUC and ROC-AUC, showing better score ranking across
the Test set. At the frozen threshold, its Fraud Precision, Recall and F1 were
the same as LR. The ensemble changed 14 individual Test decisions, but the net
confusion matrix remained unchanged: TN 3,397, FP 6, FN 25 and TP 148.

This result supports the ensemble as a useful score-level experiment, but it
does not show a Test Fraud F1 improvement over the best individual model. No
further weight or threshold changes should be made using Test results.
