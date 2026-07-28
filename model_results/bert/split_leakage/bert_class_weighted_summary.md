# Split Leakage Experiment: BERT (class-weighted)

## Design

- Model: **BERT fine-tune with train-only class weights**
- Conditions: A / B / C from `data/experiment_splits/`
- Holdout = diagnostic evaluation partition (mapped to BERT `test`)
- Seeds present in results: **0, 1, 2, 3, 4, 5, 6, 7, 8, 9**

## Mean diagnostic Holdout metrics

| Condition | PR-AUC | ROC-AUC | Fraud F1 | Fraud Precision | Fraud Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.9016 | 0.9817 | 0.8702 | 0.9205 | 0.8262 | 0.9320 |
| B | 0.8781 | 0.9779 | 0.8470 | 0.9179 | 0.7879 | 0.9202 |
| C | 0.8507 | 0.9689 | 0.8174 | 0.8863 | 0.7596 | 0.9047 |

## Deltas

- B − A PR-AUC: **-0.0235**
- C − B PR-AUC: **-0.0274**
- B − A Fraud F1: **-0.0231**
- C − B Fraud F1: **-0.0296**

## Artifact locations

- Weights: `F:/final-version-1/capstone-project-26t2-9900-h09c-almond/model_weights/bert/split_leakage/condition_*/seed_*/best/`
- Results: `F:/final-version-1/capstone-project-26t2-9900-h09c-almond/model_results/bert/split_leakage/`
