# Linear SVM Baseline

This baseline uses TF-IDF features and Linear SVM.

It uses three-fold `StratifiedGroupKFold` inside the fixed Train set. The
parameter combination with the highest mean cross-validation PR-AUC is selected.
The final decision scores are converted to 0–1 fraud scores using sigmoid
calibration with group-aware folds from Train only. The temporary binary
threshold is selected on Validation by maximising fraud-class F1.

The script does not read or evaluate the Test set.

Run from the project root:

```bash
.venv/bin/python src/models/linear_svm/train_baseline.py
```

Outputs are saved under `reports/models/linear_svm/`. The trained model is saved
locally under `artifacts/linear_svm/` and is excluded from Git.
