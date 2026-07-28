# Logistic Regression Baseline

This baseline uses TF-IDF features and Logistic Regression.

It uses three-fold `StratifiedGroupKFold` inside the fixed Train set. The
parameter combination with the highest mean cross-validation PR-AUC is fitted on
the complete Train set. The classification threshold is then selected on the
fixed Validation set by maximising fraud-class F1.

The script does not read or evaluate the Test set.

Run from the project root:

```bash
.venv/bin/python src/models/logistic_regression/train_baseline.py
```

Outputs are saved under `reports/models/logistic_regression/`. The trained
model is saved locally under `artifacts/logistic_regression/` and is excluded
from Git.
