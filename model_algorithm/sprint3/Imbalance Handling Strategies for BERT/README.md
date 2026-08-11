# Imbalance Handling Strategies for BERT

Compare two train-only imbalance strategies on the shared paper-aligned splits
under `../data/splits/`, using **default BERT fine-tune hyperparameters** from
`../BERT/code` (`BertFinetuneConfig`: lr=`2e-5`, epochs=`5`, `max_length=384`,
batch=`4`, accum=`4`, …).

## Layout

| Path | Purpose |
|---|---|
| `code/` | Runnable training scripts |
| `weight/` | Saved checkpoints (`class_weight/best`, `smote/best`) |
| `result/` | Per-strategy metrics + test-only comparison table |

## Scripts

| File | Strategy |
|---|---|
| `code/train_class_weight_bert.py` | Inverse-frequency **class weight** (no SMOTE) |
| `code/train_smote_bert.py` | **SMOTE** on train TF-IDF → nearest-text map, then fine-tune (no class weight) |
| `code/common.py` | Shared loaders / SMOTE helper / comparison writer |

Open either train script in VSCODE and click **Run Python File** (use the CUDA
env: `E:\ml\venv\Scripts\python.exe`).

```powershell
. E:\ml\activate.ps1
cd "model_algorithm/sprint3/Imbalance Handling Strategies for BERT/code"
python train_class_weight_bert.py
python train_smote_bert.py
```

## Figures

Each strategy keeps only three evaluation plots under `result/<strategy>/figures/`:

- `confusion_matrix_<strategy>.png`
- `precision_recall_curve_<strategy>.png`
- `roc_curve_<strategy>.png`

Other training plots (class distribution, training loss, model comparison) are
removed after each run.

## Test comparison

`result/comparison_test.csv` keeps only:

- Accuracy
- Fraud Precision
- Fraud Recall
- Fraud F1

Weights are retained under `weight/<strategy>/best/`.
