# Feature Extraction Method for LR

Compare four text feature extractors with the same Logistic Regression classifier on the paper-aligned seed-42 splits.

| Setting | Value |
|---|---|
| Classifier | Logistic Regression (`liblinear`, `C=1.0`, `class_weight=None`) |
| Features | TF-IDF, BoW, Word2Vec, Char TF-IDF |
| Threshold | Maximise Fraud F1 on validation; evaluate test once |
| Random seed | `42` (`PYTHONHASHSEED=42`) |

## Layout

```
Feature Extraction Method for LR/
  code/
    train_all.py              # shared helpers + train all methods + comparison chart
    train_lr_tfidf.py         # word TF-IDF + LR
    train_lr_bow.py           # Bag-of-Words + LR
    train_lr_word2vec.py      # mean Word2Vec embedding + LR
    train_lr_char_tfidf.py    # character TF-IDF + LR
  result/                     # test metrics + comparison CSV/PNG only
  weight/                     # joblib weight files only
```

## Data

- Source: `../data/splits/`
- Seed: `42`
- Size: 12,873 train / 1,431 validation / 3,576 test
- Text field: `combined_text`

## Feature settings

| Method | Extractor |
|---|---|
| TF-IDF | Word `TfidfVectorizer`, `ngram_range=(1, 2)`, `max_features=50000` |
| BoW | Word `CountVectorizer`, `ngram_range=(1, 2)`, `max_features=50000` |
| Word2Vec | gensim skip-gram (`vector_size=200`), document = mean word vectors |
| Char TF-IDF | `analyzer='char_wb'`, `ngram_range=(3, 5)`, `max_features=50000` |

## How to Run the 5 Scripts

Activate the CUDA / project environment first:

```powershell
. E:\ml\activate.ps1
cd "model_algorithm/sprint3/Feature Extraction Method for LR"
```

| Script | Purpose | How to run |
|---|---|---|
| `code/train_all.py` | Shared helpers; trains all 4 feature methods in one go and rebuilds the comparison CSV/PNG | `python code/train_all.py` |
| `code/train_lr_tfidf.py` | Train LR with word-level TF-IDF features only | `python code/train_lr_tfidf.py` |
| `code/train_lr_bow.py` | Train LR with Bag-of-Words features only | `python code/train_lr_bow.py` |
| `code/train_lr_word2vec.py` | Train LR with mean Word2Vec embeddings only (needs `gensim`) | `python code/train_lr_word2vec.py` |
| `code/train_lr_char_tfidf.py` | Train LR with character TF-IDF features only | `python code/train_lr_char_tfidf.py` |

### 1) `train_all.py` — run everything (recommended)

```powershell
python code/train_all.py
```

This trains TF-IDF, BoW, Word2Vec, and Char TF-IDF sequentially, then writes `result/comparison_test.csv` and `result/fraud_metrics_comparison.png`.

Optional variants:

```powershell
# Train only selected methods
python code/train_all.py tfidf
python code/train_all.py bow word2vec

# Rebuild the comparison chart from existing metrics (no retraining)
python code/train_all.py --plot-only
```

### 2–5) Single-method scripts

Run one feature extractor at a time from the experiment folder:

```powershell
python code/train_lr_tfidf.py
python code/train_lr_bow.py
python code/train_lr_word2vec.py
python code/train_lr_char_tfidf.py
```

Or from inside `code/`:

```powershell
cd code
python train_lr_tfidf.py
python train_lr_bow.py
python train_lr_word2vec.py
python train_lr_char_tfidf.py
```

Each single-method script calls `train_all.run_method(...)`, so outputs still go to the same `result/` and `weight/` folders.

## Outputs

- `result/test_metrics_<method>.json` — test Fraud Precision / Recall / F1 (plus threshold & confusion matrix)
- `result/comparison_test.csv` — four-method comparison table
- `result/fraud_metrics_comparison.png` — grouped bar chart of Fraud Precision / Recall / F1
- `weight/lr_<method>.joblib` — fitted pipeline + validation-selected threshold
