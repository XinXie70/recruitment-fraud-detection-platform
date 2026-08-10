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

## Run

Activate the project environment, then from this experiment folder:

```powershell
. E:\ml\activate.ps1
python code/train_all.py
```

Train one method only:

```powershell
python code/train_lr_tfidf.py
python code/train_lr_bow.py
python code/train_lr_word2vec.py
python code/train_lr_char_tfidf.py
```

Rebuild the comparison chart from an existing `result/comparison_test.csv` without retraining:

```powershell
python code/train_all.py --plot-only
```

## Outputs

- `result/test_metrics_<method>.json` — test Fraud Precision / Recall / F1 (plus threshold & confusion matrix)
- `result/comparison_test.csv` — four-method comparison table
- `result/fraud_metrics_comparison.png` — grouped bar chart of Fraud Precision / Recall / F1
- `weight/lr_<method>.joblib` — fitted pipeline + validation-selected threshold
