#XGBoost for EMSCAD data.
from __future__ import annotations
import json, os, random, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from xgboost import XGBClassifier

os.environ.setdefault("PYTHONHASHSEED", "42")
PROJECT_PATH = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_PATH.parent.parent / "sprint1" / "data" / "splits"
OUTPUT_PATH = PROJECT_PATH / "result"
MODEL_PATH = PROJECT_PATH / "weight"
RANDOM_SEED = 42
MAX_FEATURES = 20_000

def initialize_random_state(seed: int = RANDOM_SEED) -> None:
    random.seed(seed); np.random.seed(seed)
def read_data(split_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(SPLIT_PATH / f"{split_name}.csv", usecols=["record_id", "label", "combined_text"])
    dataframe["combined_text"] = dataframe["combined_text"].fillna("").astype(str)
    return dataframe
def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision_values, recall_values, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1_values = 2 * precision_values[:-1] * recall_values[:-1] / np.maximum(precision_values[:-1] + recall_values[:-1], 1e-12)
    best_index = np.flatnonzero(np.isclose(f1_values, np.max(f1_values)))[0]
    return float(thresholds[best_index])

def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fraud_precision": float(precision_score(labels, predictions, zero_division=0)),
    }
def calculate_class_weight(labels: np.ndarray) -> float:
    positive_count = max(int(labels.sum()), 1)
    negative_count = int(len(labels) - positive_count)
    return negative_count / positive_count

class FraudXGBoost:
    def __init__(self, scale_pos_weight: float) -> None:
        self.vectorizer = TfidfVectorizer(lowercase=True, min_df=2, max_df=0.98, max_features=MAX_FEATURES, sublinear_tf=True, ngram_range=(1, 2))
        self.classifier = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, n_jobs=4, tree_method="hist", device="cpu")
    def fit(self, training_data: pd.DataFrame, validation_data: pd.DataFrame) -> None:
        print("Fitting TF-IDF...", flush=True)
        training_features = self.vectorizer.fit_transform(training_data["combined_text"])
        validation_features = self.vectorizer.transform(validation_data["combined_text"])
        print(f"TF-IDF shape: {training_features.shape}", flush=True)
        print("Fitting XGBoost...", flush=True)
        self.classifier.fit(training_features, training_data["label"].to_numpy(), eval_set=[(validation_features, validation_data["label"].to_numpy())], verbose=False)
    def predict_scores(self, dataframe: pd.DataFrame) -> np.ndarray:
        features = self.vectorizer.transform(dataframe["combined_text"])
        return self.classifier.predict_proba(features)[:, 1]

def save_outputs(model: FraudXGBoost, threshold: float, metrics: dict) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checkpoint = {"vectorizer": model.vectorizer, "model": model.classifier, "threshold": float(threshold)}
    joblib.dump(checkpoint, MODEL_PATH / "xgboost_tfidf.joblib")

def main() -> None:
    initialize_random_state()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    training_data, validation_data, test_data = read_data("train"), read_data("validation"), read_data("test")
    training_labels = training_data["label"].to_numpy()
    scale_pos_weight = calculate_class_weight(training_labels)
    model = FraudXGBoost(scale_pos_weight)
    model.fit(training_data, validation_data)
    print("Predicting...", flush=True)
    validation_scores = model.predict_scores(validation_data)
    selected_threshold = choose_threshold(validation_data["label"].to_numpy(), validation_scores)
    test_scores = model.predict_scores(test_data)
    test_result = calculate_metrics(test_data["label"].to_numpy(), test_scores, selected_threshold)
    save_outputs(model, selected_threshold, test_result)
    print(json.dumps(test_result, indent=2))
    print(f"threshold={selected_threshold:.4f} scale_pos_weight={scale_pos_weight:.2f}")
if __name__ == "__main__":
    main()