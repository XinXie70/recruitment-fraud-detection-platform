#SVM for EMSCAD data.
from __future__ import annotations
import json, os, random, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.svm import LinearSVC
os.environ.setdefault("PYTHONHASHSEED", "42")
PROJECT_PATH = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_PATH.parent.parent / "sprint1" / "data" / "splits"
OUTPUT_PATH = PROJECT_PATH / "result"
MODEL_PATH = PROJECT_PATH / "weight"
RANDOM_SEED = 42
MAX_FEATURES = 50_000

def initialize_random_state(seed: int = RANDOM_SEED) -> None:
    random.seed(seed); np.random.seed(seed)
def read_data(split_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(SPLIT_PATH / f"{split_name}.csv", usecols=["record_id", "label", "combined_text"])
    dataframe["combined_text"] = dataframe["combined_text"].fillna("").astype(str)
    return dataframe
def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, min_df=2, max_df=0.98, max_features=MAX_FEATURES, sublinear_tf=True, ngram_range=(1, 2))
def create_model() -> LinearSVC:
    return LinearSVC(C=1.0, class_weight=None, max_iter=5000, random_state=RANDOM_SEED, dual="auto")

def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision_values, recall_values, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.0
    f1_values = 2 * precision_values[:-1] * recall_values[:-1] / np.maximum(precision_values[:-1] + recall_values[:-1], 1e-12)
    best_index = np.flatnonzero(np.isclose(f1_values, np.max(f1_values)))[0]
    return float(thresholds[best_index])

def calculate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    return {
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fraud_precision": float(precision_score(labels, predictions, zero_division=0)),
    }

def prepare_features(vectorizer: TfidfVectorizer, training_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame):
    training_features = vectorizer.fit_transform(training_data["combined_text"])
    validation_features = vectorizer.transform(validation_data["combined_text"])
    test_features = vectorizer.transform(test_data["combined_text"])
    return training_features, validation_features, test_features

def save_outputs(vectorizer: TfidfVectorizer, model: LinearSVC, threshold: float, metrics: dict) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checkpoint = {"vectorizer": vectorizer, "model": model, "threshold": float(threshold)}
    joblib.dump(checkpoint, MODEL_PATH / "svm_tfidf.joblib")
def main() -> None:
    initialize_random_state()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    training_data, validation_data, test_data = read_data("train"), read_data("validation"), read_data("test")
    vectorizer = create_vectorizer()
    training_features, validation_features, test_features = prepare_features(vectorizer, training_data, validation_data, test_data)
    model = create_model()
    model.fit(training_features, training_data["label"])
    validation_scores = model.decision_function(validation_features)
    selected_threshold = choose_threshold(validation_data["label"].to_numpy(), validation_scores)
    test_scores = model.decision_function(test_features)
    test_result = calculate_metrics(test_data["label"].to_numpy(), test_scores, selected_threshold)
    save_outputs(vectorizer, model, selected_threshold, test_result)
    print(json.dumps(test_result, indent=2))
    print(f"threshold={selected_threshold:.4f}")
if __name__ == "__main__":
    main()