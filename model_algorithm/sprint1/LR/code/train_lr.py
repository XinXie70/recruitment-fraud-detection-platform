#LR for EMSCAD data.
from __future__ import annotations
import json, os, random, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,precision_recall_curve,precision_score,recall_score,
)
from sklearn.pipeline import Pipeline

os.environ.setdefault("PYTHONHASHSEED", "42")
BASE_PATH = Path(__file__).resolve().parents[1]
SPLIT_PATH = BASE_PATH.parent / "data" / "splits"
OUTPUT_PATH = BASE_PATH / "result"
MODEL_PATH = BASE_PATH / "weight"
RANDOM_SEED = 42
MAX_VOCABULARY = 50_000


def initialize_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
def read_split(split_name: str) -> pd.DataFrame:
    csv_path = SPLIT_PATH / f"{split_name}.csv"
    dataframe = pd.read_csv(
        csv_path,
        usecols=["record_id","label","combined_text",],
    )

    dataframe["combined_text"] = (
        dataframe["combined_text"].fillna("").astype(str)
    )
    return dataframe

def build_classifier() -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,max_df=0.98,
        max_features=MAX_VOCABULARY,
        sublinear_tf=True,ngram_range=(1, 2),
    )
    classifier = LogisticRegression(
        solver="liblinear",max_iter=1000,
        random_state=RANDOM_SEED,C=1.0,class_weight=None,
    )
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),("classifier", classifier),]
    )


def calculate_best_threshold(
    labels: np.ndarray,probabilities: np.ndarray,
) -> float:
    precision_values, recall_values, thresholds = precision_recall_curve(
        labels,probabilities,
    )
    if thresholds.size == 0:
        return 0.5
    numerator = (
        2.0* precision_values[:-1]* recall_values[:-1]
    )
    denominator = np.maximum(
        precision_values[:-1] + recall_values[:-1],1e-12,
    )

    f1_values = numerator / denominator
    highest_f1 = np.max(f1_values)
    best_indices = np.flatnonzero(
        np.isclose(f1_values,highest_f1,)
    )
    best_index = int(best_indices[0])
    return float(
        thresholds[best_index]
    )

def evaluate_predictions(
    labels: np.ndarray,probabilities: np.ndarray,threshold: float,
) -> dict:
    predictions = np.where(
        probabilities >= threshold, 1,0,
    )
    precision = precision_score(
        labels,predictions,zero_division=0,
    )

    recall = recall_score(
        labels,predictions,zero_division=0,
    )

    f1 = f1_score(
        labels,predictions,zero_division=0,
    )
    return {
        "fraud_precision": float(precision),
        "fraud_recall": float(recall),
        "fraud_f1": float(f1),
        "threshold": float(threshold),
    }

def train_model(
    train_data: pd.DataFrame,
) -> Pipeline:
    model = build_classifier()
    model.fit(
        train_data["combined_text"],train_data["label"],
    )
    return model

def get_fraud_probabilities(
    model: Pipeline,dataframe: pd.DataFrame,
) -> np.ndarray:
    probability_matrix = model.predict_proba(
        dataframe["combined_text"]
    )
    return probability_matrix[:, 1]

def prepare_directories() -> None:
    OUTPUT_PATH.mkdir(
        parents=True,exist_ok=True,
    )
    MODEL_PATH.mkdir(
        parents=True,exist_ok=True,
    )

def save_result(
    metrics: dict,
) -> Path:
    result = {
        "fraud_f1": metrics["fraud_f1"],
        "fraud_recall": metrics["fraud_recall"],
        "fraud_precision": metrics["fraud_precision"],
    }
    result_path = (
        OUTPUT_PATH
        / "test_metrics.json"
    )
    result_path.write_text(
        json.dumps(result,indent=2,
        ),
        encoding="utf-8",
    )
    return result_path

def save_model(
    model: Pipeline,
    threshold: float,
) -> Path:
    model_file = (
        MODEL_PATH
        / "lr_tfidf.joblib"
    )
    model_package = {
        "model": model,"threshold": float(threshold),
    }
    joblib.dump(
        model_package,model_file,
    )
    return model_file
def main() -> None:
    initialize_seed()
    prepare_directories()
    train_data = read_split("train")
    validation_data = read_split("validation")
    test_data = read_split("test")
    classifier = train_model(train_data)
    validation_probabilities = get_fraud_probabilities(
        classifier,validation_data,
    )
    validation_labels = (
        validation_data["label"].to_numpy()
    )
    decision_threshold = calculate_best_threshold(
        validation_labels,validation_probabilities,
    )
    test_probabilities = get_fraud_probabilities(
        classifier,test_data,
    )
    test_labels = (
        test_data["label"].to_numpy()
    )
    test_metrics = evaluate_predictions(
        test_labels,test_probabilities,decision_threshold,
    )
    result = {
        "fraud_f1": test_metrics["fraud_f1"],
        "fraud_recall": test_metrics["fraud_recall"],
        "fraud_precision": test_metrics["fraud_precision"],
    }
    result_file = save_result(
        test_metrics
    )
    model_file = save_model(
        classifier,decision_threshold,
    )
    print(
        json.dumps(
            result,indent=2,
        )
    )
    print(
        f"Saved weight: {model_file}"
    )
    print(
        f"Saved result: {result_file}"
    )
    print(
        f"threshold={decision_threshold:.4f}"
    )
if __name__ == "__main__":
    main()