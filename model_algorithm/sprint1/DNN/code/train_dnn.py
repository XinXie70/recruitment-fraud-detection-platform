#DNN for EMSCAD data.
from __future__ import annotations
import json, os, random, joblib, numpy as np, pandas as pd, torch, torch.nn as nn; from pathlib import Path; from typing import Dict, Tuple
os.environ.setdefault("PYTHONHASHSEED", "42")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_DIR.parent / "data" / "splits"
OUTPUT_PATH = PROJECT_DIR / "result"
MODEL_PATH = PROJECT_DIR / "weight"
RANDOM_SEED = 42
VOCAB_SIZE = 30_000
HIDDEN_SIZE = 256
DROP_RATE = 0.3
TRAIN_BATCH_SIZE = 256
MAX_EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5

def initialize_random_state(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_dataset_split(split_name: str) -> pd.DataFrame:
    file_path = SPLIT_PATH / f"{split_name}.csv"
    dataframe = pd.read_csv(
        file_path,
        usecols=[
            "record_id","label","combined_text",
        ],
    )

    dataframe["combined_text"] = (
        dataframe["combined_text"]
        .fillna("") .astype(str)
    )
    return dataframe

class FraudDetectionNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        dropout_rate: float = DROP_RATE,
    ) -> None:
        super().__init__()
        second_hidden_size = hidden_size // 2
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, second_hidden_size),
            nn.ReLU(),nn.Dropout(dropout_rate),
            nn.Linear(second_hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.layers(features)
        return logits.squeeze(dim=-1)

def matrix_to_tensor(matrix) -> torch.Tensor:
    dense_matrix = (
        matrix.toarray()
        if hasattr(matrix, "toarray")
        else matrix
    )

    return torch.as_tensor(
        dense_matrix,dtype=torch.float32,
    )

@torch.no_grad()
def generate_probabilities(
    network: nn.Module,features: torch.Tensor,device: torch.device,
) -> np.ndarray:
    network.eval()

    data_loader = DataLoader(
        TensorDataset(features),batch_size=TRAIN_BATCH_SIZE,shuffle=False,
    )
    probability_batches = []
    for (feature_batch,) in data_loader:
        feature_batch = feature_batch.to(device)
        logits = network(feature_batch)
        probabilities = torch.sigmoid(logits)
        probability_batches.append(
            probabilities.cpu().numpy()
        )
    return np.concatenate(
        probability_batches,
        axis=0,
    )


def find_optimal_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    precision_values, recall_values, thresholds = precision_recall_curve(
        targets,
        probabilities,
    )

    if thresholds.size == 0:
        return 0.5

    numerator = (
        2* precision_values[:-1]* recall_values[:-1]
    )
    denominator = np.maximum(
        precision_values[:-1] + recall_values[:-1],1e-12,
    )

    f1_values = numerator / denominator
    best_f1 = np.max(f1_values)
    best_index = np.flatnonzero(
        np.isclose(
            f1_values,best_f1,
        )
    )[0]
    return float(thresholds[best_index])


def calculate_fraud_metrics(
    targets: np.ndarray,probabilities: np.ndarray,threshold: float,
) -> Dict[str, float]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    precision = precision_score(
        targets,predictions,zero_division=0,
    )

    recall = recall_score(
        targets,predictions,zero_division=0,
    )

    f1 = f1_score(
        targets,predictions,zero_division=0,
    )

    return {
        "fraud_precision": float(precision),
        "fraud_recall": float(recall),
        "fraud_f1": float(f1),
        "threshold": float(threshold),
    }

def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        min_df=2,max_df=0.98,max_features=VOCAB_SIZE,sublinear_tf=True,ngram_range=(1, 2),
    )


def prepare_features(
    train_df: pd.DataFrame,validation_df: pd.DataFrame,test_df: pd.DataFrame,
) -> Tuple[
    TfidfVectorizer,torch.Tensor,torch.Tensor,torch.Tensor,
]:
    vectorizer = create_vectorizer()
    train_matrix = vectorizer.fit_transform(
        train_df["combined_text"]
    )
    validation_matrix = vectorizer.transform(
        validation_df["combined_text"]
    )
    test_matrix = vectorizer.transform(
        test_df["combined_text"]
    )

    return (
        vectorizer,matrix_to_tensor(train_matrix),matrix_to_tensor(validation_matrix),matrix_to_tensor(test_matrix),
    )

def calculate_positive_weight(
    labels: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    positive_count = max(
        float(labels.sum()),1.0,
    )
    negative_count = (
        float(len(labels))- positive_count
    )
    weight_value = negative_count / positive_count
    return torch.tensor(
        [weight_value],dtype=torch.float32,device=device,
    )


def train_network(
    network: nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: np.ndarray,
    device: torch.device,
    positive_weight: torch.Tensor,
) -> Tuple[dict, float]:
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE,
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight,
    )
    train_loader = DataLoader(
        TensorDataset(
            train_features,train_targets,
        ),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    best_f1 = -1.0
    best_parameters = None
    patience_counter = 0
    for epoch_number in range(
        1,MAX_EPOCHS + 1,
    ):
        network.train()
        accumulated_loss = 0.0
        for feature_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(feature_batch)
            loss = loss_function(
                logits,target_batch,
            )
            loss.backward()
            optimizer.step()
            accumulated_loss += (
                float(loss.item())* len(target_batch)
            )

        validation_probabilities = generate_probabilities(
            network,
            validation_features,
            device,
        )
        current_threshold = find_optimal_threshold(
            validation_targets,validation_probabilities,
        )
        validation_metrics = calculate_fraud_metrics(
            validation_targets,validation_probabilities,current_threshold,
        )
        current_f1 = validation_metrics["fraud_f1"]
        average_loss = (
            accumulated_loss/ len(train_targets)
        )
        print(
            f"epoch={epoch_number:02d} "
            f"loss={average_loss:.4f} "
            f"val_fraud_f1={current_f1:.4f} "
            f"thr={current_threshold:.4f}"
        )
        if current_f1 > best_f1 + 1e-6:
            best_f1 = current_f1
            best_parameters = {
                key: value.detach().cpu().clone()
                for key, value
                in network.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stop at epoch {epoch_number}"
                )
                break
    if best_parameters is None:
        raise RuntimeError(
            "Training failed: no model checkpoint was created"
        )
    return best_parameters, best_f1


def save_outputs(
    vectorizer: TfidfVectorizer,model_state: dict,input_size: int,threshold: float,metrics: dict,
) -> None:
    OUTPUT_PATH.mkdir(
        parents=True,exist_ok=True,
    )
    MODEL_PATH.mkdir(
        parents=True,exist_ok=True,
    )
    metrics_file = (
        OUTPUT_PATH / "test_metrics.json"
    )

    metrics_file.write_text(
        json.dumps(
            metrics,indent=2,
        ),
        encoding="utf-8",
    )

    checkpoint = {
        "model_state_dict": model_state,
        "n_features": int(input_size),
        "hidden": HIDDEN_SIZE,
        "dropout": DROP_RATE,
        "threshold": float(threshold),
    }

    # Offline training artifact only; not loaded from untrusted input.
    torch.save(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        checkpoint, MODEL_PATH / "dnn_mlp.pt",
    )

    joblib.dump(
        vectorizer, MODEL_PATH / "tfidf.joblib",
    )


def main() -> None:
    initialize_random_state()
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    train_df = read_dataset_split("train")
    validation_df = read_dataset_split("validation")
    test_df = read_dataset_split("test")
    (
        vectorizer,train_features,validation_features,test_features,
    ) = prepare_features(
        train_df,validation_df,test_df,
    )
    train_labels = (
        train_df["label"]
        .to_numpy().astype(np.float32)
    )
    validation_labels = (
        validation_df["label"]
        .to_numpy().astype(np.float32)
    )
    test_labels = (
        test_df["label"]
        .to_numpy().astype(np.float32)
    )
    train_targets = torch.tensor(
        train_labels,dtype=torch.float32,
    )
    positive_weight = calculate_positive_weight(
        train_labels,device,
    )
    network = FraudDetectionNetwork(
        input_size=train_features.shape[1],
    ).to(device)
    best_state, best_validation_f1 = train_network(
        network=network,
        train_features=train_features,
        train_targets=train_targets,
        validation_features=validation_features,
        validation_targets=validation_labels,
        device=device,
        positive_weight=positive_weight,
    )
    network.load_state_dict(best_state)
    network.to(device)
    validation_probabilities = generate_probabilities(
        network,validation_features,device,
    )

    selected_threshold = find_optimal_threshold(
        validation_labels,validation_probabilities,
    )
    test_probabilities = generate_probabilities(
        network,test_features,device,
    )
    test_metrics = calculate_fraud_metrics(
        test_labels,test_probabilities,selected_threshold,
    )
    final_result = {
        "fraud_f1": test_metrics["fraud_f1"],
        "fraud_recall": test_metrics["fraud_recall"],
        "fraud_precision": test_metrics["fraud_precision"],
    }
    save_outputs(
        vectorizer=vectorizer,
        model_state=best_state,
        input_size=train_features.shape[1],
        threshold=selected_threshold,
        metrics=final_result,
    )
    print(
        json.dumps(
            final_result,indent=2,
        )
    )
    print(
        f"best_val_fraud_f1="
        f"{best_validation_f1:.4f} "
        f"threshold="
        f"{selected_threshold:.4f} "
        f"device={device}"
    )
    print(
        f"Saved weights under: {MODEL_PATH}"
    )
    print(
        f"Saved result: " f"{OUTPUT_PATH / 'test_metrics.json'}"
    )
if __name__ == "__main__":
    main()