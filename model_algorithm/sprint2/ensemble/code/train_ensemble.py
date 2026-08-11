#Mean-score ensemble of sprint1 and sprint2 models for EMSCAD data.
from __future__ import annotations
import json, os, random, re, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset
os.environ.setdefault("PYTHONHASHSEED", "42")

PROJECT_PATH = Path(__file__).resolve().parents[1]
SPRINT1_PATH = PROJECT_PATH.parent.parent / "sprint1"
SPRINT2_PATH = PROJECT_PATH.parent
SPLIT_PATH = SPRINT1_PATH / "data" / "splits"
OUTPUT_PATH = PROJECT_PATH / "result"
MODEL_PATH = PROJECT_PATH / "weight"

RANDOM_SEED = 42
PADDING_INDEX = 0
UNKNOWN_INDEX = 1
SEQUENCE_BATCH_SIZE = 64
DNN_BATCH_SIZE = 256
LR_WEIGHT = SPRINT1_PATH / "LR" / "weight" / "lr_tfidf.joblib"
DNN_WEIGHT = SPRINT1_PATH / "DNN" / "weight" / "dnn_mlp.pt"
DNN_VECTORIZER = SPRINT1_PATH / "DNN" / "weight" / "tfidf.joblib"
SVM_WEIGHT = SPRINT2_PATH / "SVM" / "weight" / "svm_tfidf.joblib"
XGB_WEIGHT = SPRINT2_PATH / "XGBOOST" / "weight" / "xgboost_tfidf.joblib"
RNN_WEIGHT = SPRINT2_PATH / "RNN" / "weight" / "rnn.pt"
BILSTM_WEIGHT = SPRINT2_PATH / "Bi-lstm" / "weight" / "bilstm.pt"

def initialize_random_state(seed: int = RANDOM_SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_data(split_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(SPLIT_PATH / f"{split_name}.csv", usecols=["label", "combined_text"])
    dataframe["combined_text"] = dataframe["combined_text"].fillna("").astype(str)
    return dataframe
def require_file(file_path: Path) -> Path:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing weight file: {file_path}")
    return file_path
def split_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())
def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision_values, recall_values, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f1_values = 2 * precision_values[:-1] * recall_values[:-1] / np.maximum(precision_values[:-1] + recall_values[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])
def fraud_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "fraud_f1": float(f1_score(labels, predictions, zero_division=0)),
        "fraud_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fraud_precision": float(precision_score(labels, predictions, zero_division=0)),
    }
def scale_to_unit(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - low) / (high - low)

class FraudDetectionNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, dropout_rate: float = 0.3) -> None:
        super().__init__()
        second_hidden_size = hidden_size // 2
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, second_hidden_size), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(second_hidden_size, 1),
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(dim=-1)

class JobTextDataset(Dataset):
    def __init__(self, texts: list[str], vocabulary: dict[str, int], max_len: int) -> None:
        self.encoded_texts = []
        for text in texts:
            token_ids = [vocabulary.get(token, UNKNOWN_INDEX) for token in split_text(text)[:max_len]]
            if not token_ids:
                token_ids = [UNKNOWN_INDEX]
            self.encoded_texts.append(torch.tensor(token_ids, dtype=torch.long))
    def __len__(self) -> int:
        return len(self.encoded_texts)
    def __getitem__(self, index: int):
        return self.encoded_texts[index]
def sequence_collator(batch):
    sequence_lengths = torch.tensor([sequence.size(0) for sequence in batch], dtype=torch.long)
    padded_sequences = pad_sequence(batch, batch_first=True, padding_value=PADDING_INDEX)
    return padded_sequences, sequence_lengths

class FraudRNN(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=PADDING_INDEX)
        self.recurrent_layer = nn.RNN(embedding_size, hidden_size, batch_first=True, nonlinearity="tanh")
        self.dropout_layer = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_size, 1)
    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_layer(token_ids)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_state = self.recurrent_layer(packed_embeddings)
        return self.output_layer(self.dropout_layer(hidden_state[-1])).squeeze(-1)

class FraudBiLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size, padding_idx=PADDING_INDEX)
        self.recurrent_layer = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_size * 2, 1)
    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_layer(token_ids)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden_state, _) = self.recurrent_layer(packed_embeddings)
        representation = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
        return self.output_layer(self.dropout_layer(representation)).squeeze(-1)
def score_lr(texts: list[str]) -> np.ndarray:
    artifact = joblib.load(require_file(LR_WEIGHT))
    return artifact["model"].predict_proba(texts)[:, 1].astype(np.float64)

@torch.no_grad()
def score_dnn(texts: list[str], device: torch.device) -> np.ndarray:
    checkpoint = torch.load(require_file(DNN_WEIGHT), map_location=device, weights_only=False)
    vectorizer = joblib.load(require_file(DNN_VECTORIZER))
    features = vectorizer.transform(texts)
    dense_matrix = features.toarray() if hasattr(features, "toarray") else features
    feature_tensor = torch.as_tensor(dense_matrix, dtype=torch.float32)
    network = FraudDetectionNetwork(
        input_size=int(checkpoint["n_features"]),
        hidden_size=int(checkpoint.get("hidden", 256)),
        dropout_rate=float(checkpoint.get("dropout", 0.3)),
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device)
    network.eval()
    probability_batches = []
    loader = DataLoader(TensorDataset(feature_tensor), batch_size=DNN_BATCH_SIZE, shuffle=False)
    for (feature_batch,) in loader:
        logits = network(feature_batch.to(device))
        probability_batches.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probability_batches, axis=0).astype(np.float64)
def score_svm(texts: list[str]) -> np.ndarray:
    artifact = joblib.load(require_file(SVM_WEIGHT))
    features = artifact["vectorizer"].transform(texts)
    return artifact["model"].decision_function(features).astype(np.float64)

def score_xgboost(texts: list[str]) -> np.ndarray:
    artifact = joblib.load(require_file(XGB_WEIGHT))
    features = artifact["vectorizer"].transform(texts)
    return artifact["model"].predict_proba(features)[:, 1].astype(np.float64)

@torch.no_grad()
def score_sequence(texts: list[str], checkpoint_path: Path, model_class, device: torch.device) -> np.ndarray:
    checkpoint = torch.load(require_file(checkpoint_path), map_location=device, weights_only=False)
    vocabulary = checkpoint["vocab"]
    max_len = int(checkpoint.get("max_len", 200))
    model = model_class(
        vocabulary_size=len(vocabulary),
        embedding_size=int(checkpoint.get("embed_dim", 128)),
        hidden_size=int(checkpoint.get("hidden", 128)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    loader = DataLoader(
        JobTextDataset(texts, vocabulary, max_len),
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=False,
        collate_fn=sequence_collator,
    )
    probability_batches = []
    for token_ids, lengths in loader:
        logits = model(token_ids.to(device), lengths)
        probability_batches.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probability_batches, axis=0).astype(np.float64)

def collect_member_scores(texts: list[str], device: torch.device) -> dict[str, np.ndarray]:
    return {
        "LR": score_lr(texts),
        "DNN": score_dnn(texts, device),
        "SVM": score_svm(texts),
        "XGBOOST": score_xgboost(texts),
        "RNN": score_sequence(texts, RNN_WEIGHT, FraudRNN, device),
        "Bi-lstm": score_sequence(texts, BILSTM_WEIGHT, FraudBiLSTM, device),
    }

def average_scores(member_scores: dict[str, np.ndarray], svm_low: float, svm_high: float) -> np.ndarray:
    aligned = []
    for name, scores in member_scores.items():
        if name == "SVM":
            aligned.append(scale_to_unit(scores, svm_low, svm_high))
        else:
            aligned.append(scores)
    return np.mean(np.stack(aligned, axis=0), axis=0)

def save_outputs(threshold: float, svm_low: float, svm_high: float, metrics: dict[str, float]) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "members": ["LR", "DNN", "SVM", "XGBOOST", "RNN", "Bi-lstm"],
            "aggregation": "mean",
            "threshold": float(threshold),
            "svm_score_min": float(svm_low),
            "svm_score_max": float(svm_high),
        },
        MODEL_PATH / "ensemble.joblib",
    )

def main() -> None:
    initialize_random_state()
    device = pick_device()
    validation_df = read_data("validation")
    test_df = read_data("test")
    validation_texts = validation_df["combined_text"].tolist()
    test_texts = test_df["combined_text"].tolist()
    validation_labels = validation_df["label"].to_numpy()
    test_labels = test_df["label"].to_numpy()
    validation_scores = collect_member_scores(validation_texts, device)
    test_scores = collect_member_scores(test_texts, device)
    svm_low = float(np.min(validation_scores["SVM"]))
    svm_high = float(np.max(validation_scores["SVM"]))
    validation_mean = average_scores(validation_scores, svm_low, svm_high)
    test_mean = average_scores(test_scores, svm_low, svm_high)
    selected_threshold = choose_threshold(validation_labels, validation_mean)
    metrics = fraud_metrics(test_labels, test_mean, selected_threshold)
    save_outputs(selected_threshold, svm_low, svm_high, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"threshold={selected_threshold:.4f} device={device}")
    print(f"Saved weights under: {MODEL_PATH}")
    print(f"Saved result: {OUTPUT_PATH / 'test_metrics.json'}")

if __name__ == "__main__":
    main()
