#RNN for EMSCAD data.
from __future__ import annotations
import json, os, random, re, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import Counter
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
os.environ.setdefault("PYTHONHASHSEED", "42")

PROJECT_PATH = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_PATH.parent.parent / "sprint1" / "data" / "splits"
OUTPUT_PATH = PROJECT_PATH / "result"
MODEL_PATH = PROJECT_PATH / "weight"

RANDOM_SEED = 42
SEQUENCE_LIMIT = 200
WORD_MIN_COUNT = 2
VOCABULARY_LIMIT = 20_000
EMBEDDING_SIZE = 128
RNN_HIDDEN_SIZE = 128
TRAIN_BATCH_SIZE = 64
MAX_EPOCHS = 12
LEARNING_RATE = 1e-3
EARLY_STOPPING_LIMIT = 4
PADDING_INDEX = 0
UNKNOWN_INDEX = 1

def initialize_random_state(seed: int = RANDOM_SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def split_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())
def read_data(split_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(SPLIT_PATH / f"{split_name}.csv", usecols=["label", "combined_text"])
    dataframe["combined_text"] = dataframe["combined_text"].fillna("").astype(str)
    return dataframe

def create_vocabulary(text_collection: list[str]) -> dict[str, int]:
    frequencies = Counter()
    for sentence in text_collection:
        frequencies.update(split_text(sentence))
    vocabulary = {"<PAD>": PADDING_INDEX, "<UNK>": UNKNOWN_INDEX}
    for word, frequency in frequencies.most_common(VOCABULARY_LIMIT):
        if frequency >= WORD_MIN_COUNT and word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    return vocabulary

def text_to_indices(text: str, vocabulary: dict[str, int]) -> torch.Tensor:
    token_ids = [vocabulary.get(token, UNKNOWN_INDEX) for token in split_text(text)[:SEQUENCE_LIMIT]]
    if not token_ids: token_ids = [UNKNOWN_INDEX]
    return torch.tensor(token_ids, dtype=torch.long)

class JobTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocabulary: dict[str, int]) -> None:
        self.encoded_texts = [text_to_indices(text, vocabulary) for text in texts]
        self.targets = torch.as_tensor(labels, dtype=torch.float32)
    def __len__(self) -> int:
        return len(self.targets)
    def __getitem__(self, index: int):
        return self.encoded_texts[index], self.targets[index]

def batch_collator(batch):
    sequences, targets = zip(*batch, strict=True)
    sequence_lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=PADDING_INDEX)
    return padded_sequences, sequence_lengths, torch.stack(targets)

class FraudRNN(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int = EMBEDDING_SIZE, hidden_size: int = RNN_HIDDEN_SIZE) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=PADDING_INDEX)
        self.recurrent_layer = nn.RNN(embedding_size, hidden_size, batch_first=True, nonlinearity="tanh")
        self.dropout_layer = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_size, 1)
    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_layer(token_ids)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_state = self.recurrent_layer(packed_embeddings)
        representation = self.dropout_layer(hidden_state[-1])
        return self.output_layer(representation).squeeze(-1)

def create_data_loader(dataframe: pd.DataFrame, vocabulary: dict[str, int], shuffle_data: bool) -> DataLoader:
    dataset = JobTextDataset(dataframe["combined_text"].tolist(), dataframe["label"].to_numpy(), vocabulary)
    return DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=shuffle_data, collate_fn=batch_collator)

@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for tokens, lengths, targets in loader:
        logits = model(tokens.to(device), lengths)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)

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

def create_class_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    positive_count = max(float(labels.sum()), 1.0)
    negative_count = float(len(labels)) - positive_count
    return torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)

def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, training_labels: np.ndarray, device: torch.device) -> tuple[dict, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=create_class_weight(training_labels, device))
    best_validation_f1, best_parameters, patience_counter = -1.0, None, 0
    for epoch_number in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, sample_count = 0.0, 0
        for tokens, lengths, targets in train_loader:
            tokens, targets = tokens.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(tokens, lengths), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            sample_count += len(targets)
        validation_scores, validation_labels = collect_predictions(model, validation_loader, device)
        current_threshold = choose_threshold(validation_labels, validation_scores)
        current_f1 = calculate_metrics(validation_labels, validation_scores, current_threshold)["fraud_f1"]
        print(f"epoch={epoch_number:02d} loss={total_loss / sample_count:.4f} val_f1={current_f1:.4f}")
        if current_f1 > best_validation_f1 + 1e-6:
            best_validation_f1 = current_f1
            best_parameters = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_LIMIT:
                print(f"Early stop at epoch {epoch_number}")
                break
    if best_parameters is None:
        raise RuntimeError("Training failed: no model state was saved")
    return best_parameters, best_validation_f1

def save_outputs(model_state: dict, vocabulary: dict[str, int], threshold: float, metrics: dict) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checkpoint = {"model_state_dict": model_state, "vocab": vocabulary, "threshold": float(threshold), "max_len": SEQUENCE_LIMIT, "embed_dim": EMBEDDING_SIZE, "hidden": RNN_HIDDEN_SIZE}
    # Offline training artifact only; not loaded from untrusted input.
    torch.save(checkpoint, MODEL_PATH / "rnn.pt")  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch

def main() -> None:
    initialize_random_state()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    training_data, validation_data, test_data = read_data("train"), read_data("validation"), read_data("test")
    vocabulary = create_vocabulary(training_data["combined_text"].tolist())
    training_loader = create_data_loader(training_data, vocabulary, True)
    validation_loader = create_data_loader(validation_data, vocabulary, False)
    test_loader = create_data_loader(test_data, vocabulary, False)
    training_labels = training_data["label"].to_numpy()
    model = FraudRNN(len(vocabulary)).to(device)
    best_state, best_validation_f1 = train_model(model, training_loader, validation_loader, training_labels, device)
    model.load_state_dict(best_state)
    model.to(device)
    validation_scores, validation_labels = collect_predictions(model, validation_loader, device)
    selected_threshold = choose_threshold(validation_labels, validation_scores)
    test_scores, test_labels = collect_predictions(model, test_loader, device)
    test_result = calculate_metrics(test_labels, test_scores, selected_threshold)
    save_outputs(best_state, vocabulary, selected_threshold, test_result)
    print(json.dumps(test_result, indent=2))
    print(f"best_val_f1={best_validation_f1:.4f} threshold={selected_threshold:.4f}")

if __name__ == "__main__":
    main()