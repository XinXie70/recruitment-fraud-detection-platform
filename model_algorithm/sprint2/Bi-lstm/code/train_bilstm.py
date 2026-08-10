#Bi-LSTM for EMSCAD data.
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
LSTM_HIDDEN_SIZE = 128
TRAIN_BATCH_SIZE = 64
MAX_EPOCHS = 12
LEARNING_RATE = 1e-3
EARLY_STOPPING_LIMIT = 4
PADDING_INDEX = 0
UNKNOWN_INDEX = 1

def initialize_random_state(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())

def read_data(split_name: str) -> pd.DataFrame:
    file_path = SPLIT_PATH / f"{split_name}.csv"
    dataframe = pd.read_csv(file_path, usecols=["label", "combined_text"])
    dataframe["combined_text"] = dataframe["combined_text"].fillna("").astype(str)
    return dataframe

def create_vocabulary(text_collection: list[str]) -> dict[str, int]:
    frequencies = Counter()
    for sentence in text_collection:
        frequencies.update(split_text(sentence))
    vocabulary = {"<PAD>": PADDING_INDEX, "<UNK>": UNKNOWN_INDEX}
    candidate_words = frequencies.most_common(VOCABULARY_LIMIT)

    for word, frequency in candidate_words:
        if frequency < WORD_MIN_COUNT:
            continue
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    return vocabulary

def text_to_indices(text: str, vocabulary: dict[str, int]) -> torch.Tensor:
    tokens = split_text(text)
    token_ids = [vocabulary.get(token, UNKNOWN_INDEX) for token in tokens[:SEQUENCE_LIMIT]]
    if len(token_ids) == 0:
        token_ids = [UNKNOWN_INDEX]
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
    sequences, targets = zip(*batch)
    sequence_lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=PADDING_INDEX)
    target_tensor = torch.stack(targets)
    return padded_sequences, sequence_lengths, target_tensor

class FraudBiLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int = EMBEDDING_SIZE, hidden_size: int = LSTM_HIDDEN_SIZE) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size, padding_idx=PADDING_INDEX)
        self.recurrent_layer = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_size * 2, 1)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_layer(token_ids)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_state = self.recurrent_layer(packed_embeddings)
        hidden_vectors = hidden_state[0]
        forward_vector = hidden_vectors[-2]
        backward_vector = hidden_vectors[-1]
        representation = torch.cat((forward_vector, backward_vector), dim=1)
        representation = self.dropout_layer(representation)
        logits = self.output_layer(representation)
        return logits.squeeze(dim=-1)

def create_data_loader(dataframe: pd.DataFrame, vocabulary: dict[str, int], shuffle_data: bool) -> DataLoader:
    dataset = JobTextDataset(texts=dataframe["combined_text"].tolist(), labels=dataframe["label"].to_numpy(), vocabulary=vocabulary)
    return DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=shuffle_data, collate_fn=batch_collator)

@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability_list = []
    target_list = []
    for tokens, lengths, targets in loader:
        tokens = tokens.to(device)
        logits = model(tokens, lengths)
        probabilities = torch.sigmoid(logits)
        probability_list.append(probabilities.cpu().numpy())
        target_list.append(targets.numpy())
    return np.concatenate(probability_list), np.concatenate(target_list)

def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision_values, recall_values, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    numerator = 2 * precision_values[:-1] * recall_values[:-1]
    denominator = np.maximum(precision_values[:-1] + recall_values[:-1], 1e-12)
    f1_values = numerator / denominator
    maximum_f1 = np.max(f1_values)
    best_index = np.flatnonzero(np.isclose(f1_values, maximum_f1))[0]
    return float(thresholds[best_index])

def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = np.where(probabilities >= threshold, 1, 0)

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
    positive_weight = create_class_weight(training_labels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best_validation_f1 = -1.0
    best_parameters = None
    patience_counter = 0
    for epoch_number in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for tokens, lengths, targets in train_loader:
            tokens = tokens.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        validation_scores, validation_labels = collect_predictions(model, validation_loader, device)
        current_threshold = choose_threshold(validation_labels, validation_scores)
        validation_metrics = calculate_metrics(validation_labels, validation_scores, current_threshold)
        current_f1 = validation_metrics["fraud_f1"]
        average_loss = total_loss / sample_count
        print(f"epoch={epoch_number:02d} loss={average_loss:.4f} val_f1={current_f1:.4f}")
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
    result_file = OUTPUT_PATH / "test_metrics.json"
    result_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    checkpoint = {"model_state_dict": model_state, "vocab": vocabulary, "threshold": float(threshold), "max_len": SEQUENCE_LIMIT, "embed_dim": EMBEDDING_SIZE, "hidden": LSTM_HIDDEN_SIZE}
    torch.save(checkpoint, MODEL_PATH / "bilstm.pt")

def main() -> None:
    initialize_random_state()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    training_data = read_data("train")
    validation_data = read_data("validation")
    test_data = read_data("test")
    vocabulary = create_vocabulary(training_data["combined_text"].tolist())
    training_loader = create_data_loader(training_data, vocabulary, True)
    validation_loader = create_data_loader(validation_data, vocabulary, False)
    test_loader = create_data_loader(test_data, vocabulary, False)
    training_labels = training_data["label"].to_numpy()
    model = FraudBiLSTM(vocabulary_size=len(vocabulary)).to(device)
    best_state, best_validation_f1 = train_model(model=model, train_loader=training_loader, validation_loader=validation_loader, training_labels=training_labels, device=device)
    model.load_state_dict(best_state)
    model.to(device)
    validation_scores, validation_labels = collect_predictions(model, validation_loader, device)
    selected_threshold = choose_threshold(validation_labels, validation_scores)
    test_scores, test_labels = collect_predictions(model, test_loader, device)
    test_result = calculate_metrics(test_labels, test_scores, selected_threshold)
    save_outputs(model_state=best_state, vocabulary=vocabulary, threshold=selected_threshold, metrics=test_result)
    print(json.dumps(test_result, indent=2))
    print(f"best_val_f1={best_validation_f1:.4f} threshold={selected_threshold:.4f}")

if __name__ == "__main__":
    main()