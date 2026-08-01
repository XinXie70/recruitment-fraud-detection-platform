"""Generate BetterBERT probabilities for its fixed validation split.

The script does not train the model or change the saved threshold. It only
loads the teammate's saved checkpoint and runs inference on the existing
validation rows, preserving record IDs for later ensemble alignment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"record_id", "label", "model_text"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_rows(args.validation)
    if args.limit is not None:
        rows = rows[: args.limit]

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, local_files_only=True
    ).to(device)
    model.eval()

    output_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [row["model_text"] for row in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)[:, 1]
            for row, probability in zip(batch, probabilities.cpu().tolist()):
                output_rows.append(
                    {
                        "record_id": row["record_id"],
                        "label": int(row["label"]),
                        "fraud_score": float(probability),
                    }
                )
            if start == 0 or (start + len(batch)) % 200 < args.batch_size:
                print(f"Processed {start + len(batch)}/{len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "label", "fraud_score"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
