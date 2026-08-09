"""Single-ad / batch inference for the fine-tuned BERT classifier.

Examples (PowerShell):
  . .venv\\Scripts\\Activate.ps1
  cd model_code\\bert
  python predict_bert.py --text "Urgent work-from-home job. Send bank details to apply."
  python predict_bert.py --csv path\\to\\ads.csv --text_column combined_text
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer

from config import BERT_FINETUNED_DIR, PRETRAINED_MODEL_NAME, ensure_directories
from model import BertForFraudClassification, softmax_fraud_proba
from preprocessing import clean_text, extract_text_from_row
from utils import get_device, load_json, setup_logging


def load_predictor(
    checkpoint_dir: Path,
    device: torch.device,
    threshold_override: Optional[float] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    wrapper = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    wrapper.model = type(wrapper.model).from_pretrained(checkpoint_dir)
    wrapper = wrapper.to(device)
    wrapper.eval()

    thr_path = checkpoint_dir / "threshold.json"
    meta_path = checkpoint_dir / "train_meta.json"
    threshold = 0.5
    model_name = PRETRAINED_MODEL_NAME
    if thr_path.exists():
        thr_obj = load_json(thr_path)
        threshold = float(thr_obj.get("threshold", 0.5))
        model_name = thr_obj.get("model_name", model_name)
    if threshold_override is not None:
        threshold = float(threshold_override)
    meta = load_json(meta_path) if meta_path.exists() else {}
    return wrapper, tokenizer, threshold, model_name, meta


@torch.no_grad()
def predict_texts(
    texts: List[str],
    model: BertForFraudClassification,
    tokenizer,
    device: torch.device,
    threshold: float,
    max_length: int = 256,
) -> List[Dict[str, Any]]:
    model.eval()
    results = []
    for text in texts:
        cleaned = clean_text(text)
        enc = tokenizer(
            cleaned,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        fraud_p = float(softmax_fraud_proba(logits)[0].cpu().item())
        legit_p = 1.0 - fraud_p
        pred = 1 if fraud_p >= threshold else 0
        label = "Fraudulent" if pred == 1 else "Legitimate"
        results.append(
            {
                "predicted_label": label,
                "predicted_label_id": pred,
                "fraud_probability": fraud_p,
                "legitimate_probability": legit_p,
                "fraud_risk_score": fraud_p,
                "threshold": threshold,
            }
        )
    return results


def format_prediction(result: Dict[str, Any], model_name: str) -> str:
    return (
        f"Predicted label: {result['predicted_label']}\n"
        f"Fraud probability: {result['fraud_probability']:.4f}\n"
        f"Legitimate probability: {result['legitimate_probability']:.4f}\n"
        f"Fraud risk score: {result['fraud_risk_score']:.4f}\n"
        f"Threshold: {result['threshold']:.4f}\n"
        f"Model: {model_name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fraudulent job ads with BERT")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(BERT_FINETUNED_DIR / "bert_class_weighted" / "best"),
    )
    parser.add_argument("--text", type=str, default=None, help="Single job-ad text")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV of ads")
    parser.add_argument("--text_column", type=str, default="combined_text")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--allow_cpu", action="store_true", default=True)
    args = parser.parse_args()

    ensure_directories()
    logger = setup_logging("training.log")
    device = get_device(allow_cpu=args.allow_cpu, logger=logger)
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. Train first with train_bert.py."
        )

    model, tokenizer, threshold, model_name, _meta = load_predictor(
        ckpt, device, threshold_override=args.threshold
    )

    if args.text:
        result = predict_texts(
            [args.text], model, tokenizer, device, threshold, args.max_length
        )[0]
        print(format_prediction(result, model_name))
        return

    if args.csv:
        df = pd.read_csv(args.csv)
        texts = []
        for _, row in df.iterrows():
            if args.text_column in df.columns:
                texts.append(clean_text(row[args.text_column]))
            else:
                texts.append(extract_text_from_row(row.to_dict()))
        results = predict_texts(
            texts, model, tokenizer, device, threshold, args.max_length
        )
        out = df.copy()
        out["predicted_label"] = [r["predicted_label"] for r in results]
        out["fraud_probability"] = [r["fraud_probability"] for r in results]
        out["legitimate_probability"] = [r["legitimate_probability"] for r in results]
        out["fraud_risk_score"] = [r["fraud_risk_score"] for r in results]
        out["threshold"] = threshold
        out_path = Path(args.output_csv) if args.output_csv else Path(args.csv).with_name(
            Path(args.csv).stem + "_bert_predictions.csv"
        )
        out.to_csv(out_path, index=False)
        print(f"Wrote predictions to {out_path}")
        return

    parser.error("Provide --text or --csv")


if __name__ == "__main__":
    main()
