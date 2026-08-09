"""Re-select validation threshold with F-beta and evaluate on test.

Does not retrain. Uses the existing retrain checkpoint.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

import config as cfg
from dataset import dataframe_to_text_dataset, load_split
from metrics import binary_metrics, build_predictions_frame, select_threshold
from model import BertForFraudClassification
from train_bert import collate_batch, evaluate_loader
from utils import get_device, save_json, set_seed, setup_logging


def main() -> None:
    logger = setup_logging("reselect_threshold_fbeta.log")
    set_seed(42)
    device = get_device(allow_cpu=False, logger=logger)

    ckpt = cfg.PROJECT_ROOT / "weights" / "bert_paper_protocol" / "best"
    max_length = 256
    mode = "max_fbeta"
    f_beta = 0.5

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = BertForFraudClassification.__new__(BertForFraudClassification)
    torch.nn.Module.__init__(model)
    model.model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model = model.to(device).eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def make_loader(split_name: str) -> DataLoader:
        df = load_split(split_name)
        ds = dataframe_to_text_dataset(df, tokenizer, max_length)
        return DataLoader(
            ds,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda feats: collate_batch(feats, collator),
            num_workers=0,
        ), df

    val_loader, _ = make_loader("validation")
    test_loader, test_df = make_loader("test")

    _, y_val, p_val, *_ = evaluate_loader(
        model, val_loader, device, criterion=None, use_amp=True
    )
    best_thr, sweep = select_threshold(
        y_val, p_val, mode=mode, step=0.01, f_beta=f_beta
    )
    sweep.to_csv(cfg.RESULTS_DIR / "threshold_sweep_fbeta.csv", index=False)
    val_metrics = binary_metrics(y_val, p_val, threshold=best_thr)

    _, y_test, p_test, ids, titles, companies, idxs = evaluate_loader(
        model, test_loader, device, criterion=None, use_amp=True
    )
    test_metrics = binary_metrics(y_test, p_test, threshold=best_thr)
    test_old = binary_metrics(y_test, p_test, threshold=0.03)

    payload = {
        "method": {
            "mode": mode,
            "f_beta": f_beta,
            "note": (
                "Select threshold on validation by maximizing fraud F-beta "
                "(beta<1 favors precision / higher thresholds). Not hard-coded."
            ),
        },
        "checkpoint": str(ckpt),
        "selected_threshold": best_thr,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_old_thr_0_03": test_old,
    }
    out = cfg.RESULTS_DIR / "test_metrics_fbeta_threshold.json"
    save_json(payload, out)

    # Persist for predict scripts
    save_json(
        {
            "threshold": best_thr,
            "default_threshold": 0.5,
            "selection_mode": mode,
            "f_beta": f_beta,
            "model_name": "bert-base-uncased",
            "run_name": "bert_paper_protocol",
        },
        ckpt / "threshold_fbeta.json",
    )

    preds = build_predictions_frame(
        record_ids=ids,
        titles=titles,
        companies=companies,
        y_true=y_test,
        y_prob=p_test,
        threshold=best_thr,
        model_name="BERT:retrain_fbeta",
        imbalance_strategy="checkpoint",
        row_indices=idxs,
    )
    preds.to_csv(cfg.RESULTS_DIR / "predictions_fbeta_threshold.csv", index=False)

    logger.info(
        "Selected thr=%.4f on val (fraud_f1=%.4f precision=%.4f recall=%.4f)",
        best_thr,
        val_metrics["fraud_f1"],
        val_metrics["fraud_precision"],
        val_metrics["fraud_recall"],
    )
    logger.info(
        "Test thr=%.4f fraud_f1=%.4f precision=%.4f recall=%.4f macro_f1=%.4f",
        best_thr,
        test_metrics["fraud_f1"],
        test_metrics["fraud_precision"],
        test_metrics["fraud_recall"],
        test_metrics["macro_f1"],
    )
    print(
        json.dumps(
            {
                "selected_threshold": best_thr,
                "val_fraud_f1": val_metrics["fraud_f1"],
                "test_fraud_f1": test_metrics["fraud_f1"],
                "test_fraud_precision": test_metrics["fraud_precision"],
                "test_fraud_recall": test_metrics["fraud_recall"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
