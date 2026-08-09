"""Experiment A: end-to-end BERT fine-tuning on fixed train/val/test splits.

Usage (PowerShell):
  . .venv\\Scripts\\Activate.ps1
  cd model_code\\bert
  python train_bert.py --use_class_weights true
  python train_bert.py --use_class_weights false --run_name bert_no_class_weight
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from tqdm import tqdm

from config import (
    BERT_FINETUNED_DIR,
    FIGURES_DIR,
    LABEL_COLUMN,
    PRETRAINED_MODEL_NAME,
    RESULTS_DIR,
    BertFinetuneConfig,
    config_to_dict,
    default_paths_dict,
    ensure_directories,
)
from dataset import (
    build_data_quality_report,
    compute_class_weights,
    compute_token_length_stats,
    dataframe_to_text_dataset,
    load_all_splits,
)
from metrics import (
    binary_metrics,
    build_error_analysis,
    build_predictions_frame,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_pr,
    plot_training_curves,
    save_comparison_row,
    select_threshold,
)
from model import BertForFraudClassification, softmax_fraud_proba
from utils import (
    Timer,
    collect_environment_info,
    gpu_memory_mb,
    require_cuda_for_training,
    save_json,
    set_seed,
    setup_logging,
)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def collate_batch(features: List[Dict[str, Any]], collator: DataCollatorWithPadding) -> Dict[str, Any]:
    meta_keys = ("record_id", "title", "company", "row_index")
    meta = {k: [f[k] for f in features] for k in meta_keys}
    model_feats = []
    for f in features:
        item = {
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"],
            "labels": f["labels"],
        }
        if "token_type_ids" in f:
            item["token_type_ids"] = f["token_type_ids"]
        model_feats.append(item)
    batch = collator(model_feats)
    batch.update(meta)
    return batch


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[str], List[str], List[str], List[int]]:
    model.eval()
    losses: List[float] = []
    probs: List[float] = []
    labels: List[int] = []
    record_ids: List[str] = []
    titles: List[str] = []
    companies: List[str] = []
    row_indices: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=use_amp and device.type == "cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            if criterion is not None:
                loss = criterion(logits, y)
                losses.append(float(loss.item()))

        batch_prob = softmax_fraud_proba(logits).detach().float().cpu().numpy()
        probs.extend(batch_prob.tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
        record_ids.extend(batch["record_id"])
        titles.extend(batch["title"])
        companies.extend(batch["company"])
        row_indices.extend([int(x) for x in batch["row_index"]])

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, y_true, y_prob, record_ids, titles, companies, row_indices


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    criterion: nn.Module,
    scaler: GradScaler,
    grad_accum: int,
    max_grad_norm: float,
    use_amp: bool,
) -> float:
    model.train()
    running = 0.0
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        try:
            with autocast(enabled=use_amp and device.type == "cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = criterion(outputs.logits, y) / grad_accum
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during BERT fine-tuning. Try in order:\n"
                "  1) decrease --train_batch_size\n"
                "  2) increase --gradient_accumulation_steps\n"
                "  3) decrease --max_length\n"
                "  4) enable --gradient_checkpointing\n"
                "Do NOT silently continue on CPU for long training."
            ) from exc

        running += float(loss.item()) * grad_accum
        n_steps += 1

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    return running / max(n_steps, 1)


def save_checkpoint(
    output_dir: Path,
    model: BertForFraudClassification,
    tokenizer,
    cfg: BertFinetuneConfig,
    extra: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json({"config": config_to_dict(cfg), **extra}, output_dir / "train_meta.json")


def run_training(cfg: BertFinetuneConfig) -> Dict[str, Any]:
    ensure_directories()
    logger = setup_logging("training.log")
    set_seed(cfg.random_seed)
    device = require_cuda_for_training(logger)

    logger.info("Loading fixed splits (no re-split, no CV, no merge).")
    splits = load_all_splits()
    plot_class_distribution(splits, FIGURES_DIR / "class_distribution.png")

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
    token_stats = compute_token_length_stats(
        splits["train"]["model_text"].tolist(),
        tokenizer,
        max_samples=min(3000, len(splits["train"])),
    )
    logger.info("Token length stats (train sample): %s", token_stats)
    quality = build_data_quality_report(splits, token_stats=token_stats)
    save_json(quality, RESULTS_DIR / "data_quality_report.json")
    logger.info("Saved data_quality_report.json")

    class_weights = compute_class_weights(splits["train"][LABEL_COLUMN].to_numpy())
    logger.info("Train-only class weights: %s", class_weights.tolist())

    train_ds = dataframe_to_text_dataset(splits["train"], tokenizer, cfg.max_length)
    val_ds = dataframe_to_text_dataset(splits["validation"], tokenizer, cfg.max_length)
    test_ds = dataframe_to_text_dataset(splits["test"], tokenizer, cfg.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def make_loader(ds, batch_size, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda feats: collate_batch(feats, collator),
            num_workers=0,
        )

    train_loader = make_loader(train_ds, cfg.train_batch_size, True)
    val_loader = make_loader(val_ds, cfg.eval_batch_size, False)
    test_loader = make_loader(test_ds, cfg.eval_batch_size, False)

    model = BertForFraudClassification(
        cfg.pretrained_model_name,
        num_labels=2,
        dropout=cfg.dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)

    if cfg.use_class_weights:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        imbalance_name = "Class weight"
    else:
        criterion = nn.CrossEntropyLoss()
        imbalance_name = "None"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_update_steps = (
        (len(train_loader) + cfg.gradient_accumulation_steps - 1)
        // cfg.gradient_accumulation_steps
    ) * cfg.num_train_epochs
    warmup_steps = int(total_update_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    scaler = GradScaler(enabled=cfg.fp16 and device.type == "cuda")

    history_rows: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = -1
    patience_left = cfg.early_stopping_patience
    output_dir = Path(cfg.output_dir) / cfg.run_name
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_timer = Timer()
    logger.info(
        "Start fine-tuning: model=%s use_class_weights=%s max_length=%s batch=%s accum=%s",
        cfg.pretrained_model_name,
        cfg.use_class_weights,
        cfg.max_length,
        cfg.train_batch_size,
        cfg.gradient_accumulation_steps,
    )

    for epoch in range(1, cfg.num_train_epochs + 1):
        epoch_timer = Timer()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            criterion,
            scaler,
            cfg.gradient_accumulation_steps,
            cfg.max_grad_norm,
            cfg.fp16,
        )
        val_metrics, _, _, _, _, _, _ = evaluate_loader(
            model, val_loader, device, criterion=criterion, use_amp=cfg.fp16
        )
        lr_now = float(scheduler.get_last_lr()[0])
        mem = gpu_memory_mb()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "legitimate_precision": val_metrics["legitimate_precision"],
            "legitimate_recall": val_metrics["legitimate_recall"],
            "legitimate_f1": val_metrics["legitimate_f1"],
            "fraud_precision": val_metrics["fraud_precision"],
            "fraud_recall": val_metrics["fraud_recall"],
            "fraud_f1": val_metrics["fraud_f1"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "roc_auc": val_metrics["roc_auc"],
            "pr_auc": val_metrics["pr_auc"],
            "learning_rate": lr_now,
            "gpu_memory_mb": mem,
            "epoch_duration_sec": epoch_timer.elapsed(),
            "best_val_fraud_f1": max(best_metric, val_metrics["fraud_f1"]),
        }
        history_rows.append(row)
        logger.info(
            "Epoch %s | train_loss=%.4f val_loss=%.4f fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["fraud_f1"],
            val_metrics["fraud_recall"],
            val_metrics["pr_auc"],
        )

        selection_value = float(val_metrics.get(cfg.selection_metric, val_metrics["fraud_f1"]))
        if selection_value > best_metric:
            best_metric = selection_value
            best_epoch = epoch
            patience_left = cfg.early_stopping_patience
            save_checkpoint(
                best_dir,
                model,
                tokenizer,
                cfg,
                {
                    "best_epoch": best_epoch,
                    "best_validation_metric": best_metric,
                    "selection_metric": cfg.selection_metric,
                    "imbalance_method": imbalance_name,
                    "class_weights": class_weights.tolist(),
                },
            )
            logger.info("New best checkpoint saved at epoch %s (%s=%.4f)", epoch, cfg.selection_metric, best_metric)
        else:
            patience_left -= 1
            logger.info("No improvement. Patience left: %s", patience_left)
            if patience_left <= 0:
                logger.info("Early stopping at epoch %s", epoch)
                break

    history_df = pd.DataFrame(history_rows)
    history_path = RESULTS_DIR / f"training_history_{cfg.run_name}.csv"
    history_df.to_csv(history_path, index=False)
    write_aliases = cfg.write_canonical_aliases and cfg.use_class_weights
    # Also keep a canonical name for the default/main BERT run.
    if write_aliases:
        history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    plot_training_curves(history_df, FIGURES_DIR / f"training_loss_{cfg.run_name}.png")
    if write_aliases:
        plot_training_curves(history_df, FIGURES_DIR / "training_loss.png")

    # Reload best checkpoint before thresholding / test.
    logger.info("Loading best checkpoint from %s", best_dir)
    model.model = type(model.model).from_pretrained(best_dir)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(best_dir)

    # Validation threshold selection (never use test).
    val_metrics_05, y_val, p_val, val_ids, val_titles, val_companies, val_idx = evaluate_loader(
        model, val_loader, device, criterion=criterion, use_amp=cfg.fp16
    )
    if cfg.optimize_threshold:
        mode = "max_fraud_f1"
        if cfg.min_fraud_recall_for_threshold is not None:
            mode = "min_recall_then_precision"
        best_thr, sweep = select_threshold(
            y_val,
            p_val,
            mode=mode,
            min_fraud_recall=cfg.min_fraud_recall_for_threshold,
        )
    else:
        best_thr = cfg.default_threshold
        sweep = None
    if sweep is not None:
        sweep.to_csv(RESULTS_DIR / f"threshold_sweep_{cfg.run_name}.csv", index=False)

    val_metrics_opt = binary_metrics(y_val, p_val, threshold=best_thr)
    logger.info(
        "Validation threshold selected on validation only: %.4f (fraud_f1=%.4f)",
        best_thr,
        val_metrics_opt["fraud_f1"],
    )

    # Final test evaluation once.
    infer_timer = Timer()
    test_metrics_05, y_test, p_test, test_ids, test_titles, test_companies, test_idx = evaluate_loader(
        model, test_loader, device, criterion=criterion, use_amp=cfg.fp16
    )
    inference_time = infer_timer.elapsed()
    test_metrics = binary_metrics(y_test, p_test, threshold=best_thr)
    test_metrics["loss"] = test_metrics_05["loss"]
    test_metrics["threshold_default_0_5"] = binary_metrics(y_test, p_test, threshold=0.5)
    test_metrics["threshold_optimized"] = {
        "threshold": best_thr,
        **{k: v for k, v in test_metrics.items() if k not in {"threshold_default_0_5", "threshold_optimized"}},
    }

    preds = build_predictions_frame(
        record_ids=test_ids,
        titles=test_titles,
        companies=test_companies,
        y_true=y_test,
        y_prob=p_test,
        threshold=best_thr,
        model_name=f"{cfg.model_label}:{cfg.run_name}",
        imbalance_strategy=imbalance_name,
        row_indices=test_idx,
    )
    pred_path = RESULTS_DIR / f"predictions_{cfg.run_name}.csv"
    preds.to_csv(pred_path, index=False)
    if write_aliases:
        preds.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    errors = build_error_analysis(preds, splits["test"]["model_text"].tolist())
    err_path = RESULTS_DIR / f"error_analysis_{cfg.run_name}.csv"
    errors.to_csv(err_path, index=False)
    if write_aliases:
        errors.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        FIGURES_DIR / f"confusion_matrix_{cfg.run_name}.png",
        title=f"{cfg.model_label} test CM ({cfg.run_name})",
    )
    if write_aliases:
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            FIGURES_DIR / "confusion_matrix.png",
            title="BERT test Confusion Matrix",
        )
    plot_roc_pr(
        y_test,
        p_test,
        FIGURES_DIR / f"roc_curve_{cfg.run_name}.png",
        FIGURES_DIR / f"precision_recall_curve_{cfg.run_name}.png",
    )
    if write_aliases:
        plot_roc_pr(
            y_test,
            p_test,
            FIGURES_DIR / "roc_curve.png",
            FIGURES_DIR / "precision_recall_curve.png",
        )

    metrics_stem = "bert" if cfg.model_label.upper() == "BERT" else cfg.model_label.lower()
    metrics_path = RESULTS_DIR / f"{metrics_stem}_test_metrics_{cfg.run_name}.json"
    payload = {
        "model": f"{cfg.model_label} fine-tuned",
        "run_name": cfg.run_name,
        "imbalance_method": imbalance_name,
        "pretrained_model_name": cfg.pretrained_model_name,
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "selection_metric": cfg.selection_metric,
        "threshold": best_thr,
        "validation_metrics_threshold_0_5": val_metrics_05,
        "validation_metrics_optimized_threshold": val_metrics_opt,
        "test_metrics": test_metrics,
        "inference_time_sec": inference_time,
        "training_time_sec": total_timer.elapsed(),
        "class_weights_train_only": class_weights.tolist(),
    }
    save_json(payload, metrics_path)
    if write_aliases:
        save_json(payload, RESULTS_DIR / "bert_test_metrics.json")

    save_comparison_row(
        RESULTS_DIR / "model_comparison.csv",
        {
            "Model": cfg.model_label,
            "Imbalance method": imbalance_name,
            "Accuracy": test_metrics["accuracy"],
            "Fraud Precision": test_metrics["fraud_precision"],
            "Fraud Recall": test_metrics["fraud_recall"],
            "Fraud F1": test_metrics["fraud_f1"],
            "Macro F1": test_metrics["macro_f1"],
            "ROC-AUC": test_metrics["roc_auc"],
            "PR-AUC": test_metrics["pr_auc"],
            "Inference time (s)": inference_time,
            "Notes": f"run={cfg.run_name}; thr={best_thr:.4f}; best_epoch={best_epoch}",
        },
    )
    plot_model_comparison(RESULTS_DIR / "model_comparison.csv", FIGURES_DIR / "model_comparison.png")

    run_config = {
        "experiment": f"{cfg.model_label.lower()}_finetune",
        "paths": default_paths_dict(),
        "config": config_to_dict(cfg),
        "environment": collect_environment_info(),
        "best_checkpoint": str(best_dir),
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "threshold": best_thr,
        "token_length_stats": token_stats,
        "training_time_sec": total_timer.elapsed(),
        "imbalance_method": imbalance_name,
        "class_weights": class_weights.tolist(),
        "sample_counts": {
            k: {"n": len(v), "fraud": int((v[LABEL_COLUMN] == 1).sum())} for k, v in splits.items()
        },
    }
    save_json(run_config, RESULTS_DIR / f"run_config_{cfg.run_name}.json")
    if write_aliases:
        save_json(run_config, RESULTS_DIR / "run_config.json")

    # Persist threshold next to checkpoint for predict_bert.py
    save_json(
        {
            "threshold": best_thr,
            "default_threshold": 0.5,
            "model_name": cfg.pretrained_model_name,
            "run_name": cfg.run_name,
            "imbalance_method": imbalance_name,
        },
        best_dir / "threshold.json",
    )

    logger.info("Test fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f", test_metrics["fraud_f1"], test_metrics["fraud_recall"], test_metrics["pr_auc"])
    logger.info("Done. Artifacts under %s and %s", output_dir, RESULTS_DIR)
    return payload


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune BERT on fixed job-ad splits")
    p.add_argument("--pretrained_model_name", default=PRETRAINED_MODEL_NAME)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--fp16", type=parse_bool, default=True)
    p.add_argument("--use_class_weights", type=parse_bool, default=True)
    p.add_argument("--gradient_checkpointing", type=parse_bool, default=False)
    p.add_argument("--optimize_threshold", type=parse_bool, default=True)
    p.add_argument("--min_fraud_recall_for_threshold", type=float, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=str(BERT_FINETUNED_DIR))
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_name = args.run_name
    if run_name is None:
        run_name = "bert_class_weighted" if args.use_class_weights else "bert_no_class_weight"
    cfg = BertFinetuneConfig(
        pretrained_model_name=args.pretrained_model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        dropout=args.dropout,
        random_seed=args.random_seed,
        fp16=args.fp16,
        use_class_weights=args.use_class_weights,
        gradient_checkpointing=args.gradient_checkpointing,
        optimize_threshold=args.optimize_threshold,
        min_fraud_recall_for_threshold=args.min_fraud_recall_for_threshold,
        output_dir=args.output_dir,
        run_name=run_name,
    )
    try:
        run_training(cfg)
    except SystemExit:
        raise
    except Exception:
        logger = setup_logging("training.log")
        logger.error("Training failed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
