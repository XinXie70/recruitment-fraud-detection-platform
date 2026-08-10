#Training for BERT
from __future__ import annotations
import argparse, shutil, traceback, torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from config import BERT_FINETUNED_DIR, ENSEMBLE_VAL_COPY, FIGURES_DIR, LABEL_COLUMN, MAX_LENGTH, MODEL_LABEL, PRETRAINED_MODEL_NAME, RANDOM_SEED, RESULTS_DIR, RUN_NAME, BertFinetuneConfig, Timer, collect_environment_info, config_to_dict, default_paths_dict, ensure_directories, get_device, gpu_memory_mb, load_json, require_cuda_for_training, save_json, set_seed, setup_logging
from data_metrics import binary_metrics, build_data_quality_report, build_error_analysis, build_predictions_frame, clean_text, compute_class_weights, compute_token_length_stats, dataframe_to_text_dataset, extract_text_from_row, load_all_splits, load_split, oversample_fraud_train, plot_class_distribution, plot_confusion_matrix, plot_model_comparison, plot_roc_pr, plot_training_curves, save_comparison_row, select_threshold

class BertForFraudClassification(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int=2, dropout: float=0.1, gradient_checkpointing: bool=False) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        if hasattr(config, 'hidden_dropout_prob'):
            config.hidden_dropout_prob = dropout
        if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None:
            config.classifier_dropout = dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=config)
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids
        return self.model(**kwargs)
    @property
    def encoder(self):
        return self.model.bert if hasattr(self.model, 'bert') else self.model.base_model

def softmax_fraud_proba(logits):
    return F.softmax(logits, dim=-1)[:, 1]
def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}
def collate_batch(features: List[Dict[str, Any]], collator: DataCollatorWithPadding) -> Dict[str, Any]:
    meta_keys = ('record_id', 'title', 'company', 'row_index')
    meta = {k: [f[k] for f in features] for k in meta_keys}
    model_feats = []
    for f in features:
        item = {'input_ids': f['input_ids'], 'attention_mask': f['attention_mask'], 'labels': f['labels']}
        if 'token_type_ids' in f:
            item['token_type_ids'] = f['token_type_ids']
        model_feats.append(item)
    batch = collator(model_feats)
    batch.update(meta)
    return batch

@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, criterion: Optional[nn.Module]=None, use_amp: bool=True) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[str], List[str], List[str], List[int]]:
    model.eval()
    losses: List[float] = []
    probs: List[float] = []
    labels: List[int] = []
    record_ids: List[str] = []
    titles: List[str] = []
    companies: List[str] = []
    row_indices: List[int] = []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        with autocast(enabled=use_amp and device.type == 'cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            if criterion is not None:
                loss = criterion(logits, y)
                losses.append(float(loss.item()))
        batch_prob = softmax_fraud_proba(logits).detach().float().cpu().numpy()
        probs.extend(batch_prob.tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
        record_ids.extend(batch['record_id'])
        titles.extend(batch['title'])
        companies.extend(batch['company'])
        row_indices.extend([int(x) for x in batch['row_index']])
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    metrics['loss'] = float(np.mean(losses)) if losses else float('nan')
    return (metrics, y_true, y_prob, record_ids, titles, companies, row_indices)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler, device: torch.device, criterion: nn.Module, scaler: GradScaler, grad_accum: int, max_grad_norm: float, use_amp: bool) -> float:
    model.train()
    running = 0.0
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc='train', leave=False)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = batch['labels'].to(device)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        try:
            with autocast(enabled=use_amp and device.type == 'cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = criterion(outputs.logits, y) / grad_accum
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError('CUDA OOM during BERT fine-tuning. Try in order:\n  1) decrease --train_batch_size\n  2) increase --gradient_accumulation_steps\n  3) decrease --max_length\n  4) enable --gradient_checkpointing\nDo NOT silently continue on CPU for long training.') from exc
        running += float(loss.item()) * grad_accum
        n_steps += 1
        if (step + 1) % grad_accum == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
    return running / max(n_steps, 1)

def save_checkpoint(output_dir: Path, model: BertForFraudClassification, tokenizer, cfg: BertFinetuneConfig, extra: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json({'config': config_to_dict(cfg), **extra}, output_dir / 'train_meta.json')

def run_training(cfg: BertFinetuneConfig, splits: Optional[Dict[str, pd.DataFrame]]=None, results_dir: Optional[Path]=None, figures_dir: Optional[Path]=None, log_name: str='training.log') -> Dict[str, Any]:
    ensure_directories()
    results_root = Path(results_dir) if results_dir is not None else RESULTS_DIR
    figures_root = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    results_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_name)
    set_seed(cfg.random_seed)
    device = require_cuda_for_training(logger)
    if splits is None:
        logger.info('Loading fixed splits (no re-split, no CV, no merge).')
        splits = load_all_splits()
    else:
        required = {'train', 'validation', 'test'}
        missing = required - set(splits)
        if missing:
            raise ValueError(f'splits missing keys: {sorted(missing)}')
        logger.info('Using caller-provided splits: train=%s validation=%s test=%s', len(splits['train']), len(splits['validation']), len(splits['test']))
    plot_class_distribution(splits, figures_root / 'class_distribution.png')
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
    token_stats = compute_token_length_stats(splits['train']['model_text'].tolist(), tokenizer, max_samples=min(3000, len(splits['train'])))
    logger.info('Token length stats (train sample): %s', token_stats)
    quality = build_data_quality_report(splits, token_stats=token_stats)
    save_json(quality, results_root / 'data_quality_report.json')
    logger.info('Saved data_quality_report.json')
    train_frame = splits['train']
    n_train_before = len(train_frame)
    n_fraud_before = int((train_frame[LABEL_COLUMN] == 1).sum())
    train_frame = oversample_fraud_train(train_frame, factor=cfg.fraud_oversample_factor, seed=cfg.random_seed)
    n_train_after = len(train_frame)
    n_fraud_after = int((train_frame[LABEL_COLUMN] == 1).sum())
    logger.info('Train-only fraud oversample: factor=%.2f  rows %d→%d  fraud %d→%d', float(cfg.fraud_oversample_factor), n_train_before, n_train_after, n_fraud_before, n_fraud_after)
    class_weights = compute_class_weights(train_frame[LABEL_COLUMN].to_numpy(), transform=cfg.class_weight_transform, weight_max=cfg.class_weight_max)
    logger.info('Train-only class weights (transform=%s, max=%s): %s', cfg.class_weight_transform, cfg.class_weight_max, class_weights.tolist())
    train_ds = dataframe_to_text_dataset(train_frame, tokenizer, cfg.max_length)
    val_ds = dataframe_to_text_dataset(splits['validation'], tokenizer, cfg.max_length)
    test_ds = dataframe_to_text_dataset(splits['test'], tokenizer, cfg.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    def make_loader(ds, batch_size, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda feats: collate_batch(feats, collator), num_workers=0)
    train_loader = make_loader(train_ds, cfg.train_batch_size, True)
    val_loader = make_loader(val_ds, cfg.eval_batch_size, False)
    test_loader = make_loader(test_ds, cfg.eval_batch_size, False)
    model = BertForFraudClassification(cfg.pretrained_model_name, num_labels=2, dropout=cfg.dropout, gradient_checkpointing=cfg.gradient_checkpointing).to(device)
    if cfg.use_class_weights:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        imbalance_name = f'Class weight ({cfg.class_weight_transform})' if cfg.fraud_oversample_factor <= 1.0 else f'Oversample×{cfg.fraud_oversample_factor:g}+CW({cfg.class_weight_transform})'
    else:
        criterion = nn.CrossEntropyLoss()
        imbalance_name = 'None' if cfg.fraud_oversample_factor <= 1.0 else f'Oversample×{cfg.fraud_oversample_factor:g}'
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_update_steps = (len(train_loader) + cfg.gradient_accumulation_steps - 1) // cfg.gradient_accumulation_steps * cfg.num_train_epochs
    warmup_steps = int(total_update_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)
    scaler = GradScaler(enabled=cfg.fp16 and device.type == 'cuda')
    history_rows: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = -1
    patience_left = cfg.early_stopping_patience
    output_dir = Path(cfg.output_dir)
    best_dir = output_dir / 'best'
    output_dir.mkdir(parents=True, exist_ok=True)
    total_timer = Timer()
    logger.info('Start fine-tuning: model=%s use_class_weights=%s max_length=%s batch=%s accum=%s', cfg.pretrained_model_name, cfg.use_class_weights, cfg.max_length, cfg.train_batch_size, cfg.gradient_accumulation_steps)
    for epoch in range(1, cfg.num_train_epochs + 1):
        epoch_timer = Timer()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion, scaler, cfg.gradient_accumulation_steps, cfg.max_grad_norm, cfg.fp16)
        val_metrics, _, _, _, _, _, _ = evaluate_loader(model, val_loader, device, criterion=criterion, use_amp=cfg.fp16)
        lr_now = float(scheduler.get_last_lr()[0])
        mem = gpu_memory_mb()
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_metrics['loss'], 'accuracy': val_metrics['accuracy'], 'legitimate_precision': val_metrics['legitimate_precision'], 'legitimate_recall': val_metrics['legitimate_recall'], 'legitimate_f1': val_metrics['legitimate_f1'], 'fraud_precision': val_metrics['fraud_precision'], 'fraud_recall': val_metrics['fraud_recall'], 'fraud_f1': val_metrics['fraud_f1'], 'macro_f1': val_metrics['macro_f1'], 'weighted_f1': val_metrics['weighted_f1'], 'roc_auc': val_metrics['roc_auc'], 'pr_auc': val_metrics['pr_auc'], 'learning_rate': lr_now, 'gpu_memory_mb': mem, 'epoch_duration_sec': epoch_timer.elapsed(), 'best_val_fraud_f1': max(best_metric, val_metrics['fraud_f1'])}
        history_rows.append(row)
        logger.info('Epoch %s | train_loss=%.4f val_loss=%.4f fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f', epoch, train_loss, val_metrics['loss'], val_metrics['fraud_f1'], val_metrics['fraud_recall'], val_metrics['pr_auc'])
        selection_value = float(val_metrics.get(cfg.selection_metric, val_metrics['fraud_f1']))
        if selection_value > best_metric:
            best_metric = selection_value
            best_epoch = epoch
            patience_left = cfg.early_stopping_patience
            save_checkpoint(best_dir, model, tokenizer, cfg, {'best_epoch': best_epoch, 'best_validation_metric': best_metric, 'selection_metric': cfg.selection_metric, 'imbalance_method': imbalance_name, 'class_weights': class_weights.tolist()})
            logger.info('New best checkpoint saved at epoch %s (%s=%.4f)', epoch, cfg.selection_metric, best_metric)
        else:
            patience_left -= 1
            logger.info('No improvement. Patience left: %s', patience_left)
            if patience_left <= 0:
                logger.info('Early stopping at epoch %s', epoch)
                break
    history_df = pd.DataFrame(history_rows)
    history_path = results_root / f'training_history_{cfg.run_name}.csv'
    history_df.to_csv(history_path, index=False)
    write_aliases = bool(cfg.write_canonical_aliases)
    if write_aliases:
        history_df.to_csv(results_root / 'training_history.csv', index=False)
    plot_training_curves(history_df, figures_root / f'training_loss_{cfg.run_name}.png')
    if write_aliases:
        plot_training_curves(history_df, figures_root / 'training_loss.png')
    logger.info('Loading best checkpoint from %s', best_dir)
    model.model = type(model.model).from_pretrained(best_dir)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    val_metrics_05, y_val, p_val, val_ids, val_titles, val_companies, val_idx = evaluate_loader(model, val_loader, device, criterion=criterion, use_amp=cfg.fp16)
    mode = 'fixed'
    if cfg.optimize_threshold:
        mode = getattr(cfg, 'threshold_selection_mode', None) or 'max_fraud_f1'
        if cfg.min_fraud_recall_for_threshold is not None and mode in {'max_fraud_f1', 'max_fbeta', 'near_max_f1_prefer_precision'}:
            if mode == 'max_fraud_f1':
                mode = 'min_recall_then_f1'
        best_thr, sweep = select_threshold(y_val, p_val, mode=mode, min_fraud_recall=cfg.min_fraud_recall_for_threshold, step=cfg.threshold_step, f_beta=getattr(cfg, 'threshold_f_beta', 0.5), f1_tolerance=getattr(cfg, 'threshold_f1_tolerance', 0.01))
    else:
        best_thr = cfg.default_threshold
        sweep = None
    if sweep is not None:
        sweep.to_csv(results_root / f'threshold_sweep_{cfg.run_name}.csv', index=False)
    val_metrics_opt = binary_metrics(y_val, p_val, threshold=best_thr)
    logger.info('Validation threshold selected on validation only: %.4f (fraud_f1=%.4f precision=%.4f recall=%.4f mode=%s f_beta=%s min_recall=%s step=%s)', best_thr, val_metrics_opt['fraud_f1'], val_metrics_opt['fraud_precision'], val_metrics_opt['fraud_recall'], mode, getattr(cfg, 'threshold_f_beta', None), cfg.min_fraud_recall_for_threshold, cfg.threshold_step)
    infer_timer = Timer()
    test_metrics_05, y_test, p_test, test_ids, test_titles, test_companies, test_idx = evaluate_loader(model, test_loader, device, criterion=criterion, use_amp=cfg.fp16)
    inference_time = infer_timer.elapsed()
    test_metrics = binary_metrics(y_test, p_test, threshold=best_thr)
    test_metrics['loss'] = test_metrics_05['loss']
    test_metrics['threshold_default_0_5'] = binary_metrics(y_test, p_test, threshold=0.5)
    test_metrics['threshold_optimized'] = {'threshold': best_thr, **{k: v for k, v in test_metrics.items() if k not in {'threshold_default_0_5', 'threshold_optimized'}}}
    preds = build_predictions_frame(record_ids=test_ids, titles=test_titles, companies=test_companies, y_true=y_test, y_prob=p_test, threshold=best_thr, model_name=f'{cfg.model_label}:{cfg.run_name}', imbalance_strategy=imbalance_name, row_indices=test_idx)
    pred_path = results_root / f'predictions_{cfg.run_name}.csv'
    preds.to_csv(pred_path, index=False)
    if write_aliases:
        preds.to_csv(results_root / 'predictions.csv', index=False)
    if getattr(cfg, 'write_error_analysis', True):
        errors = build_error_analysis(preds, splits['test']['model_text'].tolist())
        err_path = results_root / f'error_analysis_{cfg.run_name}.csv'
        errors.to_csv(err_path, index=False)
        if write_aliases:
            errors.to_csv(results_root / 'error_analysis.csv', index=False)
    plot_confusion_matrix(test_metrics['confusion_matrix'], figures_root / f'confusion_matrix_{cfg.run_name}.png', title=f'{cfg.model_label} test CM ({cfg.run_name})')
    if write_aliases:
        plot_confusion_matrix(test_metrics['confusion_matrix'], figures_root / 'confusion_matrix.png', title='BERT test Confusion Matrix')
    plot_roc_pr(y_test, p_test, figures_root / f'roc_curve_{cfg.run_name}.png', figures_root / f'precision_recall_curve_{cfg.run_name}.png')
    if write_aliases:
        plot_roc_pr(y_test, p_test, figures_root / 'roc_curve.png', figures_root / 'precision_recall_curve.png')
    metrics_path = results_root / f'test_metrics_{cfg.run_name}.json'
    payload = {'model': f'{cfg.model_label} fine-tuned', 'run_name': cfg.run_name, 'imbalance_method': imbalance_name, 'pretrained_model_name': cfg.pretrained_model_name, 'best_epoch': best_epoch, 'best_validation_metric': best_metric, 'selection_metric': cfg.selection_metric, 'threshold': best_thr, 'validation_metrics_threshold_0_5': val_metrics_05, 'validation_metrics_optimized_threshold': val_metrics_opt, 'test_metrics': test_metrics, 'inference_time_sec': inference_time, 'training_time_sec': total_timer.elapsed(), 'class_weights_train_only': class_weights.tolist()}
    save_json(payload, metrics_path)
    if write_aliases:
        save_json(payload, results_root / 'test_metrics.json')
    save_comparison_row(results_root / 'model_comparison.csv', {'Model': cfg.model_label, 'Imbalance method': imbalance_name, 'Accuracy': test_metrics['accuracy'], 'Fraud Precision': test_metrics['fraud_precision'], 'Fraud Recall': test_metrics['fraud_recall'], 'Fraud F1': test_metrics['fraud_f1'], 'Macro F1': test_metrics['macro_f1'], 'ROC-AUC': test_metrics['roc_auc'], 'PR-AUC': test_metrics['pr_auc'], 'Inference time (s)': inference_time, 'Notes': f'run={cfg.run_name}; thr={best_thr:.4f}; best_epoch={best_epoch}'})
    plot_model_comparison(results_root / 'model_comparison.csv', figures_root / 'model_comparison.png')
    run_config = {'experiment': f'{cfg.model_label.lower()}_finetune', 'paths': default_paths_dict(), 'config': config_to_dict(cfg), 'environment': collect_environment_info(), 'best_checkpoint': str(best_dir), 'best_epoch': best_epoch, 'best_validation_metric': best_metric, 'threshold': best_thr, 'token_length_stats': token_stats, 'training_time_sec': total_timer.elapsed(), 'imbalance_method': imbalance_name, 'class_weights': class_weights.tolist(), 'sample_counts': {k: {'n': len(v), 'fraud': int((v[LABEL_COLUMN] == 1).sum())} for k, v in splits.items()}, 'train_oversample': {'factor': float(cfg.fraud_oversample_factor), 'n_before': n_train_before, 'n_after': n_train_after, 'fraud_before': n_fraud_before, 'fraud_after': n_fraud_after}, 'class_weight_transform': cfg.class_weight_transform, 'class_weight_max': cfg.class_weight_max, 'threshold_step': cfg.threshold_step, 'min_fraud_recall_for_threshold': cfg.min_fraud_recall_for_threshold, 'results_dir': str(results_root), 'figures_dir': str(figures_root)}
    save_json(run_config, results_root / f'run_config_{cfg.run_name}.json')
    if write_aliases:
        save_json(run_config, results_root / 'run_config.json')
    save_json({'threshold': best_thr, 'default_threshold': 0.5, 'model_name': cfg.pretrained_model_name, 'run_name': cfg.run_name, 'imbalance_method': imbalance_name}, best_dir / 'threshold.json')
    logger.info('Test fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f', test_metrics['fraud_f1'], test_metrics['fraud_recall'], test_metrics['pr_auc'])
    logger.info('Done. Artifacts under %s and %s', output_dir, results_root)
    return payload

def cmd_evaluate(args: argparse.Namespace) -> None:
    ensure_directories()
    logger = setup_logging('training.log')
    set_seed(args.random_seed)
    device = get_device(allow_cpu=args.allow_cpu, logger=logger)
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
    thr_path = ckpt / 'threshold.json'
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif thr_path.exists():
        threshold = float(load_json(thr_path)['threshold'])
    else:
        threshold = 0.5
        logger.warning('No threshold.json found; using default 0.5')
    model_label = getattr(args, 'model_label', None) or MODEL_LABEL
    split_arg = getattr(args, 'split', 'both')
    if split_arg == 'both':
        splits = ['validation', 'test']
    else:
        splits = [split_arg]
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    model.model = type(model.model).from_pretrained(ckpt)
    model = model.to(device)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    summary: Dict[str, Any] = {'model': model_label, 'checkpoint': str(ckpt), 'threshold': threshold, 'max_length': args.max_length}
    for split in splits:
        split_df = load_split(split)
        ds = dataframe_to_text_dataset(split_df, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=lambda feats: collate_batch(feats, collator), num_workers=0)
        _, y_true, y_prob, ids, titles, companies, idxs = evaluate_loader(model, loader, device, criterion=None, use_amp=device.type == 'cuda')
        metrics = binary_metrics(y_true, y_prob, threshold=threshold)
        metrics_05 = binary_metrics(y_true, y_prob, threshold=0.5)
        preds = build_predictions_frame(record_ids=ids, titles=titles, companies=companies, y_true=y_true, y_prob=y_prob, threshold=threshold, model_name=model_label, imbalance_strategy='checkpoint', row_indices=idxs)
        preds.to_csv(RESULTS_DIR / f'predictions_{split}.csv', index=False)
        errors = build_error_analysis(preds, split_df['model_text'].tolist())
        errors.to_csv(RESULTS_DIR / f'error_analysis_{split}.csv', index=False)
        plot_confusion_matrix(metrics['confusion_matrix'], FIGURES_DIR / f'confusion_matrix_{split}.png', title=f'{model_label} {split} Confusion Matrix')
        plot_roc_pr(y_true, y_prob, FIGURES_DIR / f'roc_curve_{split}.png', FIGURES_DIR / f'precision_recall_curve_{split}.png')
        macro = metrics.get('classification_report', {}).get('macro avg', {})
        payload = {'model': model_label, 'checkpoint': str(ckpt), 'split': split, 'threshold': threshold, 'metrics': metrics, 'metrics_threshold_0_5': metrics_05, 'macro_precision': macro.get('precision'), 'macro_recall': macro.get('recall'), 'macro_f1': macro.get('f1-score', metrics.get('macro_f1'))}
        save_json(payload, RESULTS_DIR / f'metrics_{split}.json')
        summary[f'{split}_metrics'] = {'fraud_precision': metrics['fraud_precision'], 'fraud_recall': metrics['fraud_recall'], 'fraud_f1': metrics['fraud_f1'], 'macro_precision': payload['macro_precision'], 'macro_recall': payload['macro_recall'], 'macro_f1': payload['macro_f1'], 'pr_auc': metrics['pr_auc'], 'roc_auc': metrics['roc_auc'], 'accuracy': metrics['accuracy']}
        logger.info('%s fraud_f1=%.4f fraud_recall=%.4f macro_f1=%.4f pr_auc=%.4f accuracy=%.4f', split, metrics['fraud_f1'], metrics['fraud_recall'], float(payload['macro_f1'] or 0.0), metrics['pr_auc'], metrics['accuracy'])
        print(f"[{split}] Fraud F1: {metrics['fraud_f1']:.4f} | Macro F1: {float(payload['macro_f1'] or 0.0):.4f} | Threshold: {threshold:.4f}")
        if split == 'validation':
            val_export = pd.DataFrame({'record_id': list(ids), 'label': y_true.astype(int), 'fraud_score': y_prob.astype(float)})
            val_export.to_csv(RESULTS_DIR / 'validation_predictions.csv', index=False)
    save_json(summary, RESULTS_DIR / 'metrics_summary.json')
    logger.info('Wrote clean evaluate artifacts under %s', RESULTS_DIR)

def load_predictor(checkpoint_dir: Path, device: torch.device, threshold_override: Optional[float]=None):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    wrapper = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    wrapper.model = type(wrapper.model).from_pretrained(checkpoint_dir)
    wrapper = wrapper.to(device)
    wrapper.eval()
    thr_path = checkpoint_dir / 'threshold.json'
    meta_path = checkpoint_dir / 'train_meta.json'
    threshold = 0.5
    model_name = PRETRAINED_MODEL_NAME
    if thr_path.exists():
        thr_obj = load_json(thr_path)
        threshold = float(thr_obj.get('threshold', 0.5))
        model_name = thr_obj.get('model_name', model_name)
    if threshold_override is not None:
        threshold = float(threshold_override)
    meta = load_json(meta_path) if meta_path.exists() else {}
    return (wrapper, tokenizer, threshold, model_name, meta)

@torch.no_grad()
def predict_texts(texts: List[str], model: BertForFraudClassification, tokenizer, device: torch.device, threshold: float, max_length: int=384) -> List[Dict[str, Any]]:
    model.eval()
    results = []
    for text in texts:
        cleaned = clean_text(text)
        enc = tokenizer(cleaned, truncation=True, max_length=max_length, padding=True, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        fraud_p = float(softmax_fraud_proba(logits)[0].cpu().item())
        legit_p = 1.0 - fraud_p
        pred = 1 if fraud_p >= threshold else 0
        label = 'Fraudulent' if pred == 1 else 'Legitimate'
        results.append({'predicted_label': label, 'predicted_label_id': pred, 'fraud_probability': fraud_p, 'legitimate_probability': legit_p, 'fraud_risk_score': fraud_p, 'threshold': threshold})
    return results

def format_prediction(result: Dict[str, Any], model_name: str) -> str:
    return f"Predicted label: {result['predicted_label']}\nFraud probability: {result['fraud_probability']:.4f}\nLegitimate probability: {result['legitimate_probability']:.4f}\nFraud risk score: {result['fraud_risk_score']:.4f}\nThreshold: {result['threshold']:.4f}\nModel: {model_name}"

def cmd_predict(args: argparse.Namespace) -> None:
    ensure_directories()
    logger = setup_logging('training.log')
    device = get_device(allow_cpu=args.allow_cpu, logger=logger)
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}. Train first with: python BERT/code/run.py train')
    model, tokenizer, threshold, model_name, _meta = load_predictor(ckpt, device, threshold_override=args.threshold)
    if args.text:
        result = predict_texts([args.text], model, tokenizer, device, threshold, args.max_length)[0]
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
        results = predict_texts(texts, model, tokenizer, device, threshold, args.max_length)
        out = df.copy()
        out['predicted_label'] = [r['predicted_label'] for r in results]
        out['fraud_probability'] = [r['fraud_probability'] for r in results]
        out['legitimate_probability'] = [r['legitimate_probability'] for r in results]
        out['fraud_risk_score'] = [r['fraud_risk_score'] for r in results]
        out['threshold'] = threshold
        out_path = Path(args.output_csv) if args.output_csv else Path(args.csv).with_name(Path(args.csv).stem + '_bert_predictions.csv')
        out.to_csv(out_path, index=False)
        print(f'Wrote predictions to {out_path}')
        return
    raise SystemExit('predict requires --text or --csv')

def cmd_export_val(args: argparse.Namespace) -> None:
    ensure_directories()
    logger = setup_logging('export_validation_predictions.log')
    set_seed(args.random_seed)
    device = get_device(allow_cpu=args.allow_cpu, logger=logger)
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    model.model = type(model.model).from_pretrained(ckpt)
    model = model.to(device)
    val_df = load_split('validation')
    ds = dataframe_to_text_dataset(val_df, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=lambda feats: collate_batch(feats, collator), num_workers=0)
    _, y_true, y_prob, ids, *_ = evaluate_loader(model, loader, device, criterion=None, use_amp=device.type == 'cuda')
    out = pd.DataFrame({'record_id': list(ids), 'label': y_true.astype(int), 'fraud_score': y_prob.astype(float)})
    out_path = RESULTS_DIR / 'validation_predictions.csv'
    out.to_csv(out_path, index=False)
    logger.info('Wrote %s (%s rows)', out_path, len(out))
    legacy = RESULTS_DIR / 'bert_validation_predictions.csv'
    out.to_csv(legacy, index=False)
    if ENSEMBLE_VAL_COPY.parent.exists():
        shutil.copy2(out_path, ENSEMBLE_VAL_COPY)
        logger.info('Also refreshed ensemble copy: %s', ENSEMBLE_VAL_COPY)

def cmd_train(args: argparse.Namespace) -> None:
    run_name = args.run_name or RUN_NAME
    model_label = getattr(args, 'model_label', None) or MODEL_LABEL
    cfg = BertFinetuneConfig(pretrained_model_name=args.pretrained_model_name, learning_rate=args.learning_rate, weight_decay=args.weight_decay, num_train_epochs=args.num_train_epochs, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, max_length=args.max_length, warmup_ratio=args.warmup_ratio, early_stopping_patience=args.early_stopping_patience, dropout=args.dropout, random_seed=args.random_seed, fp16=args.fp16, use_class_weights=args.use_class_weights, class_weight_transform=args.class_weight_transform, class_weight_max=args.class_weight_max, fraud_oversample_factor=args.fraud_oversample_factor, gradient_checkpointing=args.gradient_checkpointing, optimize_threshold=args.optimize_threshold, threshold_selection_mode=args.threshold_selection_mode, threshold_f_beta=args.threshold_f_beta, min_fraud_recall_for_threshold=args.min_fraud_recall_for_threshold, threshold_step=args.threshold_step, output_dir=args.output_dir, run_name=run_name, model_label=model_label)
    try:
        run_training(cfg)
    except SystemExit:
        raise
    except Exception:
        logger = setup_logging('training.log')
        logger.error('Training failed:\n%s', traceback.format_exc())
        raise

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='BERT fraudulent job-ad detection (train / evaluate / predict / export-val)')
    sub = parser.add_subparsers(dest='command', required=True)
    p_train = sub.add_parser('train', help='Fine-tune BERT on fixed train/val/test splits')
    p_train.add_argument('--pretrained_model_name', default=PRETRAINED_MODEL_NAME)
    p_train.add_argument('--learning_rate', type=float, default=2e-05)
    p_train.add_argument('--weight_decay', type=float, default=0.01)
    p_train.add_argument('--num_train_epochs', type=int, default=5)
    p_train.add_argument('--train_batch_size', type=int, default=4)
    p_train.add_argument('--eval_batch_size', type=int, default=8)
    p_train.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p_train.add_argument('--max_length', type=int, default=MAX_LENGTH)
    p_train.add_argument('--warmup_ratio', type=float, default=0.1)
    p_train.add_argument('--early_stopping_patience', type=int, default=2)
    p_train.add_argument('--dropout', type=float, default=0.1)
    p_train.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    p_train.add_argument('--fp16', type=parse_bool, default=True)
    p_train.add_argument('--use_class_weights', type=parse_bool, default=True)
    p_train.add_argument('--class_weight_transform', type=str, default='sqrt_clip', choices=['none', 'sqrt', 'clip', 'sqrt_clip'])
    p_train.add_argument('--class_weight_max', type=float, default=5.0)
    p_train.add_argument('--fraud_oversample_factor', type=float, default=3.0)
    p_train.add_argument('--gradient_checkpointing', type=parse_bool, default=False)
    p_train.add_argument('--optimize_threshold', type=parse_bool, default=True)
    p_train.add_argument('--threshold_selection_mode', type=str, default='max_fbeta', choices=['max_fraud_f1', 'max_fbeta', 'near_max_f1_prefer_precision', 'min_recall_then_f1', 'min_recall_then_precision'])
    p_train.add_argument('--threshold_f_beta', type=float, default=0.5)
    p_train.add_argument('--min_fraud_recall_for_threshold', type=float, default=0.85)
    p_train.add_argument('--threshold_step', type=float, default=0.01)
    p_train.add_argument('--run_name', type=str, default=None)
    p_train.add_argument('--model_label', type=str, default=MODEL_LABEL)
    p_train.add_argument('--output_dir', type=str, default=str(BERT_FINETUNED_DIR))
    p_train.set_defaults(func=cmd_train)
    p_eval = sub.add_parser('evaluate', help='Evaluate checkpoint on validation and/or test splits')
    p_eval.add_argument('--checkpoint_dir', type=str, default=str(BERT_FINETUNED_DIR / 'best'))
    p_eval.add_argument('--split', type=str, default='both', choices=['validation', 'test', 'both'], help='Which split(s) to evaluate (default: both)')
    p_eval.add_argument('--model_label', type=str, default=MODEL_LABEL)
    p_eval.add_argument('--threshold', type=float, default=None)
    p_eval.add_argument('--eval_batch_size', type=int, default=8)
    p_eval.add_argument('--max_length', type=int, default=512)
    p_eval.add_argument('--allow_cpu', action='store_true')
    p_eval.add_argument('--random_seed', type=int, default=42)
    p_eval.set_defaults(func=cmd_evaluate)
    p_pred = sub.add_parser('predict', help='Predict on a single text or CSV of ads')
    p_pred.add_argument('--checkpoint_dir', type=str, default=str(BERT_FINETUNED_DIR / 'best'))
    p_pred.add_argument('--text', type=str, default=None, help='Single job-ad text')
    p_pred.add_argument('--csv', type=str, default=None, help='Optional CSV of ads')
    p_pred.add_argument('--text_column', type=str, default='combined_text')
    p_pred.add_argument('--output_csv', type=str, default=None)
    p_pred.add_argument('--threshold', type=float, default=None)
    p_pred.add_argument('--max_length', type=int, default=MAX_LENGTH)
    p_pred.add_argument('--allow_cpu', action='store_true', default=True)
    p_pred.set_defaults(func=cmd_predict)
    p_exp = sub.add_parser('export-val', help='Export validation fraud scores for ensemble FP-gate')
    p_exp.add_argument('--checkpoint_dir', type=str, default=str(BERT_FINETUNED_DIR / 'best'))
    p_exp.add_argument('--eval_batch_size', type=int, default=16)
    p_exp.add_argument('--max_length', type=int, default=512)
    p_exp.add_argument('--allow_cpu', action='store_true')
    p_exp.add_argument('--random_seed', type=int, default=42)
    p_exp.set_defaults(func=cmd_export_val)
    return parser
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
if __name__ == '__main__':
    main()