#Ensemble BERT + FP-gate for EMSCAD data
from __future__ import annotations
import json, os, random, sys
from pathlib import Path
from typing import Any, Iterable, Optional
os.environ.setdefault('PYTHONHASHSEED', '42')
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'results'
SPRINT3 = ROOT.parent
LR_ROOT = SPRINT3 / 'LR'
BERT_ROOT = SPRINT3 / 'BERT'
LR_VAL = LR_ROOT / 'results' / 'validation_predictions.csv'
LR_TEST = LR_ROOT / 'results' / 'test_predictions.csv'
BERT_VAL_CANDIDATES = (BERT_ROOT / 'results' / 'validation_predictions.csv', BERT_ROOT / 'results' / 'bert_validation_predictions.csv', ROOT / 'results' / 'bert_validation_predictions.csv', ROOT / 'results' / 'validation_predictions.csv')
BERT_TEST_CANDIDATES = (BERT_ROOT / 'results' / 'predictions_test.csv', BERT_ROOT / 'results' / 'predictions_bert_paper_protocol_maxlen512.csv')
BERT_TEST_METRICS_CANDIDATES = (BERT_ROOT / 'results' / 'metrics_test.json', BERT_ROOT / 'results' / 'bert-paper-protocol_test_metrics_bert_paper_protocol_maxlen512.json')
LR_TEST_METRICS = LR_ROOT / 'results' / 'test_metrics.json'
SEED = 42
BERT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
LR_GATES = np.unique(np.concatenate([np.array([0.0]), np.round(np.arange(0.01, 0.51, 0.01), 2), np.round(np.arange(0.55, 1.01, 0.05), 2)]))

def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None

def require_existing(paths: Iterable[Path], label: str) -> Path:
    found = first_existing(paths)
    if found is not None:
        return found
    tried = '\n'.join((f'  - {p}' for p in paths))
    raise FileNotFoundError(f'Missing {label}. Tried:\n{tried}')

def load_bert_reference_metrics(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if 'metrics' in raw and isinstance(raw['metrics'], dict):
        metrics = dict(raw['metrics'])
        report = metrics.get('classification_report') or {}
        macro = report.get('macro avg') or {}
        return {'fraud_precision': float(metrics['fraud_precision']), 'fraud_recall': float(metrics['fraud_recall']), 'fraud_f1': float(metrics['fraud_f1']), 'macro_f1': float(metrics.get('macro_f1', macro.get('f1-score', 0.0))), 'macro_precision': float(raw.get('macro_precision', macro.get('precision', 0.0))), 'macro_recall': float(raw.get('macro_recall', macro.get('recall', 0.0))), 'pr_auc': float(metrics['pr_auc']), 'roc_auc': float(metrics['roc_auc']), 'classification_report': report, 'threshold': float(raw.get('threshold', metrics.get('threshold', 0.5))), 'source': str(path)}
    if 'test_metrics' in raw:
        opt = raw['test_metrics'].get('threshold_optimized') or raw['test_metrics']
        report = opt.get('classification_report') or {}
        macro = report.get('macro avg') or {}
        return {'fraud_precision': float(opt['fraud_precision']), 'fraud_recall': float(opt['fraud_recall']), 'fraud_f1': float(opt['fraud_f1']), 'macro_f1': float(opt.get('macro_f1', macro.get('f1-score', 0.0))), 'macro_precision': float(macro.get('precision', 0.0)), 'macro_recall': float(macro.get('recall', 0.0)), 'pr_auc': float(opt['pr_auc']), 'roc_auc': float(opt['roc_auc']), 'classification_report': report, 'threshold': float(opt.get('threshold', raw.get('threshold', 0.5))), 'source': str(path)}
    raise ValueError(f'Unrecognized BERT metrics schema: {path}')
SEED = 42
BERT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
LR_GATES = np.unique(np.concatenate([np.array([0.0]), np.round(np.arange(0.01, 0.51, 0.01), 2), np.round(np.arange(0.55, 1.01, 0.05), 2)]))

def set_reproducibility(seed: int=SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

def package_versions() -> dict[str, str]:
    import sklearn
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__}

def align_scores(lr_path: Path, bert_path: Path, split: str) -> pd.DataFrame:
    lr = pd.read_csv(lr_path).rename(columns={'fraud_score': 'lr_score'})
    bert = pd.read_csv(bert_path).rename(columns={'original_id': 'record_id', 'true_label': 'label', 'fraud_probability': 'bert_score', 'fraud_score': 'bert_score'})
    lr = lr[['record_id', 'label', 'lr_score']]
    bert = bert[['record_id', 'label', 'bert_score']]
    aligned = lr.merge(bert, on=['record_id', 'label'], how='inner', validate='one_to_one')
    if len(aligned) != len(lr) or len(aligned) != len(bert):
        raise ValueError(f'{split}: LR and BERT rows are not fully aligned')
    if aligned['record_id'].duplicated().any():
        raise ValueError(f'{split}: duplicate record IDs found')
    return aligned

def apply_gate(lr_scores: np.ndarray, bert_scores: np.ndarray, bert_threshold: float, lr_gate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bert_pred = (bert_scores >= bert_threshold).astype(int)
    gated = (bert_pred == 1) & (lr_scores < lr_gate)
    final_pred = bert_pred.copy()
    final_pred[gated] = 0
    ranking_scores = bert_scores.copy()
    ranking_scores[gated] = np.minimum(bert_scores[gated], lr_scores[gated])
    return (final_pred, gated, ranking_scores)

def metrics_from_predictions(labels: np.ndarray, predictions: np.ndarray, ranking_scores: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {'accuracy': float(accuracy_score(labels, predictions)), 'roc_auc': float(roc_auc_score(labels, ranking_scores)), 'pr_auc': float(average_precision_score(labels, ranking_scores)), 'macro_precision': float(np.mean(precision)), 'macro_recall': float(np.mean(recall)), 'macro_f1': float(np.mean(f1)), 'fraud_precision': float(precision[1]), 'fraud_recall': float(recall[1]), 'fraud_f1': float(f1[1]), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def main() -> None:
    set_reproducibility(SEED)
    bert_val = require_existing(BERT_VAL_CANDIDATES, 'BERT validation predictions')
    bert_test = require_existing(BERT_TEST_CANDIDATES, 'BERT test predictions')
    bert_test_metrics_path = require_existing(BERT_TEST_METRICS_CANDIDATES, 'BERT test metrics')
    if not LR_VAL.exists():
        raise FileNotFoundError(f'Missing LR validation predictions: {LR_VAL}')
    if not LR_TEST.exists():
        raise FileNotFoundError(f'Missing LR test predictions: {LR_TEST}')
    if not LR_TEST_METRICS.exists():
        raise FileNotFoundError(f'Missing LR test metrics: {LR_TEST_METRICS}\nRun ../LR training first to regenerate predictions/metrics.')
    validation = align_scores(LR_VAL, bert_val, 'Validation')
    test = align_scores(LR_TEST, bert_test, 'Test')
    if set(validation['record_id']) & set(test['record_id']):
        raise ValueError('Validation and Test IDs overlap')
    y_val = validation['label'].to_numpy(dtype=int)
    lr_val = validation['lr_score'].to_numpy()
    bert_val_scores = validation['bert_score'].to_numpy()
    sweep_rows = []
    for bert_threshold in BERT_THRESHOLDS:
        for lr_gate in LR_GATES:
            preds, gated, ranks = apply_gate(lr_val, bert_val_scores, float(bert_threshold), float(lr_gate))
            metrics = metrics_from_predictions(y_val, preds, ranks)
            sweep_rows.append({'bert_threshold': float(bert_threshold), 'lr_gate': float(lr_gate), 'gate_flips': int(gated.sum()), 'gate_rate': float(gated.mean()), **metrics})
    sweep = pd.DataFrame(sweep_rows)
    selected = sweep.sort_values(['fraud_f1', 'fraud_recall', 'fraud_precision', 'pr_auc'], ascending=False).iloc[0]
    bert_threshold = float(selected['bert_threshold'])
    lr_gate = float(selected['lr_gate'])
    val_preds, val_gated, val_ranks = apply_gate(lr_val, bert_val_scores, bert_threshold, lr_gate)
    validation_metrics = metrics_from_predictions(y_val, val_preds, val_ranks)
    validation_metrics['bert_threshold'] = bert_threshold
    validation_metrics['lr_gate'] = lr_gate
    validation_metrics['gate_flips'] = int(val_gated.sum())
    bert_only = sweep[np.isclose(sweep['lr_gate'], 0.0)].sort_values(['fraud_f1', 'fraud_recall', 'fraud_precision', 'pr_auc'], ascending=False).iloc[0]
    y_test = test['label'].to_numpy(dtype=int)
    lr_test = test['lr_score'].to_numpy()
    bert_test_scores = test['bert_score'].to_numpy()
    test_preds, test_gated, test_ranks = apply_gate(lr_test, bert_test_scores, bert_threshold, lr_gate)
    test_metrics = metrics_from_predictions(y_test, test_preds, test_ranks)
    test_metrics['bert_threshold'] = bert_threshold
    test_metrics['lr_gate'] = lr_gate
    test_metrics['gate_flips'] = int(test_gated.sum())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUTPUT_DIR / 'validation_sweep.csv', index=False)
    validation.assign(prediction=val_preds, gated_flip=val_gated.astype(int), ranking_score=val_ranks).to_csv(OUTPUT_DIR / 'validation_predictions.csv', index=False)
    test.assign(prediction=test_preds, gated_flip=test_gated.astype(int), ranking_score=test_ranks).to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)
    config = {'experiment': 'ensemble_bert_fp_gate_lr_none_bigram_maxlen512', 'method': 'BERT decision + flip to legit when LR_score < lr_gate', 'lr_source': str(LR_ROOT.name), 'bert_source': str(BERT_ROOT.name), 'bert_threshold': bert_threshold, 'lr_gate': lr_gate, 'selection_set': 'validation', 'selection_metric': 'fraud_f1', 'test_used_for_selection': False, 'reported_test_models': ['LR (class_weight=None, bigram, no CV)', 'Optimized BERT max_length=512', 'BERT + LR(None/bigram) FP-gate'], 'reproducibility': {'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'random_seed': SEED, 'frozen_lr_predictions': str(LR_TEST), 'frozen_bert_predictions': str(bert_test), 'frozen_bert_validation_predictions': str(bert_val), 'frozen_bert_test_metrics': str(bert_test_metrics_path), 'package_versions': package_versions()}, 'validation_rows': int(len(validation)), 'test_rows': int(len(test)), 'validation_gate_flips': int(val_gated.sum()), 'test_gate_flips': int(test_gated.sum()), 'validation_bert_only_best_fraud_f1': float(bert_only['fraud_f1']), 'validation_bert_only_best_threshold': float(bert_only['bert_threshold']), 'validation_selected_fraud_f1': float(validation_metrics['fraud_f1'])}
    for filename, payload in (('config.json', config), ('validation_metrics.json', validation_metrics), ('test_metrics.json', test_metrics)):
        (OUTPUT_DIR / filename).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    legacy = OUTPUT_DIR / 'test_bert_at_selected_threshold.json'
    if legacy.exists():
        legacy.unlink()
    lr_ref = json.loads(LR_TEST_METRICS.read_text(encoding='utf-8'))
    bert_opt = load_bert_reference_metrics(bert_test_metrics_path)
    improved_macro = test_metrics['macro_f1'] > bert_opt['macro_f1'] + 1e-12
    improved_fraud = test_metrics['fraud_f1'] > bert_opt['fraud_f1'] + 1e-12
    results_md = f"# BERT + LR(None/bigram/no-CV) FP-gate ensemble\n\n## Method\n\n1. Predict with Optimized BERT (`max_length=512`) using `bert_threshold`\n2. If BERT predicts fraud **and** `LR_score < lr_gate`, flip to legitimate\n3. Select `(bert_threshold, lr_gate)` on Validation by Fraud F1\n4. Freeze and evaluate once on Test\n\nLR branch: `../LR` (`class_weight=None`, unigram+bigram, no Train CV).\n\nBERT branch: `../BERT` (Optimized BERT, `max_length=512`).\n\n## Reproducibility\n\n- Selection seed / hash seed: `{SEED}` / `{os.environ.get('PYTHONHASHSEED')}`\n- Frozen LR predictions: `{LR_TEST}`\n- Frozen BERT predictions: `{bert_test}`\n- Frozen BERT metrics: `{bert_test_metrics_path}`\n- Packages: `{package_versions()}`\n- Test report keeps only: LR / BERT / this ensemble\n\n## Selected configuration\n\n- BERT threshold: `{bert_threshold:.2f}`\n- LR gate: `{lr_gate:.2f}`\n- Validation gate flips: `{int(val_gated.sum())}/{len(validation)}`\n- Test gate flips: `{int(test_gated.sum())}/{len(test)}`\n- Validation Fraud F1: `{validation_metrics['fraud_f1']:.4f}`\n- Validation pure-BERT best Fraud F1: `{float(bert_only['fraud_f1']):.4f}` (thr {float(bert_only['bert_threshold']):.2f})\n\n## Test comparison (Macro metrics)\n\n| Model | Macro P | Macro R | Macro F1 | PR-AUC | ROC-AUC |\n|---|---:|---:|---:|---:|---:|\n| LR (`None`, bigram, no CV) | {lr_ref['macro_precision']:.4f} | {lr_ref['macro_recall']:.4f} | {lr_ref['macro_f1']:.4f} | {lr_ref['pr_auc']:.4f} | {lr_ref['roc_auc']:.4f} |\n| Optimized BERT max_length=512 | {bert_opt['macro_precision']:.4f} | {bert_opt['macro_recall']:.4f} | {bert_opt['macro_f1']:.4f} | {bert_opt['pr_auc']:.4f} | {bert_opt['roc_auc']:.4f} |\n| **BERT + LR(None/bigram) FP-gate** | {test_metrics['macro_precision']:.4f} | {test_metrics['macro_recall']:.4f} | {test_metrics['macro_f1']:.4f} | {test_metrics['pr_auc']:.4f} | {test_metrics['roc_auc']:.4f} |\n\n## Test comparison (Fraud metrics)\n\n| Model | Fraud P | Fraud R | Fraud F1 |\n|---|---:|---:|---:|\n| LR (`None`, bigram, no CV) | {lr_ref['fraud_precision']:.4f} | {lr_ref['fraud_recall']:.4f} | {lr_ref['fraud_f1']:.4f} |\n| Optimized BERT max_length=512 | {bert_opt['fraud_precision']:.4f} | {bert_opt['fraud_recall']:.4f} | {bert_opt['fraud_f1']:.4f} |\n| **BERT + LR(None/bigram) FP-gate** | {test_metrics['fraud_precision']:.4f} | {test_metrics['fraud_recall']:.4f} | {test_metrics['fraud_f1']:.4f} |\n\nMacro F1 vs Optimized BERT alone: {('higher' if improved_macro else 'not higher')} ({bert_opt['macro_f1']:.4f} → {test_metrics['macro_f1']:.4f}).\nFraud F1 vs Optimized BERT alone: {('higher' if improved_fraud else 'not higher')} ({bert_opt['fraud_f1']:.4f} → {test_metrics['fraud_f1']:.4f}).\n\nConfusion matrix on Test: TN {test_metrics['tn']}, FP {test_metrics['fp']}, FN {test_metrics['fn']}, TP {test_metrics['tp']}.\n"
    (OUTPUT_DIR / 'RESULTS.md').write_text(results_md, encoding='utf-8')
    (ROOT / 'README.md').write_text('# BERT + LR(None/bigram/no-CV) FP-gate\n\nOptimized BERT (`max_length=512`) is primary. When BERT predicts fraud and the LR\nscore is below the gate threshold, the decision is flipped to legitimate (FP gate).\n\n## Components\n\n| Branch | Path |\n|---|---|\n| LR | `../LR` (`class_weight=None`, bigram, no CV) |\n| BERT | `../BERT` (Optimized BERT) |\n| Ensemble | This directory (FP-gate) |\n\nThe test report compares only these three arms: LR / BERT / Ensemble.\n\n## Reproduce\n\nFrom `sprint3/`, activate the CUDA environment and run:\n\n```powershell\n. E:\\ml\\activate.ps1\n\n# 1) BERT: export validation scores (skip if frozen files already exist)\npython BERT/code/run.py export-val\n\n# 2) LR: reproduce predictions + metrics (skip if already present)\npython LR/code/train_lr_none_bigram_no_cv.py\n\n# 3) FP-gate ensemble\npython ensemble_BERT_FP/code/run_fp_gate_ensemble.py\n```\n\nRequired artifacts (short names preferred; legacy long names still accepted):\n\n- `../BERT/results/validation_predictions.csv` (or `bert_validation_predictions.csv`)\n- `../BERT/results/predictions_test.csv`\n- `../BERT/results/metrics_test.json`\n- `../LR/results/validation_predictions.csv`\n- `../LR/results/test_predictions.csv`\n- `../LR/results/test_metrics.json`\n\nRisk-score / risk-level docs live under sibling `../risk_score/` and `../risk_level/` when present.\n', encoding='utf-8')
    print(json.dumps({'config': config, 'validation': validation_metrics, 'test': test_metrics}, indent=2))
    print(f'Results saved to: {OUTPUT_DIR}')
if __name__ == '__main__':
    main()