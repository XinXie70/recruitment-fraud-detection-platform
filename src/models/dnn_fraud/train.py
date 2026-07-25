"""
StratifiedGroupKFold 交叉验证与最终模型训练/测试评估。

测试集不得参与特征拟合、模型选择或阈值调整。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as cfg
from .data_utils import assert_fold_no_group_overlap, warn_if_large_groups
from .features import (
    compute_balanced_class_weight,
    fit_transform_train_features,
    maybe_oversample,
    resolve_class_weight,
    transform_features,
)
from .metrics_utils import (
    CORE_METRIC_SPECS,
    evaluate_binary_predictions,
    print_evaluation_metrics,
    save_evaluation_artifacts,
    search_threshold_oof,
    summarize_cv_results,
)
from .model import build_dnn, clear_tf_session, make_callbacks, predict_proba


def _print_fold_header(
    fold_idx: int,
    y_train: np.ndarray,
    y_val: np.ndarray,
    groups_train: np.ndarray,
    groups_val: np.ndarray,
) -> None:
    n_tr, n_va = len(y_train), len(y_val)
    fraud_tr = int((y_train == 1).sum())
    fraud_va = int((y_val == 1).sum())
    print("\n" + "-" * 72)
    print(f"折 {fold_idx}/{cfg.N_SPLITS}")
    print(f"  训练集样本数量: {n_tr}")
    print(f"  验证集样本数量: {n_va}")
    print(f"  训练集欺诈样本数量: {fraud_tr}")
    print(f"  验证集欺诈样本数量: {fraud_va}")
    print(f"  训练集欺诈比例: {fraud_tr / n_tr:.6f}")
    print(f"  验证集欺诈比例: {fraud_va / n_va:.6f}")
    print(f"  训练集唯一分组数量: {len(set(groups_train))}")
    print(f"  验证集唯一分组数量: {len(set(groups_val))}")
    overlap = set(groups_train) & set(groups_val)
    print(f"  训练集与验证集的分组交集数量: {len(overlap)}")
    # 分组隔离：同一公司不得同时出现在训练集和验证集
    assert_fold_no_group_overlap(groups_train, groups_val, fold_idx)

    if fraud_tr == 0 or fraud_tr == n_tr:
        raise RuntimeError(f"第 {fold_idx} 折训练集未同时包含正类和负类。")
    if fraud_va == 0 or fraud_va == n_va:
        raise RuntimeError(
            f"第 {fold_idx} 折验证集未同时包含正类和负类；"
            "ROC-AUC/PR-AUC 等指标无法计算。"
        )


def run_stratified_group_kfold(
    development_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, Dict[str, Any]]:
    """
    在开发集上执行 StratifiedGroupKFold。

    分层交叉验证：在保持分组完整性的前提下，尽可能维持类别比例。
    分组完整性优先级高于每折类别比例完全一致。
    """
    warn_if_large_groups(development_df, cfg.N_SPLITS)

    texts = development_df[cfg.TEXT_COLUMN].astype(str).values
    labels = development_df[cfg.LABEL_COLUMN].astype(int).values
    groups = development_df["group_id"].astype(str).values
    record_ids = development_df[cfg.RECORD_ID_COLUMN].astype(str).values

    # 分层交叉验证：在保持分组完整性的前提下，尽可能维持类别比例
    sgkf = StratifiedGroupKFold(
        n_splits=cfg.N_SPLITS,
        shuffle=cfg.CV_SHUFFLE,
        random_state=cfg.RANDOM_STATE,
    )

    oof_prob = np.full(len(development_df), np.nan, dtype=np.float64)
    fold_rows: List[Dict[str, Any]] = []
    best_epochs: List[int] = []

    # 注意：严禁在此循环外对完整开发集拟合 TF-IDF
    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(texts, labels, groups), start=1
    ):
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        groups_train = groups[train_idx]
        groups_val = groups[val_idx]
        _print_fold_header(fold_idx, y_train, y_val, groups_train, groups_val)

        texts_train = texts[train_idx]
        texts_val = texts[val_idx]

        # 防止数据泄露：TF-IDF 仅在当前折训练文本上拟合
        x_train, vectorizer, reducer = fit_transform_train_features(texts_train)
        # 验证集仅 transform；验证集保持真实的不平衡分布
        x_val = transform_features(texts_val, vectorizer, reducer)

        # 采样隔离：仅对当前折训练数据执行过采样
        x_train_res, y_train_res, sampler_name = maybe_oversample(x_train, y_train)

        cw_cfg = resolve_class_weight(cfg.SAMPLING_SCHEME)
        if cw_cfg == "balanced_placeholder":
            class_weight = compute_balanced_class_weight(y_train)  # 基于过采样前标签
        elif isinstance(cw_cfg, dict):
            class_weight = cw_cfg
        else:
            class_weight = None

        # 每一折必须重新初始化模型，防止模型权重跨折污染
        clear_tf_session()
        model = build_dnn(input_dim=x_train_res.shape[1])
        callbacks = make_callbacks()

        history = model.fit(
            x_train_res,
            y_train_res,
            validation_data=(x_val, y_val),  # 未采样的原始验证集
            epochs=cfg.DNN_EPOCHS,
            batch_size=cfg.DNN_BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        # 记录最佳 epoch（用于最终模型固定训练轮数）
        monitor = cfg.DNN_MONITOR_METRIC
        if monitor in history.history:
            scores = history.history[monitor]
            if cfg.DNN_MONITOR_MODE == "max":
                best_ep = int(np.argmax(scores)) + 1
            else:
                best_ep = int(np.argmin(scores)) + 1
        else:
            best_ep = len(history.history.get("loss", []))
        best_epochs.append(best_ep)

        # AUC 指标必须基于预测概率计算
        val_prob = predict_proba(model, x_val)
        oof_prob[val_idx] = val_prob

        metrics = evaluate_binary_predictions(
            y_val,
            val_prob,
            threshold=cfg.DEFAULT_THRESHOLD,
            fold_label=f"折{fold_idx}",
        )
        metrics.update(
            {
                "fold": fold_idx,
                "best_epoch": best_ep,
                "sampler": sampler_name or "none",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_train_groups": len(set(groups_train)),
                "n_val_groups": len(set(groups_val)),
                "group_overlap": 0,
            }
        )
        fold_rows.append(metrics)

        print_evaluation_metrics(
            f"折 {fold_idx}/{cfg.N_SPLITS} 验证集指标（阈值={cfg.DEFAULT_THRESHOLD:.2f}）",
            metrics,
            threshold=cfg.DEFAULT_THRESHOLD,
        )

        # 释放当前折对象引用（下一折会 clear_session）
        del model, vectorizer, reducer, x_train, x_val, x_train_res

    if np.isnan(oof_prob).any():
        raise RuntimeError("部分开发集样本未获得 out-of-fold 预测，CV 划分异常。")

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_cv_results(fold_df)

    oof_pred_default = (oof_prob >= cfg.DEFAULT_THRESHOLD).astype(int)
    oof_df = pd.DataFrame(
        {
            "original_index": development_df.index.astype(int),
            "record_id": record_ids,
            "true_label": labels,
            "pred_proba": oof_prob,
            "pred_label": oof_pred_default,
            "group_id": groups,
        }
    )
    # 标注所属 fold
    fold_assign = np.full(len(development_df), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(texts, labels, groups), start=1
    ):
        fold_assign[val_idx] = fold_idx
    oof_df["fold"] = fold_assign

    # 基于全部 OOF 统一选阈值（避免每折不同阈值再平均）
    best_threshold, thr_df = search_threshold_oof(labels, oof_prob)
    oof_df["pred_label_opt"] = (oof_prob >= best_threshold).astype(int)
    oof_df["threshold_used_for_default_col"] = cfg.DEFAULT_THRESHOLD

    # 用最优阈值重算 OOF 汇总参考指标
    oof_metrics = evaluate_binary_predictions(
        labels, oof_prob, threshold=best_threshold, fold_label="OOF"
    )
    print_evaluation_metrics(
        "开发集 OOF 指标（最优阈值）",
        oof_metrics,
        threshold=best_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(output_dir / "cv_fold_results.csv", index=False)
    summary_df.to_csv(output_dir / "cv_summary.csv", index=False)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    thr_df.to_csv(output_dir / "threshold_search_results.csv", index=False)
    save_evaluation_artifacts(
        output_dir,
        prefix="oof",
        metrics=oof_metrics,
        threshold=best_threshold,
        context="oof_best_threshold",
    )

    fold_cm_rows = []
    for _, row in fold_df.iterrows():
        fold_cm_rows.append(
            {
                "fold": int(row["fold"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tp": int(row["tp"]),
                "matrix": (
                    f"[[{int(row['tn'])} {int(row['fp'])}] "
                    f"[{int(row['fn'])} {int(row['tp'])}]]"
                ),
            }
        )
    pd.DataFrame(fold_cm_rows).to_csv(
        output_dir / "cv_fold_confusion_matrix.csv", index=False
    )

    print("\n交叉验证汇总（均值 ± 标准差）:")
    metric_label_map = dict(CORE_METRIC_SPECS)
    for _, row in summary_df.iterrows():
        label = metric_label_map.get(row["metric"], row["metric"])
        print(
            f"  {label}: {row['mean']:.4f} ± {row['std']:.4f} "
            f"(min={row['min']:.4f}, max={row['max']:.4f})"
        )

    cv_meta = {
        "best_threshold": best_threshold,
        "best_epochs": best_epochs,
        "median_best_epoch": int(np.median(best_epochs)),
        "mean_best_epoch": float(np.mean(best_epochs)),
        "oof_metrics_at_best_threshold": oof_metrics,
        "sampling_scheme": cfg.SAMPLING_SCHEME,
    }
    return fold_df, summary_df, oof_df, best_threshold, cv_meta


def _split_group_holdout(
    development_df: pd.DataFrame,
    holdout_fraction: float = cfg.FINAL_HOLDOUT_FRACTION,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从开发集再划分小型分组验证集，仅用于最终训练 Early Stopping。
    训练部分与监控验证部分不存在分组重叠；测试集不参与。
    """
    labels = development_df[cfg.LABEL_COLUMN].astype(int).values
    groups = development_df["group_id"].astype(str).values
    texts = development_df[cfg.TEXT_COLUMN].astype(str).values

    # 用 1/holdout 近似折数，取第一折验证作为 holdout
    n_splits = max(2, int(round(1.0 / holdout_fraction)))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )
    train_idx, val_idx = next(sgkf.split(texts, labels, groups))
    assert_fold_no_group_overlap(groups[train_idx], groups[val_idx], fold_idx=0)
    return train_idx, val_idx


def train_final_model(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
    cv_meta: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    使用完整开发集重新训练最终模型，并对独立测试集仅评估一次。

    测试集不得参与特征拟合、模型选择或阈值调整。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    texts_dev = development_df[cfg.TEXT_COLUMN].astype(str).values
    y_dev = development_df[cfg.LABEL_COLUMN].astype(int).values
    texts_test = test_df[cfg.TEXT_COLUMN].astype(str).values
    y_test = test_df[cfg.LABEL_COLUMN].astype(int).values

    # 最终 TF-IDF / SVD 只能在完整开发集上拟合
    print("\n" + "=" * 72)
    print("最终模型训练（完整开发集）")
    print("=" * 72)

    if cfg.FINAL_ES_STRATEGY == "group_holdout":
        tr_idx, va_idx = _split_group_holdout(development_df)
        x_fit_texts = texts_dev[tr_idx]
        y_fit = y_dev[tr_idx]
        x_mon_texts = texts_dev[va_idx]
        y_mon = y_dev[va_idx]
        # 特征仍只在最终训练子集上拟合，避免 holdout 监控集参与词表/IDF/SVD
        x_fit, vectorizer, reducer = fit_transform_train_features(x_fit_texts)
        x_mon = transform_features(x_mon_texts, vectorizer, reducer)
        fixed_epochs = cfg.DNN_EPOCHS
        print(
            f"最终 Early Stopping 策略=group_holdout: "
            f"fit={len(tr_idx)}, monitor={len(va_idx)}"
        )
    else:
        # fixed_epochs：使用 CV 最佳 epoch 中位数，在完整开发集上固定训练
        x_fit, vectorizer, reducer = fit_transform_train_features(texts_dev)
        y_fit = y_dev
        x_mon, y_mon = None, None
        fixed_epochs = int(cv_meta.get("median_best_epoch", cfg.DNN_EPOCHS))
        fixed_epochs = max(1, min(fixed_epochs, cfg.DNN_EPOCHS))
        print(
            f"最终 Early Stopping 策略=fixed_epochs: "
            f"使用 CV 最佳 epoch 中位数 = {fixed_epochs}"
        )

    # 仅对用于训练的开发子集过采样；测试集保持原始不平衡分布
    x_fit_res, y_fit_res, sampler_name = maybe_oversample(x_fit, y_fit)

    cw_cfg = resolve_class_weight(cfg.SAMPLING_SCHEME)
    if cw_cfg == "balanced_placeholder":
        class_weight = compute_balanced_class_weight(y_fit)
    elif isinstance(cw_cfg, dict):
        class_weight = cw_cfg
    else:
        class_weight = None

    if cfg.FINAL_ES_STRATEGY == "group_holdout":
        # 第一阶段：在分组 holdout 上监控，估计合适的训练轮数（绝不使用测试集）
        clear_tf_session()
        probe_model = build_dnn(input_dim=x_fit_res.shape[1])
        probe_history = probe_model.fit(
            x_fit_res,
            y_fit_res,
            validation_data=(x_mon, y_mon),
            epochs=cfg.DNN_EPOCHS,
            batch_size=cfg.DNN_BATCH_SIZE,
            callbacks=make_callbacks(),
            class_weight=class_weight,
            verbose=1,
        )
        ran_epochs = max(1, len(probe_history.history.get("loss", [1])))
        print(
            f"group_holdout 探针训练结束，使用完整开发集按 {ran_epochs} epochs 重训最终模型..."
        )
        del probe_model

        # 第二阶段：完整 development_df 上重新拟合特征并固定轮数训练
        x_fit, vectorizer, reducer = fit_transform_train_features(texts_dev)
        x_fit_res, y_fit_res, sampler_name = maybe_oversample(x_fit, y_dev)
        if cw_cfg == "balanced_placeholder":
            class_weight = compute_balanced_class_weight(y_dev)
        clear_tf_session()
        model = build_dnn(input_dim=x_fit_res.shape[1])
        model.fit(
            x_fit_res,
            y_fit_res,
            epochs=ran_epochs,
            batch_size=cfg.DNN_BATCH_SIZE,
            class_weight=class_weight,
            verbose=1,
        )
    else:
        clear_tf_session()
        model = build_dnn(input_dim=x_fit_res.shape[1])
        model.fit(
            x_fit_res,
            y_fit_res,
            epochs=fixed_epochs,
            batch_size=cfg.DNN_BATCH_SIZE,
            class_weight=class_weight,
            verbose=1,
        )

    # 测试集仅 transform + 已确定阈值评估一次
    x_test = transform_features(texts_test, vectorizer, reducer)
    test_prob = predict_proba(model, x_test)
    test_metrics = evaluate_binary_predictions(
        y_test, test_prob, threshold=threshold, fold_label="TEST"
    )

    print("\n" + "=" * 72)
    print(
        "这是在模型结构、超参数、训练策略和分类阈值全部确定后，"
        "对独立测试集进行的最终一次评估。"
    )
    print("=" * 72)
    print(f"测试集样本数量: {len(test_df)}")
    print(f"测试集欺诈样本数量: {int((y_test == 1).sum())}")
    print(f"测试集欺诈比例: {(y_test == 1).mean():.6f}")
    print_evaluation_metrics("独立测试集最终指标", test_metrics, threshold=threshold)

    pred_df = pd.DataFrame(
        {
            "original_index": test_df.index.astype(int),
            "record_id": test_df[cfg.RECORD_ID_COLUMN].astype(str).values,
            "true_label": y_test,
            "pred_proba": test_prob,
            "pred_label": (test_prob >= threshold).astype(int),
            "group_id": test_df["group_id"].astype(str).values,
            "threshold": threshold,
        }
    )
    cm_df = pd.DataFrame(
        [
            {"label": "actual_0", "pred_0": int(test_metrics["tn"]), "pred_1": int(test_metrics["fp"])},
            {"label": "actual_1", "pred_0": int(test_metrics["fn"]), "pred_1": int(test_metrics["tp"])},
        ]
    )
    metrics_df = pd.DataFrame([test_metrics])

    pred_df.to_csv(output_dir / "final_test_predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "final_test_metrics.csv", index=False)
    cm_df.to_csv(output_dir / "final_confusion_matrix.csv", index=False)

    model_path = output_dir / "final_model.keras"
    model.save(model_path)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
    if reducer is not None:
        joblib.dump(reducer, output_dir / "dimensionality_reducer.pkl")

    train_config = cfg.ExperimentConfig().to_dict()
    train_config.update(
        {
            "final_threshold": threshold,
            "final_sampler": sampler_name,
            "final_es_strategy": cfg.FINAL_ES_STRATEGY,
            "cv_meta_best_epochs": cv_meta.get("best_epochs"),
            "cv_meta_median_best_epoch": cv_meta.get("median_best_epoch"),
            "cv_data_scope": cv_meta.get("cv_data_scope"),
            "note_tf_determinism": (
                "即使设置随机种子，GPU 环境中的部分 TensorFlow 操作仍可能存在轻微非确定性。"
            ),
        }
    )
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    return {
        "test_metrics": test_metrics,
        "threshold": threshold,
        "model_path": str(model_path),
    }
