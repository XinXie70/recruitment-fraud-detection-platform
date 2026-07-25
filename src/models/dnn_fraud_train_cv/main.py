"""
DNN 欺诈招聘文本分类 — train-only CV 版本入口。

数据流（与 merged 版本的区别标注 *）：
  train.csv *
        ↓
  StratifiedGroupKFold（仅 train 内部）
        ↓
  OOF 选阈值 / 确认训练配置
        ↓
  train.csv + validation.csv → final_training_df 重训最终模型
        ↓
  仅 test.csv 评估一次
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from models.dnn_fraud.data_utils import (
    check_dev_test_group_overlap,
    load_split_csvs,
    prepare_dataframe,
    validate_compatible_schemas,
)
from models.dnn_fraud.model import set_global_seeds
from models.dnn_fraud.train import run_stratified_group_kfold, train_final_model

from models.dnn_fraud_train_cv import config as cfg
from models.dnn_fraud_train_cv.data_utils import (
    apply_config_to_shared_modules,
    compute_data_statistics_train_cv,
    merge_final_training_set,
)


def _apply_smoke_overrides() -> None:
    if not cfg.SMOKE_TEST:
        return
    print("[SMOKE_TEST] 启用冒烟配置：N_SPLITS=2, DNN_EPOCHS=2")
    cfg.N_SPLITS = 2
    cfg.DNN_EPOCHS = 2
    cfg.DNN_EARLY_STOPPING_PATIENCE = 1
    cfg.DNN_REDUCE_LR_PATIENCE = 1
    cfg.TFIDF_MAX_FEATURES = 2000
    cfg.SVD_N_COMPONENTS = 50
    apply_config_to_shared_modules()


def main() -> None:
    _apply_smoke_overrides()
    apply_config_to_shared_modules()
    set_global_seeds(cfg.RANDOM_STATE)

    output_dir = cfg.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DNN + TF-IDF + StratifiedGroupKFold（train-only CV 版本）")
    print("=" * 72)
    print(f"CV 数据范围: {cfg.CV_DATA_SCOPE}（仅 train.csv）")
    print(f"输出目录: {output_dir}")
    print(f"采样方案: {cfg.SAMPLING_SCHEME} ({cfg.SAMPLER_METHOD})")

    train_raw, val_raw, test_raw = load_split_csvs()
    validate_compatible_schemas(train_raw, val_raw, test_raw)

    # * 交叉验证仅在 train.csv 内进行；validation 保持独立
    cv_train_df = prepare_dataframe(train_raw, "train")
    validation_df = prepare_dataframe(val_raw, "validation")
    test_df = prepare_dataframe(test_raw, "test")
    final_training_df = merge_final_training_set(cv_train_df, validation_df)

    stats_text = compute_data_statistics_train_cv(
        train_raw,
        val_raw,
        test_raw,
        cv_train_df,
        validation_df,
        final_training_df,
        test_df,
    )
    print(stats_text)
    (output_dir / "data_statistics.txt").write_text(stats_text, encoding="utf-8")

    # CV 阶段：检查 train 与 test 的分组隔离
    print("\n[CV 阶段分组检查] train.csv <-> test.csv")
    cv_overlap_df = check_dev_test_group_overlap(
        cv_train_df,
        test_df,
        strict=cfg.STRICT_TEST_GROUP_ISOLATION,
    )
    cv_overlap_df.to_csv(output_dir / "cv_group_overlap_report.csv", index=False)

    # 最终重训前：检查 train+validation 与 test 的分组隔离
    print("\n[最终重训分组检查] (train+validation) <-> test.csv")
    final_overlap_df = check_dev_test_group_overlap(
        final_training_df,
        test_df,
        strict=cfg.STRICT_TEST_GROUP_ISOLATION,
    )
    final_overlap_df.to_csv(output_dir / "group_overlap_report.csv", index=False)

    # * StratifiedGroupKFold 仅在 cv_train_df 上执行
    fold_df, summary_df, oof_df, best_threshold, cv_meta = run_stratified_group_kfold(
        cv_train_df, output_dir
    )
    cv_meta["cv_data_scope"] = cfg.CV_DATA_SCOPE

    # 最终模型仍在 train + validation 上重训；test 仅 transform + 评估
    final_info = train_final_model(
        development_df=final_training_df,
        test_df=test_df,
        threshold=best_threshold,
        cv_meta=cv_meta,
        output_dir=output_dir,
    )

    print("\n训练完成（train-only CV 版本）。主要产物:")
    for name in [
        "data_statistics.txt",
        "cv_group_overlap_report.csv",
        "group_overlap_report.csv",
        "cv_fold_results.csv",
        "cv_summary.csv",
        "oof_predictions.csv",
        "oof_metrics.csv",
        "oof_confusion_matrix.csv",
        "cv_fold_confusion_matrix.csv",
        "threshold_search_results.csv",
        "final_test_metrics.csv",
        "final_test_predictions.csv",
        "final_confusion_matrix.csv",
        "final_model.keras",
        "tfidf_vectorizer.pkl",
        "dimensionality_reducer.pkl",
        "training_config.json",
    ]:
        path = output_dir / name
        status = "OK" if path.exists() else "缺失(可能未启用)"
        print(f"  [{status}] {path}")

    tm = final_info["test_metrics"]
    print(f"\n最终测试阈值: {final_info['threshold']:.4f}")
    print(
        f"最终测试 Fraud F1={tm['fraud_f1']:.4f}, "
        f"PR-AUC={tm['pr_auc']:.4f}, Recall={tm['fraud_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
