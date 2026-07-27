"""
DNN 欺诈招聘文本分类 — train-only CV 版本入口。

数据流：
  train.csv
        ↓
  StratifiedGroupKFold（仅 train 内部）
        ↓
  OOF 选阈值
        ↓
  train + validation → 重训最终模型（权重写入 model_weights/dnn）
        ↓
  test.csv 评估一次（报告写入 model_results/dnn）
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# model_code/ 必须在 sys.path 上，才能 import dnn.*
_MODEL_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(_MODEL_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_CODE_ROOT))

from dnn.lib.data_utils import (
    check_dev_test_group_overlap,
    load_split_csvs,
    prepare_dataframe,
    validate_compatible_schemas,
)
from dnn.lib.model import set_global_seeds
from dnn.lib.train import run_stratified_group_kfold, train_final_model
from dnn.train_cv import config as cfg
from dnn.train_cv.data_utils import (
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

    results_dir = cfg.RESULTS_DIR
    weights_dir = cfg.WEIGHTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DNN + TF-IDF + StratifiedGroupKFold（train-only CV）")
    print("=" * 72)
    print(f"CV 数据范围: {cfg.CV_DATA_SCOPE}（仅 train.csv）")
    print(f"结果目录: {results_dir}")
    print(f"权重目录: {weights_dir}")
    print(f"采样方案: {cfg.SAMPLING_SCHEME} ({cfg.SAMPLER_METHOD})")

    train_raw, val_raw, test_raw = load_split_csvs()
    validate_compatible_schemas(train_raw, val_raw, test_raw)

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
    (results_dir / "data_statistics.txt").write_text(stats_text, encoding="utf-8")

    print("\n[CV 阶段分组检查] train.csv <-> test.csv")
    cv_overlap_df = check_dev_test_group_overlap(
        cv_train_df,
        test_df,
        strict=cfg.STRICT_TEST_GROUP_ISOLATION,
    )
    cv_overlap_df.to_csv(results_dir / "cv_group_overlap_report.csv", index=False)

    print("\n[最终重训分组检查] (train+validation) <-> test.csv")
    final_overlap_df = check_dev_test_group_overlap(
        final_training_df,
        test_df,
        strict=cfg.STRICT_TEST_GROUP_ISOLATION,
    )
    final_overlap_df.to_csv(results_dir / "group_overlap_report.csv", index=False)

    fold_df, summary_df, oof_df, best_threshold, cv_meta = run_stratified_group_kfold(
        cv_train_df, results_dir
    )
    cv_meta["cv_data_scope"] = cfg.CV_DATA_SCOPE

    final_info = train_final_model(
        development_df=final_training_df,
        test_df=test_df,
        threshold=best_threshold,
        cv_meta=cv_meta,
        output_dir=results_dir,
        weights_dir=weights_dir,
    )

    print("\n训练完成。主要产物:")
    for name in [
        "data_statistics.txt",
        "cv_fold_results.csv",
        "cv_summary.csv",
        "oof_predictions.csv",
        "threshold_search_results.csv",
        "final_test_metrics.csv",
        "final_test_predictions.csv",
        "final_confusion_matrix.csv",
    ]:
        path = results_dir / name
        status = "OK" if path.exists() else "缺失"
        print(f"  [results {status}] {path}")
    for name in [
        "final_model.keras",
        "tfidf_vectorizer.pkl",
        "dimensionality_reducer.pkl",
        "training_config.json",
    ]:
        path = weights_dir / name
        status = "OK" if path.exists() else "缺失(可能未启用)"
        print(f"  [weights {status}] {path}")

    tm = final_info["test_metrics"]
    print(f"\n最终测试阈值: {final_info['threshold']:.4f}")
    print(
        f"最终测试 Fraud F1={tm['fraud_f1']:.4f}, "
        f"PR-AUC={tm['pr_auc']:.4f}, Recall={tm['fraud_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
