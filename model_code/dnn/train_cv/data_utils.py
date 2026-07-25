"""train-only CV 版本的数据统计与合并辅助。"""

from __future__ import annotations

from typing import List

import pandas as pd

from . import config as cfg


def merge_final_training_set(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    将已清洗的 train 与 validation 合并，仅用于交叉验证结束后的最终模型重训。
    validation 不参与 StratifiedGroupKFold / OOF / 阈值搜索。
    """
    return pd.concat([train_df, validation_df], ignore_index=True)


def compute_data_statistics_train_cv(
    train_raw: pd.DataFrame,
    validation_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cv_train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    final_training_df: pd.DataFrame,
    prepared_test_df: pd.DataFrame,
) -> str:
    """生成 train-only CV 版本的数据完整性报告。"""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("数据完整性检查报告（train-only StratifiedGroupKFold 版本）")
    lines.append("=" * 72)
    lines.append(f"\n[CV 数据范围] {cfg.CV_DATA_SCOPE}")
    lines.append("  - StratifiedGroupKFold / OOF / 阈值搜索：仅 train.csv")
    lines.append("  - validation.csv：不参与 CV，仅用于最终模型重训")
    lines.append("  - test.csv：完全独立，仅最终评估一次")

    for name, df in [
        ("train.csv", train_raw),
        ("validation.csv", validation_raw),
        ("test.csv", test_raw),
    ]:
        lines.append(f"\n[{name}]")
        lines.append(f"  样本数量: {len(df)}")
        lines.append(f"  字段: {list(df.columns)}")

    def _class_stats(df: pd.DataFrame, title: str) -> None:
        col = cfg.LABEL_COLUMN
        n = len(df)
        n_fraud = int((df[col] == 1).sum())
        n_legit = int((df[col] == 0).sum())
        ratio = n_fraud / n if n else float("nan")
        lines.append(f"\n[{title}]")
        lines.append(f"  样本数量: {n}")
        lines.append(f"  欺诈样本数量: {n_fraud}")
        lines.append(f"  非欺诈样本数量: {n_legit}")
        lines.append(f"  欺诈样本比例: {ratio:.6f}")
        if "group_id" in df.columns:
            lines.append(f"  唯一分组数量: {df['group_id'].nunique()}")
            sizes = df.groupby("group_id").size()
            lines.append(
                "  每组样本数分布: "
                f"min={sizes.min()}, p50={sizes.median():.1f}, "
                f"mean={sizes.mean():.2f}, max={sizes.max()}"
            )

    _class_stats(cv_train_df, "CV 集 cv_train_df（仅 train.csv，已清洗）")
    _class_stats(validation_df, "固定验证集 validation.csv（已清洗，不参与 CV）")
    _class_stats(final_training_df, "最终重训集 final_training_df（train + validation）")
    _class_stats(prepared_test_df, "测试集 test.csv（已清洗）")

    lines.append("\n[字段兼容性]")
    lines.append(
        f"  三文件字段一致/兼容: "
        f"train={list(train_raw.columns)}, "
        f"validation={list(validation_raw.columns)}, "
        f"test={list(test_raw.columns)}"
    )
    return "\n".join(lines) + "\n"


def apply_config_to_shared_modules() -> None:
    """
    将本版本 config 覆盖写入 dnn.lib.config，
    使复用的 train/features/model/metrics 模块读取相同超参数与路径。
    """
    import dnn.lib.config as shared

    for name in dir(cfg):
        if not name.isupper() or name.startswith("_"):
            continue
        setattr(shared, name, getattr(cfg, name))

    shared.OUTPUT_DIR = cfg.OUTPUT_DIR
    shared.RESULTS_DIR = cfg.RESULTS_DIR
    shared.WEIGHTS_DIR = cfg.WEIGHTS_DIR
    shared.CV_DATA_SCOPE = cfg.CV_DATA_SCOPE
    shared.ExperimentConfig = cfg.ExperimentConfig
