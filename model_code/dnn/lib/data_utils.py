"""
数据加载、字段校验、文本清洗、group_id 构建与分组泄露检查。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as cfg


REQUIRED_COLUMNS = [cfg.TEXT_COLUMN, cfg.LABEL_COLUMN, cfg.GROUP_COLUMN]


class DataValidationError(Exception):
    """数据完整性或字段兼容性错误。"""


class GroupLeakageError(Exception):
    """开发集/测试集或折内存在分组重叠时抛出。"""


def load_split_csvs(
    train_path: Path = cfg.TRAIN_CSV,
    validation_path: Path = cfg.VALIDATION_CSV,
    test_path: Path = cfg.TEST_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载三个 CSV；找不到或无法读取时明确报错。"""
    paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    frames: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件: {path}")
        try:
            frames[name] = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"CSV 文件无法读取: {path}") from exc
    return frames["train"], frames["validation"], frames["test"]


def validate_compatible_schemas(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """检查三个文件字段结构兼容，并确认必要字段存在。"""
    schemas = {
        "train": set(train_df.columns),
        "validation": set(validation_df.columns),
        "test": set(test_df.columns),
    }
    # 要求交集覆盖必要字段；允许某个文件多出无关列，但核心列必须一致存在
    for name, cols in schemas.items():
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise DataValidationError(
                f"{name}.csv 缺少必要字段: {missing}；现有字段: {sorted(cols)}"
            )

    # 额外提示：三个文件核心字段集合是否完全一致（含可选追踪列）
    optional = [cfg.RECORD_ID_COLUMN]
    if cfg.TITLE_COLUMN:
        optional.append(cfg.TITLE_COLUMN)
    for name, cols in schemas.items():
        for col in optional:
            if col not in cols:
                print(f"[警告] {name}.csv 缺少可选字段 {col}，将使用行索引替代追踪。")

    train_core = schemas["train"] & set(REQUIRED_COLUMNS + optional)
    val_core = schemas["validation"] & set(REQUIRED_COLUMNS + optional)
    test_core = schemas["test"] & set(REQUIRED_COLUMNS + optional)
    if not (train_core == val_core == test_core):
        raise DataValidationError(
            "三个 CSV 的核心字段集合不一致，无法安全合并。"
            f" train={sorted(train_core)}, validation={sorted(val_core)}, "
            f"test={sorted(test_core)}"
        )


def _normalize_whitespace(text: str) -> str:
    """克制清洗：去首尾空白、合并连续空白；保留 URL/邮箱/金额/电话/标点。"""
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clean_text_series(series: pd.Series) -> pd.Series:
    """
    对文本列做基础清洗。
    目的：统一空白格式，避免无意义噪声；不删除潜在欺诈信号。
    清洗不依赖任何验证集/测试集统计量。
    """
    cleaned = series.fillna("").astype(str).map(_normalize_whitespace)
    return cleaned


def clean_and_validate_labels(series: pd.Series) -> pd.Series:
    """将标签转为 {0,1} 整数；异常值明确报错。欺诈类必须编码为 1。"""
    if series.isna().any():
        n_missing = int(series.isna().sum())
        raise DataValidationError(f"标签字段存在缺失值: {n_missing} 条，不得静默忽略。")

    try:
        labels = pd.to_numeric(series, errors="raise").astype(int)
    except Exception as exc:  # noqa: BLE001
        raise DataValidationError("标签包含无法转换为数值的值。") from exc

    unique = sorted(labels.unique().tolist())
    if set(unique) != {0, 1}:
        raise DataValidationError(
            f"标签不是严格二分类 {{0,1}}，实际取值: {unique}。"
            "欺诈类别必须明确编码为 1，非欺诈为 0。"
        )
    if 1 not in unique:
        raise DataValidationError("标签中不存在欺诈类 1，无法训练欺诈检测模型。")
    return labels


def standardize_company_token(value: object) -> str:
    """公司/分组字段标准化：小写、去首尾空白、合并连续空格。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def build_group_id(
    df: pd.DataFrame,
    group_column: str = cfg.GROUP_COLUMN,
    title_column: Optional[str] = cfg.TITLE_COLUMN,
    index_prefix: str = "missing_company",
) -> pd.Series:
    """
    为开发集与测试集使用完全相同规则构建 group_id。

    本项目数据管道已生成稳定的 group_id（近重复归组），优先直接使用并标准化。
    若分组字段缺失/空白，不得统一填成同一个 "unknown"：
    而是为每条样本构建稳定且独立的备用标识 missing_company_<original_index>，
    保证：
      1) 同一条数据每次运行得到相同分组；
      2) 不同且无关联的缺失公司样本不会全部进入同一组。

    若未来缺少可靠公司字段且提供 TITLE_COLUMN，可退化为
    标准化公司名 + 职位名称（会弱化公司级隔离，仅作备选）。
    """
    raw = df[group_column] if group_column in df.columns else pd.Series([""] * len(df))
    standardized = raw.map(standardize_company_token)

    # 保留原始行索引以保证缺失组标识稳定可复现
    original_index = df.index.astype(str)

    group_ids: List[str] = []
    for i, (token, orig_idx) in enumerate(zip(standardized.tolist(), original_index.tolist())):
        if token:
            group_ids.append(token)
            continue

        # 缺失分组：独立备用标识
        if title_column and title_column in df.columns:
            title = standardize_company_token(df.iloc[i][title_column])
            if title:
                # 备选：公司缺失时用职位构造临时组（弱化公司隔离，已在注释中说明）
                group_ids.append(f"{index_prefix}_title_{title}_{orig_idx}")
                continue
        group_ids.append(f"{index_prefix}_{orig_idx}")

    return pd.Series(group_ids, index=df.index, name="group_id_built")


def prepare_dataframe(
    df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """清洗文本/标签并构建 group_id；空文本按规则删除并警告。"""
    out = df.copy()
    out["_source_split"] = source_name
    out["_original_index"] = out.index.astype(int)

    if cfg.RECORD_ID_COLUMN not in out.columns:
        out[cfg.RECORD_ID_COLUMN] = [
            f"{source_name}_{i}" for i in range(len(out))
        ]

    out[cfg.TEXT_COLUMN] = clean_text_series(out[cfg.TEXT_COLUMN])
    out[cfg.LABEL_COLUMN] = clean_and_validate_labels(out[cfg.LABEL_COLUMN])
    out["group_id"] = build_group_id(out)

    empty_mask = out[cfg.TEXT_COLUMN].str.len() == 0
    n_empty = int(empty_mask.sum())
    if n_empty > 0:
        print(
            f"[警告] {source_name}: 发现 {n_empty} 条空文本，将删除这些样本 "
            "（清洗后文本为空，无法作为有效输入）。"
        )
        out = out.loc[~empty_mask].reset_index(drop=True)

    return out


def merge_development_set(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    将 train.csv 与 validation.csv 合并为统一开发集。
    原 validation 不再作为固定验证集，改由 StratifiedGroupKFold 动态划分。
    """
    train_prep = prepare_dataframe(train_df, "train")
    val_prep = prepare_dataframe(validation_df, "validation")
    development_df = pd.concat([train_prep, val_prep], ignore_index=True)
    return development_df


def compute_data_statistics(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    development_df: pd.DataFrame,
    prepared_test_df: pd.DataFrame,
) -> str:
    """生成数据完整性检查报告文本。"""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("数据完整性检查报告")
    lines.append("=" * 72)

    for name, df in [
        ("train.csv", train_df),
        ("validation.csv", validation_df),
        ("test.csv", test_df),
    ]:
        lines.append(f"\n[{name}]")
        lines.append(f"  样本数量: {len(df)}")
        lines.append(f"  字段: {list(df.columns)}")

    def _class_stats(df: pd.DataFrame, title: str) -> None:
        n = len(df)
        n_fraud = int((df[cfg.LABEL_COLUMN] == 1).sum())
        n_legit = int((df[cfg.LABEL_COLUMN] == 0).sum())
        ratio = n_fraud / n if n else float("nan")
        lines.append(f"\n[{title}]")
        lines.append(f"  样本数量: {n}")
        lines.append(f"  欺诈样本数量: {n_fraud}")
        lines.append(f"  非欺诈样本数量: {n_legit}")
        lines.append(f"  欺诈样本比例: {ratio:.6f}")
        lines.append(
            f"  文本缺失值数量: {int(df[cfg.TEXT_COLUMN].isna().sum())}"
            if cfg.TEXT_COLUMN in df.columns
            else "  文本字段缺失（未准备）"
        )
        if cfg.TEXT_COLUMN in df.columns:
            empty_text = int((df[cfg.TEXT_COLUMN].fillna("").astype(str).str.strip() == "").sum())
            lines.append(f"  文本为空字符串的样本数量: {empty_text}")
        if cfg.LABEL_COLUMN in df.columns:
            lines.append(f"  标签缺失值数量: {int(df[cfg.LABEL_COLUMN].isna().sum())}")
            uniq = sorted(pd.Series(df[cfg.LABEL_COLUMN]).dropna().unique().tolist())
            lines.append(f"  标签唯一值: {uniq}")
            lines.append(f"  标签是否为二分类: {set(uniq) <= {0, 1} and len(uniq) == 2}")
            lines.append(f"  标签是否只包含 0 和 1: {set(uniq) == {0, 1}}")
            lines.append(f"  欺诈类别是否明确编码为 1: {1 in set(uniq)}")
        if "group_id" in df.columns:
            lines.append(f"  分组字段缺失值数量: {int(df['group_id'].isna().sum())}")
            lines.append(f"  唯一分组数量: {df['group_id'].nunique()}")
            sizes = df.groupby("group_id").size()
            lines.append(
                "  每组样本数分布: "
                f"min={sizes.min()}, p50={sizes.median():.1f}, "
                f"mean={sizes.mean():.2f}, max={sizes.max()}"
            )

    _class_stats(development_df, "开发集 development_df (train+validation)")
    _class_stats(prepared_test_df, "测试集 test.csv（已清洗）")

    lines.append("\n[字段兼容性]")
    lines.append(
        f"  三文件字段一致/兼容: "
        f"train={list(train_df.columns)}, "
        f"validation={list(validation_df.columns)}, "
        f"test={list(test_df.columns)}"
    )
    lines.append(
        "\n[group_id 规则说明]\n"
        "  优先使用数据管道预生成的 group_id（近重复广告归为同组），\n"
        "  标准化为小写并压缩空白；缺失/空白时为每条样本分配独立的\n"
        "  missing_company_<original_index>，避免所有缺失公司被当成同一组。\n"
        "  目标：评估模型对未见过公司/未见过重复簇的泛化能力。"
    )
    return "\n".join(lines) + "\n"


def check_dev_test_group_overlap(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strict: bool = cfg.STRICT_TEST_GROUP_ISOLATION,
) -> pd.DataFrame:
    """
    检查开发集与测试集是否存在相同 group_id。
    StratifiedGroupKFold 只能保证开发集内部折间无分组重叠，
    不能自动修复开发集与既有 test.csv 之间的分组重叠。
    """
    dev_groups = set(development_df["group_id"].astype(str))
    test_groups = set(test_df["group_id"].astype(str))
    overlap = sorted(dev_groups & test_groups)

    print("\n" + "=" * 72)
    print("开发集 <-> 测试集 分组泄露检查")
    print("=" * 72)
    print(f"开发集唯一分组数量: {len(dev_groups)}")
    print(f"测试集唯一分组数量: {len(test_groups)}")
    print(f"开发集与测试集重叠分组数量: {len(overlap)}")

    rows = []
    if overlap:
        print(f"部分重叠分组示例: {overlap[:20]}")
        for gid in overlap:
            n_dev = int((development_df["group_id"] == gid).sum())
            n_test = int((test_df["group_id"] == gid).sum())
            rows.append(
                {
                    "group_id": gid,
                    "n_dev_samples": n_dev,
                    "n_test_samples": n_test,
                }
            )
            print(
                f"  重叠组 {gid}: 开发集样本={n_dev}, 测试集样本={n_test}"
            )
        message = (
            f"检测到开发集与测试集存在 {len(overlap)} 个相同 group_id，"
            "存在潜在公司/分组泄露。\n"
            "StratifiedGroupKFold 只能保证开发集内部每一折的训练集和验证集"
            "不存在分组重叠，不能自动修复开发集与已有 test.csv 之间的分组重叠。\n"
            "若 STRICT_TEST_GROUP_ISOLATION=True，不应继续将该 test.csv "
            "宣称为完全独立测试集。\n"
            "建议：从原始完整数据重新进行按 group_id 的组级切分，"
            "确保 train/validation/test 的分组交集为空后再训练。"
        )
        if strict:
            raise GroupLeakageError(message)
        print("[警告] " + message)
    else:
        print("重叠分组数量为 0：开发集与测试集分组隔离通过。")

    return pd.DataFrame(rows)


def assert_fold_no_group_overlap(
    train_groups: np.ndarray,
    val_groups: np.ndarray,
    fold_idx: int,
) -> None:
    """每一折必须执行分组泄露断言。"""
    overlap = set(train_groups) & set(val_groups)
    if overlap:
        raise GroupLeakageError(
            f"第 {fold_idx} 折训练集与验证集存在分组重叠 "
            f"({len(overlap)} 个)，例如: {list(overlap)[:10]}。"
            "分组隔离：同一公司不得同时出现在训练集和验证集。"
        )


def warn_if_large_groups(
    development_df: pd.DataFrame,
    n_splits: int = cfg.N_SPLITS,
) -> None:
    """分组数过少或单组过大时给出明确警告。"""
    n_groups = development_df["group_id"].nunique()
    if n_groups < n_splits:
        raise DataValidationError(
            f"分组数量 ({n_groups}) 少于交叉验证折数 ({n_splits})，无法进行 StratifiedGroupKFold。"
        )
    sizes = development_df.groupby("group_id").size()
    max_size = int(sizes.max())
    n_dev = len(development_df)
    if max_size > 0.3 * n_dev:
        print(
            f"[警告] 存在过大分组: max_group_size={max_size} "
            f"({max_size / n_dev:.1%} of development)。可能导致折分布严重失衡。"
        )
