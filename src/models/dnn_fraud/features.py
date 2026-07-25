"""
TF-IDF、降维与采样器构建。

关键防泄露原则：
- 防止数据泄露：TF-IDF 仅在当前折训练文本上拟合
- 防止特征泄露：降维或特征选择模型仅在当前折训练集上拟合
- 采样隔离：仅对当前折训练数据执行过采样
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config as cfg


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    构建全新的 TfidfVectorizer（每折必须重新创建）。

    词级 (1,2)-gram：捕获单词与短短语（如 work from home）；
    不设置 stop_words，保留 URL/数字/邮箱等潜在欺诈信号对应的 token。
    """
    return TfidfVectorizer(
        max_features=cfg.TFIDF_MAX_FEATURES,
        ngram_range=cfg.TFIDF_NGRAM_RANGE,
        min_df=cfg.TFIDF_MIN_DF,
        max_df=cfg.TFIDF_MAX_DF,
        sublinear_tf=cfg.TFIDF_SUBLINEAR_TF,
        stop_words=cfg.TFIDF_STOP_WORDS,
        lowercase=cfg.TFIDF_LOWERCASE,
        analyzer=cfg.TFIDF_ANALYZER,
    )


def build_svd(n_features_in: Optional[int] = None) -> TruncatedSVD:
    """
    构建 TruncatedSVD。
    n_components 不得超过当前折训练特征维度 - 1。
    """
    n_components = cfg.SVD_N_COMPONENTS
    if n_features_in is not None:
        max_allowed = max(1, n_features_in - 1)
        if n_components > max_allowed:
            raise ValueError(
                f"SVD 维度 ({n_components}) 高于允许范围 (max={max_allowed})。"
            )
        n_components = min(n_components, max_allowed)
    return TruncatedSVD(
        n_components=n_components,
        random_state=cfg.RANDOM_STATE,
    )


def estimate_dense_bytes(n_rows: int, n_cols: int, dtype_bytes: int = 8) -> int:
    """估算稠密矩阵内存占用（字节）。"""
    return int(n_rows) * int(n_cols) * int(dtype_bytes)


def ensure_dense_safe(
    matrix: Any,
    context: str,
    max_bytes: int = cfg.MAX_DENSE_BYTES,
) -> np.ndarray:
    """
    在必要时将稀疏矩阵转为稠密，但严禁未经检查直接 toarray()。
    若预计内存超过阈值，抛出明确错误并提示限制 max_features / 使用 SVD。
    """
    if not sparse.issparse(matrix):
        arr = np.asarray(matrix, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{context}: 特征矩阵包含 NaN/Inf。")
        return arr

    n_rows, n_cols = matrix.shape
    needed = estimate_dense_bytes(n_rows, n_cols, dtype_bytes=4)  # float32
    if needed > max_bytes:
        raise MemoryError(
            f"{context}: 稀疏矩阵转稠密存在内存风险。"
            f" shape={matrix.shape}, 预计约 {needed / 1024**3:.2f} GiB "
            f"(阈值 {max_bytes / 1024**3:.2f} GiB)。"
            "请降低 TFIDF_MAX_FEATURES，或启用 TruncatedSVD，或改用支持稀疏输入的模型。"
        )
    arr = matrix.astype(np.float32).toarray()
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{context}: 稠密化后特征包含 NaN/Inf。")
    return arr


def fit_transform_train_features(
    texts_train,
    use_svd: bool = cfg.USE_SVD,
) -> Tuple[np.ndarray, TfidfVectorizer, Optional[TruncatedSVD]]:
    """
    仅在当前折（或最终开发集）训练文本上拟合 TF-IDF（及可选 SVD）。

    防止数据泄露：TF-IDF 仅在当前折训练文本上拟合
    防止特征泄露：降维或特征选择模型仅在当前折训练集上拟合
    """
    vectorizer = build_tfidf_vectorizer()
    # 防止数据泄露：TF-IDF 仅在当前折训练文本上拟合
    x_tfidf = vectorizer.fit_transform(texts_train)

    if x_tfidf.shape[1] == 0:
        raise RuntimeError("TF-IDF 未产生有效特征，请检查文本清洗或 min_df/max_df 设置。")

    reducer: Optional[TruncatedSVD] = None
    if use_svd:
        # 防止特征泄露：降维或特征选择模型仅在当前折训练集上拟合
        reducer = build_svd(n_features_in=x_tfidf.shape[1])
        x_dense = reducer.fit_transform(x_tfidf).astype(np.float32)
    else:
        x_dense = ensure_dense_safe(x_tfidf, context="训练特征稠密化")

    if not np.isfinite(x_dense).all():
        raise RuntimeError("训练特征包含 NaN/Inf。")
    return x_dense, vectorizer, reducer


def transform_features(
    texts,
    vectorizer: TfidfVectorizer,
    reducer: Optional[TruncatedSVD],
) -> np.ndarray:
    """仅 transform：验证集/测试集不得参与词表、IDF、SVD 拟合。"""
    x_tfidf = vectorizer.transform(texts)
    if reducer is not None:
        x = reducer.transform(x_tfidf).astype(np.float32)
    else:
        x = ensure_dense_safe(x_tfidf, context="验证/测试特征稠密化")
    if not np.isfinite(x).all():
        raise RuntimeError("转换后特征包含 NaN/Inf。")
    return x


def _safe_k_neighbors(n_minority: int, requested: int) -> int:
    """根据当前折少数类数量动态调整 k_neighbors。"""
    # SMOTE 需要少数类样本数 > k_neighbors
    max_k = max(1, n_minority - 1)
    return max(1, min(requested, max_k))


def build_sampler(
    y_train: np.ndarray,
    method: str = cfg.SAMPLER_METHOD,
    random_state: int = cfg.RANDOM_STATE,
):
    """
    构建采样器；少数类不足时自动降 k 或回退到 RandomOverSampler。

    采样隔离：仅对当前折训练数据执行过采样
    """
    y = np.asarray(y_train).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"当前折训练集缺少某一类（pos={n_pos}, neg={n_neg}），无法采样。"
        )

    method = method.lower()
    if method == "random":
        return RandomOverSampler(random_state=random_state), "random"

    k = _safe_k_neighbors(n_pos, cfg.SMOTE_K_NEIGHBORS)
    if n_pos < 2:
        print(
            f"[警告] 少数类样本仅 {n_pos}，SMOTE 不可用，回退到 RandomOverSampler。"
        )
        return RandomOverSampler(random_state=random_state), "random_fallback"

    if k < cfg.SMOTE_K_NEIGHBORS:
        print(
            f"[警告] 少数类样本={n_pos}，将 k_neighbors 从 "
            f"{cfg.SMOTE_K_NEIGHBORS} 下调为 {k}。"
        )

    try:
        if method == "svmsmote":
            return SVMSMOTE(k_neighbors=k, random_state=random_state), "svmsmote"
        # 默认 SMOTE；SMOTE 需要稠密输入（本流程在 SVD 之后已是稠密）
        return SMOTE(k_neighbors=k, random_state=random_state), "smote"
    except Exception as exc:  # noqa: BLE001
        print(f"[警告] 构建 {method} 失败 ({exc})，回退 RandomOverSampler。")
        return RandomOverSampler(random_state=random_state), "random_fallback"


def maybe_oversample(
    x_train: np.ndarray,
    y_train: np.ndarray,
    scheme: str = cfg.SAMPLING_SCHEME,
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    按方案决定是否过采样。

    方案 A：仅过采样
    方案 B：仅类别权重（此处不过采样）
    方案 C：过采样 + 轻度类别权重（权重在训练时另行传入）

    验证集保持真实的不平衡分布 — 本函数绝不能作用于验证/测试集。
    """
    scheme = scheme.upper()
    if scheme == "B":
        return x_train, y_train, None

    # 采样隔离：仅对当前折训练数据执行过采样
    sampler, name = build_sampler(y_train)
    x_res, y_res = sampler.fit_resample(x_train, y_train)
    print(
        f"  采样 ({name}): {x_train.shape[0]} -> {x_res.shape[0]} "
        f"(fraud {int((y_train == 1).sum())} -> {int((y_res == 1).sum())})"
    )
    return np.asarray(x_res, dtype=np.float32), np.asarray(y_res).astype(int), name


def resolve_class_weight(scheme: str = cfg.SAMPLING_SCHEME):
    """
    返回 Keras class_weight 字典或 None。

    已使用过采样时避免默认叠加很强的类别权重，防止对少数类双重补偿。
    方案 C 使用轻度权重，并作为明确实验配置。
    """
    scheme = scheme.upper()
    if scheme == "A":
        return None
    if scheme == "B":
        # 仅类别权重：用 balanced 思想的固定映射，由调用方也可改为 sklearn 计算
        return "balanced_placeholder"
    if scheme == "C":
        return {0: 1.0, 1: float(cfg.MILD_CLASS_WEIGHT_FRAUD)}
    raise ValueError(f"未知采样方案: {scheme}")


def compute_balanced_class_weight(y: np.ndarray) -> dict:
    """sklearn 风格 balanced class_weight。"""
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    # n_samples / (n_classes * n_c)
    w0 = n / (2.0 * n_neg)
    w1 = n / (2.0 * n_pos)
    return {0: float(w0), 1: float(w1)}
