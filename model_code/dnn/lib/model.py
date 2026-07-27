"""
DNN 模型构建与训练辅助。

每一折必须重新初始化模型，防止不同折之间的权重污染。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as cfg


def clear_tf_session() -> None:
    """每一折开始前清理 TensorFlow 会话，避免模型状态和显存残留。"""
    keras.backend.clear_session()


def build_dnn(
    input_dim: int,
    hidden_units: Optional[List[int]] = None,
    dropout: float = cfg.DNN_DROPOUT,
    learning_rate: float = cfg.DNN_LEARNING_RATE,
) -> keras.Model:
    """
    构建二分类 DNN（Sequential）：
    Dense(ReLU) → Dropout → Dense(ReLU) → Dropout → Dense(ReLU) → Dropout → Sigmoid

    每一折必须重新初始化模型，防止模型权重跨折污染。
    """
    if input_dim <= 0:
        raise ValueError(f"非法 input_dim={input_dim}")

    units = list(hidden_units or cfg.DNN_HIDDEN_UNITS)
    if len(units) != 3:
        raise ValueError("规范要求 3 个 Dense 隐藏层，请配置长度为 3 的 DNN_HIDDEN_UNITS。")

    model = keras.Sequential(name="fraud_dnn")
    model.add(layers.Input(shape=(input_dim,)))
    for i, u in enumerate(units):
        model.add(layers.Dense(u, activation="relu", name=f"dense_{i+1}"))
        model.add(layers.Dropout(dropout, name=f"dropout_{i+1}"))
    model.add(layers.Dense(1, activation="sigmoid", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def make_callbacks(
    monitor: str = cfg.DNN_MONITOR_METRIC,
    mode: str = cfg.DNN_MONITOR_MODE,
) -> Tuple[list, keras.callbacks.History]:
    """
    Early Stopping 与 ReduceLROnPlateau。
    只能监控当前折验证集指标；不得使用 test.csv。
    """
    early = keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=cfg.DNN_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    reduce = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=cfg.DNN_REDUCE_LR_FACTOR,
        patience=cfg.DNN_REDUCE_LR_PATIENCE,
        min_lr=cfg.DNN_MIN_LR,
        verbose=1,
    )
    return [early, reduce]


def predict_proba(model: keras.Model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """返回欺诈类概率，形状 (n,)。检查概率范围。"""
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    probs = np.asarray(probs).reshape(-1)
    if not np.isfinite(probs).all():
        raise RuntimeError("模型训练/预测过程中出现 NaN 概率。")
    if probs.min() < -1e-6 or probs.max() > 1.0 + 1e-6:
        raise RuntimeError(
            f"预测概率不在合理范围 [0,1]: min={probs.min()}, max={probs.max()}"
        )
    return np.clip(probs, 0.0, 1.0)


def set_global_seeds(seed: int = cfg.RANDOM_STATE) -> None:
    """统一设置 Python / NumPy / TensorFlow 随机种子。"""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if cfg.ENABLE_TF_DETERMINISM:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[警告] 无法启用 TensorFlow 确定性操作 ({exc})。"
                "即使设置随机种子，GPU 环境中部分操作仍可能轻微非确定。"
            )
