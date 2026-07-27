"""
DNN 欺诈招聘分类 — train-only CV 版本配置。

StratifiedGroupKFold 仅在 train.csv 内执行；validation 不参与 CV / OOF / 阈值搜索。
最终模型在 train + validation 上重训。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from dnn.lib.config import *  # noqa: F401,F403

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = PROJECT_ROOT / "model_weights" / "dnn"
RESULTS_DIR = PROJECT_ROOT / "model_results" / "dnn"
OUTPUT_DIR = RESULTS_DIR

CV_DATA_SCOPE = "train_only"


@dataclass
class ExperimentConfig:
    """可序列化的实验配置快照。"""

    random_state: int = RANDOM_STATE
    n_splits: int = N_SPLITS
    cv_data_scope: str = CV_DATA_SCOPE
    sampling_scheme: str = SAMPLING_SCHEME
    sampler_method: str = SAMPLER_METHOD
    use_svd: bool = USE_SVD
    svd_n_components: int = SVD_N_COMPONENTS
    tfidf_max_features: int = TFIDF_MAX_FEATURES
    tfidf_ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE
    dnn_hidden_units: List[int] = field(default_factory=lambda: list(DNN_HIDDEN_UNITS))
    dnn_dropout: float = DNN_DROPOUT
    dnn_learning_rate: float = DNN_LEARNING_RATE
    dnn_batch_size: int = DNN_BATCH_SIZE
    dnn_epochs: int = DNN_EPOCHS
    default_threshold: float = DEFAULT_THRESHOLD
    threshold_objective: str = THRESHOLD_OBJECTIVE
    final_es_strategy: str = FINAL_ES_STRATEGY
    strict_test_group_isolation: bool = STRICT_TEST_GROUP_ISOLATION
    text_column: str = TEXT_COLUMN
    label_column: str = LABEL_COLUMN
    group_column: str = GROUP_COLUMN
    title_column: Optional[str] = TITLE_COLUMN
    record_id_column: str = RECORD_ID_COLUMN

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)
