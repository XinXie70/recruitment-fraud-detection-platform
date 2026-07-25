"""
DNN 欺诈招聘文本分类 — 集中配置。

修改字段名、路径或超参数时，优先只改本文件，避免在业务逻辑中硬编码。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 路径配置（相对项目根目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
VALIDATION_CSV = PROJECT_ROOT / "data" / "splits" / "validation.csv"
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dnn_fraud"


# ---------------------------------------------------------------------------
# 字段配置：若实际 CSV 字段名不同，只改这里即可运行
# ---------------------------------------------------------------------------
TEXT_COLUMN = "combined_text"  # 招聘文本字段
LABEL_COLUMN = "label"  # 二分类标签：0=非欺诈，1=欺诈
GROUP_COLUMN = "group_id"  # 公司/重复组字段（已由数据管道预构建）
TITLE_COLUMN: Optional[str] = None  # 可选职位名称字段；本数据已合入 combined_text
RECORD_ID_COLUMN = "record_id"  # 样本追踪 ID，非模型特征


# ---------------------------------------------------------------------------
# 全局随机种子与严格隔离开关
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# 开发集与测试集若存在相同 group_id，严格模式下立即停止
STRICT_TEST_GROUP_ISOLATION = True

# 即使设置随机种子，GPU 上部分 TensorFlow 算子仍可能轻微非确定
ENABLE_TF_DETERMINISM = True


# ---------------------------------------------------------------------------
# 交叉验证
# ---------------------------------------------------------------------------
N_SPLITS = 5
CV_SHUFFLE = True


# ---------------------------------------------------------------------------
# TF-IDF
# 使用词级 (1,2)-gram：既捕获单词语义，也捕获“work from home”“wire transfer”
# 等常见欺诈短语；不删除 URL/邮箱/数字等潜在欺诈信号。
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True
TFIDF_STOP_WORDS: Optional[str] = None  # 不删停用词，避免削弱欺诈信号
TFIDF_LOWERCASE = True
TFIDF_ANALYZER = "word"


# ---------------------------------------------------------------------------
# 降维（TruncatedSVD）
# DNN 需要稠密输入；直接稠密化 8000 维 TF-IDF 内存风险高，
# 故在每折训练集内拟合 SVD，将维度压到可管理规模后再过采样与训练。
# ---------------------------------------------------------------------------
USE_SVD = True
SVD_N_COMPONENTS = 300
# 稠密化前的安全阈值（字节）；超过则拒绝直接 toarray()
MAX_DENSE_BYTES = 2 * 1024**3  # 2 GiB


# ---------------------------------------------------------------------------
# 采样方案
# A: 仅过采样（默认，推荐）
# B: 仅类别权重
# C: 过采样 + 轻度类别权重（需明确理由，避免双重补偿）
# ---------------------------------------------------------------------------
SAMPLING_SCHEME = "A"  # "A" | "B" | "C"
SAMPLER_METHOD = "smote"  # "smote" | "svmsmote" | "random"
SMOTE_K_NEIGHBORS = 5  # 会按当前折少数类数量动态下调
MILD_CLASS_WEIGHT_FRAUD = 1.5  # 仅方案 C 使用的轻度欺诈类权重


# ---------------------------------------------------------------------------
# DNN 结构与训练
# ---------------------------------------------------------------------------
DNN_HIDDEN_UNITS: List[int] = [256, 128, 64]
DNN_DROPOUT = 0.3
DNN_LEARNING_RATE = 1e-3
DNN_BATCH_SIZE = 64
DNN_EPOCHS = 40
# 设为 True 时使用极少 epoch / 折数，仅用于语法与流水线冒烟（正式实验务必 False）
SMOKE_TEST = False
DNN_EARLY_STOPPING_PATIENCE = 6
DNN_REDUCE_LR_PATIENCE = 3
DNN_REDUCE_LR_FACTOR = 0.5
DNN_MIN_LR = 1e-6
# Early Stopping 监控指标：val_pr_auc | val_loss | val_recall
DNN_MONITOR_METRIC = "val_pr_auc"
DNN_MONITOR_MODE = "max"


# ---------------------------------------------------------------------------
# 阈值搜索（仅基于开发集 out-of-fold 预测，严禁使用测试集）
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.5
THRESHOLD_SEARCH_START = 0.05
THRESHOLD_SEARCH_END = 0.95
THRESHOLD_SEARCH_STEP = 0.01
# 优化目标：fraud_f1 | balanced_accuracy | youden_j | recall_at_precision
THRESHOLD_OBJECTIVE = "fraud_f1"
THRESHOLD_MIN_PRECISION: Optional[float] = None  # 例如 0.3 时配合 recall_at_precision


# ---------------------------------------------------------------------------
# 最终模型 Early Stopping 策略
# "fixed_epochs"：使用 CV 各折最佳 epoch 的中位数，在完整开发集上固定训练
# "group_holdout"：从开发集再划出小型分组验证集仅用于监控（不碰测试集）
# ---------------------------------------------------------------------------
FINAL_ES_STRATEGY = "fixed_epochs"
FINAL_HOLDOUT_FRACTION = 0.1  # 仅 group_holdout 策略使用


@dataclass
class ExperimentConfig:
    """可序列化的实验配置快照，写入 training_config.json。"""

    random_state: int = RANDOM_STATE
    n_splits: int = N_SPLITS
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
