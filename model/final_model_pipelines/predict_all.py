"""
Unified invocation of all binary classifiers.

Each model outputs risk_score mapped to three-tier display labels via the risk mapping layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.bert_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as bert_predict_batch,
    predict_job_posting as bert_predict,
)
from final_model_pipelines.bilstm_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as bilstm_predict_batch,
    predict_job_posting as bilstm_predict,
)
from final_model_pipelines.dnn_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as dnn_predict_batch,
    predict_job_posting as dnn_predict,
)
from final_model_pipelines.lr_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as lr_predict_batch,
    predict_job_posting as lr_predict,
)
from final_model_pipelines.rnn_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as rnn_predict_batch,
    predict_job_posting as rnn_predict,
)
from final_model_pipelines.roberta_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as roberta_predict_batch,
    predict_job_posting as roberta_predict,
)
from final_model_pipelines.svm_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as svm_predict_batch,
    predict_job_posting as svm_predict,
)
from final_model_pipelines.xgboost_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as xgboost_predict_batch,
    predict_job_posting as xgboost_predict,
)


def _strip_model_key(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "model"}


def predict_with_all_models(input_text: str) -> dict:
    """Run all models on a single text and return structured JSON."""
    return {
        "logistic_regression": _strip_model_key(lr_predict(input_text)),
        "svm": _strip_model_key(svm_predict(input_text)),
        "xgboost": _strip_model_key(xgboost_predict(input_text)),
        "dnn": _strip_model_key(dnn_predict(input_text)),
        "rnn": _strip_model_key(rnn_predict(input_text)),
        "bilstm": _strip_model_key(bilstm_predict(input_text)),
        "bert": _strip_model_key(bert_predict(input_text)),
        "roberta": _strip_model_key(roberta_predict(input_text)),
    }


def predict_batch_with_all_models(input_texts: list[str]) -> list[dict]:
    """Batch prediction across all models."""
    lr_results = lr_predict_batch(input_texts)
    svm_results = svm_predict_batch(input_texts)
    xgb_results = xgboost_predict_batch(input_texts)
    dnn_results = dnn_predict_batch(input_texts)
    rnn_results = rnn_predict_batch(input_texts)
    bilstm_results = bilstm_predict_batch(input_texts)
    bert_results = bert_predict_batch(input_texts)
    roberta_results = roberta_predict_batch(input_texts)
    return [
        {
            "logistic_regression": _strip_model_key(lr),
            "svm": _strip_model_key(svm),
            "xgboost": _strip_model_key(xgb),
            "dnn": _strip_model_key(dnn),
            "rnn": _strip_model_key(rnn),
            "bilstm": _strip_model_key(bilstm),
            "bert": _strip_model_key(bert),
            "roberta": _strip_model_key(roberta),
        }
        for lr, svm, xgb, dnn, rnn, bilstm, bert, roberta in zip(
            lr_results,
            svm_results,
            xgb_results,
            dnn_results,
            rnn_results,
            bilstm_results,
            bert_results,
            roberta_results,
            strict=True,
        )
    ]


if __name__ == "__main__":
    sample = (
        "URGENT HIRING! Work from home, earn $5000/week. "
        "No experience needed. Send bank details to apply."
    )
    import json

    print(json.dumps(predict_with_all_models(sample), indent=2, ensure_ascii=False))
