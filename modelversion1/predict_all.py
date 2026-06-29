"""
Unified invocation of Logistic Regression and DNN.

Both models are binary classifiers; risk_score is mapped to three-tier display
labels via the risk mapping layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.dnn_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as dnn_predict_batch,
    predict_job_posting as dnn_predict,
)
from final_model_pipelines.lr_pipeline.predict import (  # noqa: E402
    predict_batch_job_postings as lr_predict_batch,
    predict_job_posting as lr_predict,
)


def _strip_model_key(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "model"}


def predict_with_all_models(input_text: str) -> dict:
    """Run LR + DNN on a single text and return structured JSON."""
    return {
        "logistic_regression": _strip_model_key(lr_predict(input_text)),
        "dnn": _strip_model_key(dnn_predict(input_text)),
    }


def predict_batch_with_all_models(input_texts: list[str]) -> list[dict]:
    """Batch prediction."""
    lr_results = lr_predict_batch(input_texts)
    dnn_results = dnn_predict_batch(input_texts)
    return [
        {
            "logistic_regression": _strip_model_key(l),
            "dnn": _strip_model_key(d),
        }
        for l, d in zip(lr_results, dnn_results)
    ]


if __name__ == "__main__":
    sample = (
        "URGENT HIRING! Work from home, earn $5000/week. "
        "No experience needed. Send bank details to apply."
    )
    import json

    print(json.dumps(predict_with_all_models(sample), indent=2, ensure_ascii=False))
