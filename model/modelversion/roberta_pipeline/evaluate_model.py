"""RoBERTa evaluation: dual-threshold tuning, three-tier risk stats, visualization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.evaluation_utils import run_full_evaluation  # noqa: E402
from final_model_pipelines.roberta_pipeline.data_preprocessing import load_or_create_splits  # noqa: E402
from final_model_pipelines.roberta_pipeline.model_config import (  # noqa: E402
    EVAL_BATCH_SIZE,
    LABEL_COL,
    MODEL_DISPLAY_NAME,
    OUTPUT_DIR,
    SAMPLE_TEXTS,
    SAVED_MODEL_DIR,
    TEXT_COL,
)
from final_model_pipelines.roberta_pipeline.predict import predict_job_posting  # noqa: E402
from final_model_pipelines.transformer_common import (  # noqa: E402
    load_transformer_artifacts,
    predict_proba,
    resolve_device,
)


def evaluate() -> dict:
    model, tokenizer, meta = load_transformer_artifacts(SAVED_MODEL_DIR)
    max_len = int(meta.get("max_len", 256))
    device = resolve_device()
    _, val_df, test_df = load_or_create_splits()

    val_prob = predict_proba(
        model,
        tokenizer,
        val_df[TEXT_COL],
        max_len=max_len,
        batch_size=EVAL_BATCH_SIZE,
        device=device,
    )
    test_prob = predict_proba(
        model,
        tokenizer,
        test_df[TEXT_COL],
        max_len=max_len,
        batch_size=EVAL_BATCH_SIZE,
        device=device,
    )

    summary = run_full_evaluation(
        model_display_name=MODEL_DISPLAY_NAME,
        y_val=val_df[LABEL_COL].values,
        prob_val=val_prob,
        y_test=test_df[LABEL_COL].values,
        prob_test=test_prob,
        sample_texts=SAMPLE_TEXTS,
        predict_fn=predict_job_posting,
        output_dir=OUTPUT_DIR,
        saved_model_dir=SAVED_MODEL_DIR,
    )

    print("\n=== RoBERTa Binary Metrics ===")
    print(pd.DataFrame(summary["binary_metrics"]).to_string(index=False))
    print("\n=== Thresholds ===")
    print(json.dumps(summary["risk_mapping_thresholds"], indent=2))
    print("\n=== Tier Statistics (Test) ===")
    test_tiers = next(t for t in summary["tier_statistics_supplementary"] if t["split"] == "Test")
    print(json.dumps(test_tiers, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    evaluate()
