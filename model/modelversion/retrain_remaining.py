"""Train and evaluate remaining pipelines starting from RNN."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINES_ROOT = Path(__file__).resolve().parent
LOG_DIR = PIPELINES_ROOT / "training_logs"

MODELS = [
    "rnn_pipeline",
    "bilstm_pipeline",
    "bert_pipeline",
    "roberta_pipeline",
]


def run_step(name: str, script: Path, env: dict[str, str]) -> dict:
    started = time.time()
    print(f"\n{'=' * 60}\nRunning {name}: {script}\n{'=' * 60}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    elapsed = time.time() - started
    status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"
    print(f"[{status}] {name} finished in {elapsed / 60:.1f} min", flush=True)
    return {"step": name, "script": str(script), "status": status, "seconds": round(elapsed, 1)}


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("CUDA_PATH", r"E:\ml\cuda\v12.8")
    env.setdefault("HF_HOME", r"E:\ml\cache\huggingface")
    env.setdefault("TRANSFORMERS_CACHE", r"E:\ml\cache\huggingface\transformers")
    env.setdefault("TORCH_HOME", r"E:\ml\cache\torch")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    results: list[dict] = []
    overall_start = time.time()

    for pipeline in MODELS:
        train_script = PIPELINES_ROOT / pipeline / "train_model.py"
        eval_script = PIPELINES_ROOT / pipeline / "evaluate_model.py"
        results.append(run_step(f"{pipeline}/train", train_script, env))
        if results[-1]["status"] != "ok":
            continue
        results.append(run_step(f"{pipeline}/evaluate", eval_script, env))

    results.append(run_step("compare_all_models", PIPELINES_ROOT / "compare_all_models.py", env))

    summary = {
        "python": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "total_minutes": round((time.time() - overall_start) / 60, 1),
        "steps": results,
    }
    out = LOG_DIR / "retrain_remaining_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    failed = [r for r in results if r["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
