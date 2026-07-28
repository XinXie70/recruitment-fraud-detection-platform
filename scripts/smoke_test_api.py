"""End-to-end smoke test for the FastAPI fraud detection service.

Runs on the E: ML environment and writes results to E:\\ml\\smoke-test-results\\.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path("E:/ml/smoke-test-results")
OUTPUT_FILE = OUTPUT_DIR / "fake-job-api-smoke.json"
HOST = "127.0.0.1"
PORT = 8765
BASE_URL = f"http://{HOST}:{PORT}"

FRAUD_TEXT = (
    "Urgent work-from-home job. No experience needed. "
    "Send your bank details and SSN to apply immediately."
)
LEGIT_TEXT = (
    "Software engineer role at a registered company. "
    "Apply through our careers portal with your resume."
)


def ensure_lr_weights() -> None:
    artifact = PROJECT_ROOT / "model_weights/logistic_regression/logistic_regression_baseline.joblib"
    if artifact.exists():
        print(f"LR weights found: {artifact}")
        return
    print("LR weights missing — training baseline...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src/models/logistic_regression/train_baseline.py")],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def wait_for_server(client: httpx.Client, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = client.get(f"{BASE_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"API did not become ready within {timeout}s")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_lr_weights()

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    results: dict = {"status": "failed", "checks": []}
    try:
        with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
            wait_for_server(client)

            checks = [
                ("health", "GET", "/health", None),
                ("lr_fraud", "POST", "/predict/lr", {"text": FRAUD_TEXT}),
                ("lr_legit", "POST", "/predict/lr", {"text": LEGIT_TEXT}),
                ("bert_fraud", "POST", "/predict/bert", {"text": FRAUD_TEXT}),
                ("ensemble_fraud", "POST", "/predict/ensemble", {"text": FRAUD_TEXT}),
                ("ensemble_risk_fraud", "POST", "/predict/ensemble/risk", {"text": FRAUD_TEXT}),
                ("ensemble_risk_legit", "POST", "/predict/ensemble/risk", {"text": LEGIT_TEXT}),
            ]

            all_ok = True
            for name, method, path, body in checks:
                if method == "GET":
                    resp = client.get(path)
                else:
                    resp = client.post(path, json=body)
                entry = {
                    "name": name,
                    "path": path,
                    "status_code": resp.status_code,
                    "body": resp.json() if resp.status_code == 200 else resp.text,
                }
                results["checks"].append(entry)
                ok = resp.status_code == 200
                all_ok = all_ok and ok
                print(f"[{'PASS' if ok else 'FAIL'}] {name} -> {resp.status_code}")

            risk_fraud = next(c for c in results["checks"] if c["name"] == "ensemble_risk_fraud")
            risk_legit = next(c for c in results["checks"] if c["name"] == "ensemble_risk_legit")
            if all_ok:
                fraud_level = risk_fraud["body"]["risk_level"]
                legit_level = risk_legit["body"]["risk_level"]
                if fraud_level == "Low" and legit_level == "High":
                    all_ok = False
                    print(f"WARN: unexpected risk ordering fraud={fraud_level} legit={legit_level}")
                else:
                    print(f"Risk levels: fraud={fraud_level}, legit={legit_level}")

            results["status"] = "passed" if all_ok else "failed"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSmoke test {results['status'].upper()}")
    print(f"Results saved to {OUTPUT_FILE}")
    return 0 if results["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
