"""Risk band assignment for ensemble fraud scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_CONFIG_PATH = PROJECT_ROOT / "reports" / "models" / "risk_band_v1_config.json"


def load_risk_config(path: Path | None = None) -> dict:
    config_path = path or RISK_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Risk config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def assign_risk_level(
    fraud_score: float,
    low_threshold: float,
    high_threshold: float,
) -> Literal["Low", "Suspicious", "High"]:
    if fraud_score < low_threshold:
        return "Low"
    if fraud_score >= high_threshold:
        return "High"
    return "Suspicious"


def build_risk_result(fraud_score: float, config: dict | None = None) -> dict:
    cfg = config or load_risk_config()
    low = float(cfg["low_suspicious_threshold"])
    high = float(cfg["suspicious_high_threshold"])
    return {
        "risk_score": round(fraud_score * 100, 2),
        "risk_level": assign_risk_level(fraud_score, low, high),
        "binary_threshold": high,
        "low_suspicious_threshold": low,
        "suspicious_high_threshold": high,
    }
