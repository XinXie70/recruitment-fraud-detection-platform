from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
MODEL_DIR = PROJECT_ROOT / "model"
for path in (PROJECT_ROOT, BACKEND_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/fake_job_detection_tests.db")
os.environ.setdefault("OLLAMA_ENABLED", "false")
os.environ.setdefault("XAI_USE_SHAP", "false")
