"""One-click entry for VS Code / Cursor "Run Python File".

Open this file and click Run Python File to start the full BERT pipeline
(default: train). Artifacts write to BERT/weight/ and BERT/results/.

Optional CLI (same as bert.py) still works, e.g.:
  python run.py train
  python run.py evaluate
  python run.py predict --text "..."
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import PROJECT_ROOT, RESULTS_DIR, WEIGHTS_DIR, ensure_directories
from training import main


if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Weights dir : {WEIGHTS_DIR}", flush=True)
    print(f"Results dir : {RESULTS_DIR}", flush=True)
    # "Run Python File" passes no subcommand; default to train.
    if len(sys.argv) == 1:
        sys.argv.append("train")
    main()
