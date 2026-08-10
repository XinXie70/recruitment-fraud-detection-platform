"""BERT fraudulent job-ad detection — CLI entrypoint.

One-click in VS Code / Cursor: open and run ``run.py`` (defaults to train).

CLI (from sprint3/):
  python BERT/code/bert.py train
  python BERT/code/bert.py evaluate
  python BERT/code/bert.py predict --text "..."
  python BERT/code/bert.py export-val

Artifacts write to ``BERT/weight/`` and ``BERT/results/``.

Implementation is split across:
  - run.py          VS Code "Run Python File" entry (default: train)
  - config.py       paths, constants, configs, utilities
  - data_metrics.py preprocessing, datasets, metrics, plots
  - training.py     model, train/eval loops, CLI commands
"""

from __future__ import annotations

from training import main

# Re-export common symbols so `import bert` remains convenient.
from config import (  # noqa: F401
    BERT_FINETUNED_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    WEIGHTS_DIR,
    BertFinetuneConfig,
    ensure_directories,
)
from training import (  # noqa: F401
    BertForFraudClassification,
    cmd_evaluate,
    cmd_export_val,
    cmd_predict,
    cmd_train,
    run_training,
)

if __name__ == "__main__":
    main()
