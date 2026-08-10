#Run BERT for EMSCAD data
#python BERT/code/run.py train                     Train the BERT (class-weight) model using the training set.
#python BERT/code/run.py evaluate                  Load the trained weights to evaluate on the test set.
#python BERT/code/run.py predict --text "..."      Model inference on a single text sequence
#python BERT/code/run.py export-val                Export the validation set scores.
from __future__ import annotations
import sys
from pathlib import Path
from config import PROJECT_ROOT, RESULTS_DIR, WEIGHTS_DIR, ensure_directories
from training import main
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
def prepare_run() -> None:
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Weights dir : {WEIGHTS_DIR}", flush=True)
    print(f"Results dir : {RESULTS_DIR}", flush=True)
    if len(sys.argv) == 1:
        sys.argv.append("train")
def run() -> None:
    prepare_run(); main()
if __name__ == "__main__":
    run()