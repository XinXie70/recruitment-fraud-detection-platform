#Train LR with character-level TF-IDF features for EMSCAD data
from __future__ import annotations
from train_all import run_method
def main() -> None:
    run_method("char_tfidf")
if __name__ == "__main__":
    main()
