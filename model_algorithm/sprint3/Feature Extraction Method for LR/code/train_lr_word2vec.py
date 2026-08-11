#Train LR with Word2Vec document embeddings for EMSCAD data
from __future__ import annotations
from train_all import run_method
def main() -> None:
    run_method("word2vec")
if __name__ == "__main__":
    main()
