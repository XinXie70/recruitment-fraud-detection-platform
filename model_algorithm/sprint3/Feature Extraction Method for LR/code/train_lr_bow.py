"""Train LR with Bag-of-Words (CountVectorizer) features on paper-aligned splits."""

from __future__ import annotations

from train_all import run_method


def main() -> None:
    run_method("bow")


if __name__ == "__main__":
    main()
