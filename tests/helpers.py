from __future__ import annotations

from collections.abc import Sequence
from time import sleep


class FakeAdapter:
    def __init__(self, key: str, display_name: str, score: float):
        self.key = key
        self.display_name = display_name
        self.score = score

    def predict_raw(self, text: str) -> float:
        return self.score

    def predict_raw_batch(self, texts: Sequence[str]) -> list[float]:
        return [self.score for _ in texts]


class TextAwareAdapter(FakeAdapter):
    def predict_raw(self, text: str) -> float:
        return self.predict_raw_batch([text])[0]

    def predict_raw_batch(self, texts: Sequence[str]) -> list[float]:
        scores = []
        for text in texts:
            lowered = text.lower()
            score = 0.2
            if "fee" in lowered:
                score += 0.4
            if "urgent" in lowered:
                score += 0.2
            if "interview" in lowered:
                score -= 0.1
            scores.append(min(1.0, max(0.0, score)))
        return scores


class FailingAdapter(FakeAdapter):
    def predict_raw(self, text: str) -> float:
        raise RuntimeError("mock model failure")

    def predict_raw_batch(self, texts: Sequence[str]) -> list[float]:
        raise RuntimeError("mock model failure")


class SlowAdapter(FakeAdapter):
    def predict_raw(self, text: str) -> float:
        sleep(0.05)
        return self.score

    def predict_raw_batch(self, texts: Sequence[str]) -> list[float]:
        sleep(0.05)
        return super().predict_raw_batch(texts)
