"""Transfer analysis helper for controller generalisation tests."""

from __future__ import annotations

from typing import Dict, Iterable, List


def test_transfer(train_scores: Iterable[float], test_scores: Iterable[float]) -> Dict[str, float]:
    train_values = list(train_scores)
    test_values = list(test_scores)
    return {
        "train_mean": sum(train_values) / max(1, len(train_values)),
        "test_mean": sum(test_values) / max(1, len(test_values)),
        "drop": (sum(train_values) / max(1, len(train_values))) - (sum(test_values) / max(1, len(test_values))),
    }
