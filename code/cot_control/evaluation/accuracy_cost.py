"""Accuracy-cost bookkeeping for matched-budget evaluation."""

from __future__ import annotations

from typing import Dict, Iterable


def summarize_accuracy_cost(rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    total_reward = 0.0
    total_tokens = 0
    total_examples = 0

    for row in rows:
        total_reward += float(row.get("reward", 0.0))
        total_tokens += int(row.get("token_cost", 0))
        total_examples += 1

    return {
        "avg_reward": total_reward / max(1, total_examples),
        "avg_tokens": total_tokens / max(1, total_examples),
        "total_examples": float(total_examples),
    }
