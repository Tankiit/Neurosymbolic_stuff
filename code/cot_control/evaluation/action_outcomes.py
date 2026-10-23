"""Summaries of action-specific outcomes."""

from __future__ import annotations

from typing import Dict, Iterable, List


def summarize_action_outcomes(rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    """Compute the average reward for each action in a row collection."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for row in rows:
        action = str(row.get("action", "unknown"))
        reward = float(row.get("reward", 0.0))
        totals[action] = totals.get(action, 0.0) + reward
        counts[action] = counts.get(action, 0) + 1

    return {action: totals[action] / counts[action] for action in counts}
