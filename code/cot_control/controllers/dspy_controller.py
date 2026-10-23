"""DSPy-style policy interface for action selection."""

from __future__ import annotations

from typing import Dict

from ..reasoning import Action


class DSPyController:
    """A minimal controller stub representing a text-based action selector."""

    def __init__(self, instruction: str | None = None):
        self.instruction = instruction or (
            "Choose the next reasoning action from CONTINUE, REVISE, BRANCH, or STOP. "
            "Prefer REVISE for local mistakes, BRANCH for ambiguity, and STOP for supported answers."
        )

    def choose_action(self, state_text: str) -> str:
        text = state_text.lower()
        if "branch" in text and "error" not in text:
            return Action.BRANCH.value
        if "error" in text or "mistake" in text:
            return Action.REVISE.value
        if "answer" in text and "ready" in text:
            return Action.STOP.value
        return Action.CONTINUE.value

    def predict_scores(self, state_text: str) -> Dict[str, float]:
        chosen = self.choose_action(state_text)
        scores = {action.value: 0.2 for action in Action}
        scores[chosen] = 0.9
        return scores
