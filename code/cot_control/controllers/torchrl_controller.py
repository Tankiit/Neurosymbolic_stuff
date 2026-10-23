"""TorchRL-style policy placeholder for sequential action selection."""

from __future__ import annotations

from typing import Dict

from ..reasoning import Action


class TorchRLController:
    """A minimal neural-policy placeholder that mirrors the TorchRL design goal."""

    def __init__(self, action_space: list[str] | None = None):
        self.action_space = action_space or [
            Action.CONTINUE.value,
            Action.REVISE.value,
            Action.BRANCH.value,
            Action.STOP.value,
        ]

    def choose_action(self, state_text: str) -> str:
        text = state_text.lower()
        if "branch" in text:
            return Action.BRANCH.value
        if "revise" in text or "error" in text:
            return Action.REVISE.value
        if "stop" in text or "answer" in text:
            return Action.STOP.value
        return Action.CONTINUE.value

    def predict_scores(self, state_text: str) -> Dict[str, float]:
        chosen = self.choose_action(state_text)
        scores = {action: 0.1 for action in self.action_space}
        scores[chosen] = 0.9
        return scores
