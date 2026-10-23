"""Fixed policy controller using a hand-written action rule."""

from __future__ import annotations

from typing import Dict

from ..reasoning import Action


class FixedController:
    """A simple baseline rule-based policy."""

    def __init__(self, default_action: str = Action.CONTINUE.value):
        self.default_action = default_action

    def choose_action(self, state_text: str) -> str:
        if "error" in state_text.lower() or "revise" in state_text.lower():
            return Action.REVISE.value
        if "branch" in state_text.lower():
            return Action.BRANCH.value
        if "stop" in state_text.lower():
            return Action.STOP.value
        return self.default_action

    def predict_scores(self, state_text: str) -> Dict[str, float]:
        chosen = self.choose_action(state_text)
        scores = {action.value: 0.0 for action in Action}
        scores[chosen] = 1.0
        return scores
