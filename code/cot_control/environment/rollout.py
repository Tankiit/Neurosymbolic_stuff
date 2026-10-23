"""Small wrapper around a single reasoning transition."""

from __future__ import annotations

from typing import Any, Dict

from ..reasoning import Action
from .reasoning_state import ReasoningState
from .transitions import apply_action


def rollout_step(state: ReasoningState, action: Action, step_text: str) -> ReasoningState:
    """Apply a single action transition and return the updated reasoning state."""
    return apply_action(state, action, step_text)
