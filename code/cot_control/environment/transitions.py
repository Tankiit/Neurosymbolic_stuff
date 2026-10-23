"""Action semantics for Continue / Revise / Branch / Stop."""

from __future__ import annotations

from typing import Dict

from ..reasoning import Action
from .reasoning_state import ReasoningState


def apply_action(state: ReasoningState, action: Action, step_text: str) -> ReasoningState:
    """Apply an action to the current reasoning state and return the new state."""
    new_state = ReasoningState(
        problem=state.problem,
        subgoals=list(state.subgoals),
        history=list(state.history),
        active_branch=state.active_branch,
        remaining_budget=state.remaining_budget - 1,
        completed=state.completed,
        final_answer=state.final_answer,
    )

    if action == Action.CONTINUE:
        new_state.history.append(step_text)
    elif action == Action.REVISE:
        if new_state.history:
            new_state.history[-1] = step_text
    elif action == Action.BRANCH:
        new_state.history.append(f"[branch:{new_state.active_branch + 1}] {step_text}")
        new_state.active_branch += 1
    elif action == Action.STOP:
        new_state.completed = True
        new_state.final_answer = step_text
    else:
        raise ValueError(f"Unsupported action: {action}")

    return new_state
