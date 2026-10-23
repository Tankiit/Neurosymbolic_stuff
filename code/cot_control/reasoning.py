from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Action(str, Enum):
    CONTINUE = "continue"
    REVISE = "revise"
    BRANCH = "branch"
    STOP = "stop"


@dataclass
class Step:
    index: int
    text: str


@dataclass
class ReasoningState:
    problem: str
    history: List[str] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    branch_id: int = 0
    active: bool = True

    @property
    def state_text(self) -> str:
        parts = [self.problem]
        if self.history:
            parts.extend(self.history)
        if self.steps:
            parts.extend(step.text for step in self.steps)
        return "\n".join(parts)


@dataclass
class RolloutResult:
    problem: str
    state: str
    action: Action
    reward: float
    token_cost: int
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def decompose_problem(problem: str) -> List[str]:
    sentences = [part.strip() for part in problem.split(".") if part.strip()]
    if not sentences:
        return [problem]
    return sentences


def next_step_for_action(state: ReasoningState, action: Action) -> str:
    if action == Action.CONTINUE:
        return f"Continue from step {max(1, len(state.steps) + 1)}: reason toward the answer."
    if action == Action.REVISE:
        return "Revise the last step by checking the previous inference and correcting the error."
    if action == Action.BRANCH:
        return "Branch: explore a distinct interpretation or alternative reasoning path."
    return "Stop: answer is ready and no further reasoning is needed."


def simulate_rollout(problem: str, action: Action, step_index: int = 0) -> RolloutResult:
    history = decompose_problem(problem)
    state = ReasoningState(problem=problem, history=history[:max(1, step_index + 1)])
    candidate = next_step_for_action(state, action)
    state.steps.append(Step(index=0, text=candidate))

    if action == Action.CONTINUE:
        reward = 0.8
    elif action == Action.REVISE:
        reward = 0.9
    elif action == Action.BRANCH:
        reward = 0.75
    else:
        reward = 0.7

    answer = "solution"
    token_cost = 80 + step_index * 15 + (action.value == "continue") * 10
    return RolloutResult(
        problem=problem,
        state=state.state_text,
        action=action,
        reward=reward,
        token_cost=token_cost,
        answer=answer,
        metadata={"step_index": step_index, "candidate": candidate},
    )
