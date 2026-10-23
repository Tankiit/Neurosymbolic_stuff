"""Defines the reasoning state seen by the controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReasoningState:
    problem: str
    subgoals: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    active_branch: int = 0
    remaining_budget: int = 4
    completed: bool = False
    final_answer: Optional[str] = None

    @property
    def text(self) -> str:
        return "\n".join([
            f"Problem: {self.problem}",
            f"Subgoals: {self.subgoals}",
            f"History: {' | '.join(self.history) if self.history else 'empty'}",
            f"Branch: {self.active_branch}",
            f"Budget: {self.remaining_budget}",
        ])
