"""Continue the current reasoning trajectory toward the next subgoal (real DSPy predictor)."""

from __future__ import annotations

from typing import List, Optional

import dspy

from .signatures import ContinueStep


class ContinueStepProgram:
    """Execute the CONTINUE action; demos are teacher-forced gold GSM8K steps."""

    def __init__(self, demos: Optional[List[dspy.Example]] = None, predictor: Optional[dspy.Module] = None):
        self.predictor = predictor or dspy.Predict(ContinueStep)
        if demos:
            self.predictor.demos = list(demos)

    @property
    def instruction(self) -> str:
        return self.predictor.signature.instructions

    def __call__(self, question: str, subgoals: List[str], history: str) -> str:
        return self.predictor(question=question, subgoals=subgoals, history=history).next_step
