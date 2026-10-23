"""Propose a genuinely different reasoning approach (real DSPy predictor)."""

from __future__ import annotations

from typing import List, Optional

import dspy

from .signatures import BranchStep


class BranchStepProgram:
    """Execute the BRANCH action.

    No demonstrations are attached yet: single gold solutions contain no
    alternative paths, so branch demos must come from Week-2 multi-path
    counterfactual rollouts.
    """

    def __init__(self, demos: Optional[List[dspy.Example]] = None, predictor: Optional[dspy.Module] = None):
        self.predictor = predictor or dspy.Predict(BranchStep)
        if demos:
            self.predictor.demos = list(demos)

    @property
    def instruction(self) -> str:
        return self.predictor.signature.instructions

    def __call__(self, question: str, history: str) -> str:
        return self.predictor(question=question, history=history).alternative_step
