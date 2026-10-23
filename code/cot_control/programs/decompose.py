"""Decompose a problem into ordered subgoals (real DSPy predictor)."""

from __future__ import annotations

from typing import List, Optional

import dspy

from .signatures import Decompose


class DecomposeProgram:
    """Execute the DECOMPOSE operation; demos come from gold GSM8K solutions."""

    def __init__(self, demos: Optional[List[dspy.Example]] = None, predictor: Optional[dspy.Module] = None):
        self.predictor = predictor or dspy.Predict(Decompose)
        if demos:
            self.predictor.demos = list(demos)

    @property
    def instruction(self) -> str:
        return self.predictor.signature.instructions

    def __call__(self, question: str) -> List[str]:
        return self.predictor(question=question).subgoals
