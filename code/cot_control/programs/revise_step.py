"""Repair a previous step using the valid prefix of the history (real DSPy predictor)."""

from __future__ import annotations

from typing import List, Optional

import dspy

from .signatures import ReviseStep


class ReviseStepProgram:
    """Execute the REVISE action.

    Demos pair a numerically corrupted line (the target step) with the gold
    line as the repair; see programs/demos.py for the construction.
    """

    def __init__(self, demos: Optional[List[dspy.Example]] = None, predictor: Optional[dspy.Module] = None):
        self.predictor = predictor or dspy.Predict(ReviseStep)
        if demos:
            self.predictor.demos = list(demos)

    @property
    def instruction(self) -> str:
        return self.predictor.signature.instructions

    def __call__(self, question: str, history: str, target_step: str) -> str:
        return self.predictor(question=question, history=history, target_step=target_step).revised_step
