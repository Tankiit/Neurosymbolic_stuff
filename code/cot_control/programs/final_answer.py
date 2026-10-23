"""Produce the final answer from the selected reasoning path (real DSPy predictor)."""

from __future__ import annotations

from typing import List, Optional

import dspy

from .signatures import FinalAnswer


class FinalAnswerProgram:
    """Shared final-answer extractor.

    Held at default instructions in all main comparisons (confound control);
    demos come from gold GSM8K solutions and HotpotQA supporting facts.
    """

    def __init__(self, demos: Optional[List[dspy.Example]] = None, predictor: Optional[dspy.Module] = None):
        self.predictor = predictor or dspy.Predict(FinalAnswer)
        if demos:
            self.predictor.demos = list(demos)

    @property
    def instruction(self) -> str:
        return self.predictor.signature.instructions

    def __call__(self, question: str, history: str) -> str:
        return self.predictor(question=question, history=history).answer
