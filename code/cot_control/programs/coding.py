"""Coding-domain reasoning operations for the Continue/Revise/Branch/Stop controller.

Executable feedback (compiler messages, test outcomes, patches) makes coding a
controlled environment for action selection; see wiki/Coding-Domain.md.
No demonstrations are attached yet: they must come from real coding
trajectories, not fabricated ones.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import dspy


class PlanSolution(dspy.Signature):
    """Break the problem into an ordered implementation plan."""

    problem: str = dspy.InputField()
    plan: List[str] = dspy.OutputField()


class ContinueImplementation(dspy.Signature):
    """Implement the next planned component, returning the full updated program."""

    problem: str = dspy.InputField()
    plan: List[str] = dspy.InputField()
    code: str = dspy.InputField()
    updated_code: str = dspy.OutputField()


class ReviseImplementation(dspy.Signature):
    """Repair the current program using the test or compiler feedback."""

    problem: str = dspy.InputField()
    code: str = dspy.InputField()
    test_feedback: str = dspy.InputField()
    updated_code: str = dspy.OutputField()


class BranchAlgorithm(dspy.Signature):
    """Replace the current approach with a genuinely different algorithm."""

    problem: str = dspy.InputField()
    code: str = dspy.InputField()
    test_feedback: str = dspy.InputField()
    updated_code: str = dspy.OutputField()


class SubmitProgram(dspy.Signature):
    """Finalize the current program for submission; change it as little as possible."""

    problem: str = dspy.InputField()
    code: str = dspy.InputField()
    final_code: str = dspy.OutputField()


class _CodingOp:
    """Shared wrapper: a dspy.Predict over one coding signature."""

    signature: type[dspy.Signature]
    output_field: str

    def __init__(self, demos: Optional[List[dspy.Example]] = None, predictor: Optional[dspy.Module] = None):
        self.predictor = predictor or dspy.Predict(self.signature)
        if demos:
            self.predictor.demos = list(demos)

    @property
    def instruction(self) -> str:
        return self.predictor.signature.instructions

    def __call__(self, **kwargs: Any) -> Any:
        return getattr(self.predictor(**kwargs), self.output_field)


class PlanSolutionProgram(_CodingOp):
    signature = PlanSolution
    output_field = "plan"


class ContinueImplementationProgram(_CodingOp):
    signature = ContinueImplementation
    output_field = "updated_code"


class ReviseImplementationProgram(_CodingOp):
    signature = ReviseImplementation
    output_field = "updated_code"


class BranchAlgorithmProgram(_CodingOp):
    signature = BranchAlgorithm
    output_field = "updated_code"


class SubmitProgramProgram(_CodingOp):
    signature = SubmitProgram
    output_field = "final_code"


def build_coding_programs() -> Dict[str, _CodingOp]:
    return {
        "plan": PlanSolutionProgram(),
        "continue": ContinueImplementationProgram(),
        "revise": ReviseImplementationProgram(),
        "branch": BranchAlgorithmProgram(),
        "submit": SubmitProgramProgram(),
    }


_SAMPLE_RE = re.compile(
    r"Sample Input \d+:\s*(.*?)\s*Sample Output \d+:\s*(.*?)(?=Sample Input \d+:|\nNote\b|$)",
    re.S,
)


def _normalize_ws(text: str) -> str:
    """Collapse the markdown blank lines LiveCodeBench embeds between every token."""
    return re.sub(r"\n\s*\n", "\n", text).strip()


def parse_sample_tests(question_content: str, cap: int = 8) -> List[Dict[str, str]]:
    """Extract embedded sample input/output pairs from a LiveCodeBench statement."""
    pairs = []
    for match in _SAMPLE_RE.finditer(question_content):
        given, expected = _normalize_ws(match.group(1)), _normalize_ws(match.group(2))
        if given and expected:
            pairs.append({"input": given, "output": expected})
    return pairs[:cap]
