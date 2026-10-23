"""DSPy signatures for the reasoning operations.

One module per operation so GEPA can optimise execution instructions per
action independently of the controller policy; see wiki/System-Architecture.md.
FinalAnswer deliberately stays at default instructions in all main comparisons
so answer-prompt quality cannot confound action-selection effects.
"""

from __future__ import annotations

import dspy


class Decompose(dspy.Signature):
    """Break the problem into ordered, actionable subgoals."""

    question: str = dspy.InputField()
    subgoals: list[str] = dspy.OutputField()


class ContinueStep(dspy.Signature):
    """Advance the current reasoning toward its next subgoal."""

    question: str = dspy.InputField()
    subgoals: list[str] = dspy.InputField()
    history: str = dspy.InputField()
    next_step: str = dspy.OutputField()


class ReviseStep(dspy.Signature):
    """Repair a specified step using the valid preceding context."""

    question: str = dspy.InputField()
    history: str = dspy.InputField()
    target_step: str = dspy.InputField()
    revised_step: str = dspy.OutputField()


class BranchStep(dspy.Signature):
    """Propose a genuinely different reasoning approach."""

    question: str = dspy.InputField()
    history: str = dspy.InputField()
    alternative_step: str = dspy.OutputField()


class FinalAnswer(dspy.Signature):
    """Produce the final answer from the selected reasoning path."""

    question: str = dspy.InputField()
    history: str = dspy.InputField()
    answer: str = dspy.OutputField()


class SelectAction(dspy.Signature):
    """Choose the next operation for the partial solution."""

    question: str = dspy.InputField()
    subgoals: list[str] = dspy.InputField()
    history: str = dspy.InputField()
    remaining_budget: int = dspy.InputField()
    action: str = dspy.OutputField(desc="One of CONTINUE, REVISE, BRANCH, STOP")
