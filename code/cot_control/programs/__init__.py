"""Reasoning program modules backed by real DSPy predictors.

Each program wraps a dspy.Predict over the signatures in signatures.py and
carries demonstrations built from the real GSM8K / HotpotQA samples
(see demos.py). Calling a program requires a configured dspy LM.
"""

from .branch_step import BranchStepProgram
from .coding import (
    BranchAlgorithmProgram,
    ContinueImplementationProgram,
    PlanSolutionProgram,
    ReviseImplementationProgram,
    SubmitProgramProgram,
    build_coding_programs,
    parse_sample_tests,
)
from .continue_step import ContinueStepProgram
from .decompose import DecomposeProgram
from .demos import build_program_demos, build_programs
from .final_answer import FinalAnswerProgram
from .revise_step import ReviseStepProgram
from .signatures import (
    BranchStep,
    ContinueStep,
    Decompose,
    FinalAnswer,
    ReviseStep,
    SelectAction,
)

__all__ = [
    "DecomposeProgram",
    "ContinueStepProgram",
    "ReviseStepProgram",
    "BranchStepProgram",
    "FinalAnswerProgram",
    "Decompose",
    "ContinueStep",
    "ReviseStep",
    "BranchStep",
    "FinalAnswer",
    "SelectAction",
    "build_program_demos",
    "build_programs",
    "PlanSolutionProgram",
    "ContinueImplementationProgram",
    "ReviseImplementationProgram",
    "BranchAlgorithmProgram",
    "SubmitProgramProgram",
    "build_coding_programs",
    "parse_sample_tests",
]
