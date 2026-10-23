from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy")

from dspy.utils import DummyLM

from cot_control.programs import (
    ContinueStepProgram,
    FinalAnswerProgram,
    build_program_demos,
    build_programs,
)


def test_real_data_demos_cover_expected_operations():
    demos = build_program_demos()
    assert len(demos["decompose"]) >= 20
    assert len(demos["continue"]) >= 40
    assert len(demos["revise"]) >= 5
    assert any(d.answer == "Delhi" for d in demos["final_answer"])
    assert any(d.answer == "72" for d in demos["final_answer"])
    assert demos["branch"] == []  # awaits Week-2 multi-path rollouts


def test_continue_demo_is_teacher_forced_from_gold_solution():
    demos = build_program_demos()["continue"]
    first = demos[0]
    assert "Natalia" in first.question
    assert first.next_step not in first.history  # target is the next gold line


def test_revise_demo_pairs_corruption_with_gold_repair():
    demos = build_program_demos()["revise"]
    demo = demos[0]
    assert demo.revised_step != demo.target_step
    assert demo.target_step in demo.history


def test_programs_execute_against_dummy_lm():
    demos = build_program_demos()
    program = ContinueStepProgram(demos=demos["continue"][:2])
    assert len(program.predictor.demos) == 2
    dspy.settings.configure(lm=DummyLM([{"next_step": "canned step"}]))
    out = program(question="q", subgoals=["g"], history="h")
    assert out == "canned step"


def test_build_programs_wires_demos_everywhere_except_branch():
    programs = build_programs()
    assert all(len(programs[name].predictor.demos) > 0 for name in programs if name != "branch")
    assert programs["branch"].predictor.demos == []


def test_instruction_exposes_signature_docstring():
    program = FinalAnswerProgram()
    assert "final answer" in program.instruction.lower()
