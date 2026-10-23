from __future__ import annotations

import json
from pathlib import Path

import pytest

from cot_control.evaluation.execution import run_program_against_tests

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "livecodebench_sample.jsonl"


def _lcb_rows():
    if not DATA.exists():
        pytest.skip("LiveCodeBench sample not fetched yet")
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_parse_sample_tests_from_real_livecodebench_statement():
    from cot_control.programs import parse_sample_tests

    rows = _lcb_rows()
    good_kid = next(r for r in rows if r["question_id"] == "1873_B")
    tests = parse_sample_tests(good_kid["question_content"])
    assert len(tests) == 1
    assert tests[0]["input"].startswith("4")
    assert tests[0]["output"].splitlines()[0] == "16"
    assert "430467210" in tests[0]["output"]

    short_sort = next(r for r in rows if r["question_id"] == "1873_A")
    tests_a = parse_sample_tests(short_sort["question_content"])
    assert tests_a and tests_a[0]["output"].splitlines()[0] == "YES"
    assert all("Note" not in t["output"] for t in tests_a)  # Note section must not leak in


def test_sandbox_passes_correct_program():
    code = (
        "t = int(input())\n"
        "for _ in range(t):\n"
        "    print(input()[::-1])\n"
    )
    tests = [{"input": "2\nab\ncd\n", "output": "ba\ndc\n"}]
    report = run_program_against_tests(code, tests, timeout=5.0)
    assert report.passed == 1 and report.total == 1
    assert report.error is None


def test_sandbox_catches_wrong_and_crashing_programs():
    tests = [{"input": "1\nab\n", "output": "ba\n"}]
    wrong = run_program_against_tests("print('nope')\n", tests, timeout=5.0)
    assert wrong.passed == 0 and wrong.error is None

    crashing = run_program_against_tests("raise ValueError('boom')\n", tests, timeout=5.0)
    assert crashing.passed == 0 and "boom" in (crashing.error or "")


def test_sandbox_enforces_timeout():
    tests = [{"input": "1\nab\n", "output": "ba\n"}]
    report = run_program_against_tests("import time\ntime.sleep(30)\n", tests, timeout=1.5)
    assert report.passed == 0
    assert "timeout" in (report.error or "")


def test_feedback_text_summarizes_failures():
    tests = [{"input": "1\nab\n", "output": "ba\n"}]
    report = run_program_against_tests("print('nope')\n", tests, timeout=5.0)
    text = report.feedback_text()
    assert text.startswith("passed 0/1 tests")
    assert "expected" in text and "got" in text


def test_coding_programs_execute_against_dummy_lm():
    dspy = pytest.importorskip("dspy")
    from dspy.utils import DummyLM

    from cot_control.programs import ReviseImplementationProgram, build_coding_programs

    assert set(build_coding_programs()) == {"plan", "continue", "revise", "branch", "submit"}
    program = ReviseImplementationProgram()
    assert "repair" in program.instruction.lower() or "test" in program.instruction.lower()
    dspy.settings.configure(lm=DummyLM([{"updated_code": "print('fixed')"}]))
    out = program(problem="p", code="print('buggy')", test_feedback="failed 0/1 tests")
    assert out == "print('fixed')"
