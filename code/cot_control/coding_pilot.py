"""Counterfactual coding-action pilot on the real LiveCodeBench sample.

Shared prefix per problem: plan -> initial implementation -> sandbox test report.
From that identical checkpoint state, four single-action arms are executed and
tested directly: STOP (submit as-is), CONTINUE, REVISE, BRANCH. Single-action
rollouts isolate each action's effect without dilution by later steps (the
lesson from the GSM8K E2 null result).

Run inside torch-multimodal with Ollama serving:

    PYTHONPATH=. python -m cot_control.coding_pilot
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import dspy

from .evaluation.execution import run_program_against_tests
from .pilot import token_window
from .programs import build_coding_programs
from .programs.coding import parse_sample_tests

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
LCB_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "livecodebench_sample.jsonl"
PROBLEM_CHARS = 3500
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def patch_size(old: str, new: str) -> int:
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="")
    return sum(1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--out", default=str(RUNS_DIR / "coding_pilot.json"))
    args = parser.parse_args()

    lm = dspy.LM(
        f"openai/{args.model}",
        api_base=args.base_url,
        api_key="ollama",
        temperature=0.0,
        max_tokens=1024,
    )
    dspy.settings.configure(lm=lm)
    programs = build_coding_programs()

    rows = [json.loads(line) for line in LCB_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: Dict = {"model": args.model, "problems": [], "arms": {}}
    arm_stats = {arm: {"full_pass": 0, "pass_sum": 0.0, "n": 0, "tokens_out": 0, "patch_lines": 0}
                 for arm in ("stop", "continue", "revise", "branch")}

    t_start = time.time()
    for row in rows:
        tests = parse_sample_tests(row["question_content"])
        if not tests:
            results["problems"].append({"question_id": row["question_id"], "skipped": "no sample tests"})
            continue
        problem = row["question_content"][:PROBLEM_CHARS]

        mark0 = len(lm.history)
        plan = programs["plan"](problem=problem)
        initial_raw = programs["continue"](problem=problem, plan=plan, code="")
        initial = strip_fences(initial_raw)
        prefix_tokens = token_window(lm, mark0)[1]
        initial_report = run_program_against_tests(initial, tests, timeout=args.timeout)
        feedback = initial_report.feedback_text()

        record = {
            "question_id": row["question_id"],
            "difficulty": row["difficulty"],
            "n_tests": initial_report.total,
            "initial_pass": f"{initial_report.passed}/{initial_report.total}",
            "initial_tokens_out": prefix_tokens,
            "arms": {},
        }

        arm_codes = {
            "stop": (lambda: initial),
            "continue": (lambda: strip_fences(programs["continue"](problem=problem, plan=plan, code=initial))),
            "revise": (lambda: strip_fences(programs["revise"](problem=problem, code=initial, test_feedback=feedback))),
            "branch": (lambda: strip_fences(programs["branch"](problem=problem, code=initial, test_feedback=feedback))),
        }
        for arm, make_code in arm_codes.items():
            mark = len(lm.history)
            t0 = time.time()
            code = make_code()
            report = run_program_against_tests(code, tests, timeout=args.timeout)
            _, tokens_out = token_window(lm, mark)
            full = report.passed == report.total
            delta = patch_size(initial, code)
            record["arms"][arm] = {
                "passed": report.passed,
                "total": report.total,
                "full_pass": full,
                "pass_rate": report.passed / report.total,
                "error": report.error,
                "patch_lines": delta,
                "tokens_out": tokens_out,
                "seconds": round(time.time() - t0, 1),
            }
            stats = arm_stats[arm]
            stats["full_pass"] += full
            stats["pass_sum"] += report.passed / report.total
            stats["n"] += 1
            stats["tokens_out"] += tokens_out
            stats["patch_lines"] += delta
            print(f"[{row['question_id']}:{arm}] {report.passed}/{report.total} full={full} patch={delta}")

        results["problems"].append(record)

    for arm, stats in arm_stats.items():
        n = max(1, stats["n"])
        results["arms"][arm] = {
            "n": stats["n"],
            "full_pass": stats["full_pass"],
            "mean_pass_rate": round(stats["pass_sum"] / n, 3),
            "tokens_out_per_problem": round(stats["tokens_out"] / n, 1),
            "mean_patch_lines": round(stats["patch_lines"] / n, 1),
        }

    # did the action choice matter anywhere?
    diverged = [
        p["question_id"]
        for p in results["problems"]
        if "arms" in p and len({a["full_pass"] for a in p["arms"].values()}) > 1
    ]
    results["action_changed_outcome_on"] = diverged
    results["total_seconds"] = round(time.time() - t_start, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    for arm, s in results["arms"].items():
        print(f"{arm:9} full_pass {s['full_pass']}/{s['n']} | mean pass-rate {s['mean_pass_rate']} | "
              f"patch {s['mean_patch_lines']} lines | {s['tokens_out_per_problem']} out-tokens/problem")
    print("action changed outcome on:", diverged or "none")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
