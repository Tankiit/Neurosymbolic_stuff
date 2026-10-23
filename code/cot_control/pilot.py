"""Stage-1 pilot experiments on the real GSM8K sample via a local Ollama LM.

E1 — demonstration value: CONTINUE-only rollouts with demo-seeded programs
     versus demo-free programs; accuracy and generated-token cost.
E2 — action value at matched states: from an identical corrupted prefix,
     REVISE-then-continue versus CONTINUE-only; recovery rate per arm.
     Both arms use demo-seeded programs, isolating the action choice.

Run inside the torch-multimodal env with Ollama serving:

    conda activate torch-multimodal
    PYTHONPATH=. python -m cot_control.pilot --n 20 --e2-n 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dspy

from .programs import (
    BranchStepProgram,
    ContinueStepProgram,
    DecomposeProgram,
    FinalAnswerProgram,
    ReviseStepProgram,
    build_program_demos,
)
from .programs.demos import GSM8K_PATH, _NUMBER, _read_jsonl, _clean, _gsm8k_lines

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def configure_lm(model: str, base_url: str, max_tokens: int = 512) -> dspy.LM:
    lm = dspy.LM(
        f"openai/{model}",
        api_base=base_url,
        api_key="ollama",
        temperature=0.0,
        max_tokens=max_tokens,
    )
    dspy.settings.configure(lm=lm)
    return lm


def make_programs(with_demos: bool, seed: int = 0) -> Dict[str, object]:
    demos = build_program_demos(seed=seed) if with_demos else {}
    return {
        "decompose": DecomposeProgram(demos=demos.get("decompose")),
        "continue": ContinueStepProgram(demos=demos.get("continue")),
        "revise": ReviseStepProgram(demos=demos.get("revise")),
        "branch": BranchStepProgram(),
        "final_answer": FinalAnswerProgram(demos=demos.get("final_answer")),
    }


def token_window(lm: dspy.LM, start: int) -> Tuple[int, int]:
    """(input, output) tokens over lm.history entries added since `start`."""
    tin = tout = 0
    for entry in lm.history[start:]:
        usage = entry.get("usage") or {}
        tin += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        tout += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return tin, tout


def score_numeric(pred: str, gold: str) -> bool:
    found = _NUMBER_RE.findall(str(pred))
    if not found:
        return False
    try:
        return abs(float(found[-1]) - float(gold)) < 1e-6
    except ValueError:
        return False


def run_continue_trajectory(row: dict, programs: Dict[str, object], max_steps: int = 4) -> Dict:
    subgoals = programs["decompose"](row["question"])
    steps = []
    for _ in range(min(len(subgoals) or 1, max_steps)):
        history = "\n".join(steps)
        steps.append(programs["continue"](question=row["question"], subgoals=subgoals, history=history))
    answer = programs["final_answer"](question=row["question"], history="\n".join(steps))
    return {"answer": answer, "steps": steps, "subgoals": subgoals}


def corrupted_state(row: dict, rng: random.Random) -> Optional[Dict]:
    """Gold prefix with one number corrupted in the last line (the revise demo construction)."""
    lines = _gsm8k_lines(row)
    for i in range(len(lines) - 1, -1, -1):  # corrupt the latest corruptible line
        numbers = _NUMBER.findall(lines[i])
        if not numbers:
            continue
        num = numbers[0]
        value = float(num)
        bad = str(int(value + 3)) if value.is_integer() else str(round(value * 1.5, 2))
        corrupted = lines[i].replace(num, bad, 1)
        return {
            "prefix": "\n".join(lines[: i + 1][:-1] + [corrupted]),
            "target_step": corrupted,
            "gold_line": lines[i],
            "gold_prefix": "\n".join(lines[: i + 1]),
        }
    return None


def run_e2_arm(row: dict, programs: Dict[str, object], state: Dict, subgoals: List[str], arm: str) -> Dict:
    if arm == "revise":
        repaired = programs["revise"](
            question=row["question"], history=state["prefix"], target_step=state["target_step"]
        )
        history = state["prefix"].rsplit(state["target_step"], 1)[0] + repaired
        steps = [f"[revised] {repaired}"]
    else:
        history = state["prefix"]
        steps = []
    for _ in range(2):
        step = programs["continue"](question=row["question"], subgoals=subgoals, history=history)
        history = f"{history}\n{step}" if history else step
        steps.append(step)
    answer = programs["final_answer"](question=row["question"], history=history)
    return {"answer": answer, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--n", type=int, default=20, help="E1 problems")
    parser.add_argument("--e2-n", type=int, default=10, help="E2 problems")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RUNS_DIR / "pilot_gsm8k.json"))
    args = parser.parse_args()

    rows = _read_jsonl(GSM8K_PATH)[: args.n]
    lm = configure_lm(args.model, args.base_url)
    rng = random.Random(args.seed)

    results: Dict = {"model": args.model, "seed": args.seed, "e1": {}, "e2": {}, "records": {"e1": [], "e2": []}}

    # E1: demos vs no-demos on CONTINUE-only rollouts
    for condition, with_demos in (("with_demos", True), ("no_demos", False)):
        programs = make_programs(with_demos, seed=args.seed)
        mark = len(lm.history)
        t0 = time.time()
        correct = 0
        for row in rows:
            traj = run_continue_trajectory(row, programs, max_steps=args.max_steps)
            ok = score_numeric(traj["answer"], row["answer"])
            correct += ok
            results["records"]["e1"].append(
                {"condition": condition, "question": row["question"][:80], "gold": row["answer"],
                 "pred": traj["answer"], "correct": ok, "n_steps": len(traj["steps"])}
            )
        tin, tout = token_window(lm, mark)
        results["e1"][condition] = {
            "accuracy": correct / len(rows),
            "correct": correct,
            "n": len(rows),
            "tokens_in": tin,
            "tokens_out": tout,
            "tokens_out_per_problem": tout / len(rows),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[E1:{condition}] accuracy {correct}/{len(rows)} | out-tokens/problem {tout/len(rows):.0f} | {time.time()-t0:.0f}s")

    # E2: matched corrupted states, REVISE vs CONTINUE
    programs = make_programs(True, seed=args.seed)
    arms = {"continue": [0, 0], "revise": [0, 0]}  # [correct, total]
    for row in rows[: args.e2_n]:
        state = corrupted_state(row, rng)
        if state is None:
            continue
        subgoals = programs["decompose"](row["question"])
        for arm in ("continue", "revise"):
            mark = len(lm.history)
            out = run_e2_arm(row, programs, state, subgoals, arm)
            ok = score_numeric(out["answer"], row["answer"])
            arms[arm][0] += ok
            arms[arm][1] += 1
            results["records"]["e2"].append(
                {"arm": arm, "question": row["question"][:80], "gold": row["answer"],
                 "pred": out["answer"], "correct": ok, "target_step": state["target_step"][:60]}
            )
    for arm, (correct, total) in arms.items():
        results["e2"][arm] = {"recovery": correct / total if total else 0.0, "correct": correct, "n": total}
        print(f"[E2:{arm}] recovery {correct}/{total}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
