"""Build DSPy demonstrations from the real GSM8K and HotpotQA samples.

Gold GSM8K solutions provide step-level supervision by teacher forcing:
each solution line is the target `next_step` given the earlier lines as
history. HotpotQA supporting facts provide evidence-grounded final-answer
demonstrations. Revise demonstrations are built by corrupting one number
in a gold line, with the gold line as the repair target. Branch has no
honest demonstration source until Week-2 multi-path rollouts exist.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List

import dspy

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
GSM8K_PATH = DATA_DIR / "gsm8k_train_sample.jsonl"
HOTPOTQA_PATH = DATA_DIR / "hotpotqa_distractor_train_sample.jsonl"

_ANNOTATION = re.compile(r"<<[^>]*>>")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _clean(text: str) -> str:
    """Strip GSM8K calculator annotations, e.g. '48/2 = <<48/2=24>>24'."""
    return _ANNOTATION.sub("", text).strip()


def _read_jsonl(path: Path) -> List[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def _gsm8k_lines(row: dict) -> List[str]:
    return [_clean(line) for line in row["answer_steps"].splitlines() if _clean(line)]


def gsm8k_decompose_demos(path: Path = GSM8K_PATH) -> List[dspy.Example]:
    """Question -> gold solution lines as the subgoal plan."""
    demos = []
    for row in _read_jsonl(path):
        demos.append(
            dspy.Example(question=row["question"], subgoals=_gsm8k_lines(row)).with_inputs("question")
        )
    return demos


def gsm8k_continue_demos(path: Path = GSM8K_PATH) -> List[dspy.Example]:
    """Teacher-forced step completion: earlier gold lines -> next gold line."""
    demos = []
    for row in _read_jsonl(path):
        lines = _gsm8k_lines(row)
        for i, line in enumerate(lines):
            demos.append(
                dspy.Example(
                    question=row["question"],
                    subgoals=lines,
                    history="\n".join(lines[:i]),
                    next_step=line,
                ).with_inputs("question", "subgoals", "history")
            )
    return demos


def gsm8k_revise_demos(path: Path = GSM8K_PATH, seed: int = 0) -> List[dspy.Example]:
    """Corrupted line -> gold line; the corruption is controlled and documented."""
    rng = random.Random(seed)
    demos = []
    for row in _read_jsonl(path):
        lines = _gsm8k_lines(row)
        for i, line in enumerate(lines):
            numbers = _NUMBER.findall(line)
            if not numbers:
                continue
            num = rng.choice(numbers)
            value = float(num)
            corrupted_num = str(int(value + 3)) if value.is_integer() else str(round(value * 1.5, 2))
            corrupted = line.replace(num, corrupted_num, 1)
            demos.append(
                dspy.Example(
                    question=row["question"],
                    history="\n".join(lines[:i] + [corrupted]),
                    target_step=corrupted,
                    revised_step=line,
                ).with_inputs("question", "history", "target_step")
            )
    return demos


def gsm8k_final_demos(path: Path = GSM8K_PATH) -> List[dspy.Example]:
    demos = []
    for row in _read_jsonl(path):
        demos.append(
            dspy.Example(
                question=row["question"],
                history=_clean(row["answer_steps"]),
                answer=row["answer"],
            ).with_inputs("question", "history")
        )
    return demos


def hotpotqa_final_demos(path: Path = HOTPOTQA_PATH) -> List[dspy.Example]:
    """Supporting-fact sentences (in supporting-facts order) -> gold answer."""
    demos = []
    for row in _read_jsonl(path):
        paragraphs = row["gold_paragraphs"]
        selected = []
        for fact in row["supporting_facts"]:
            title, sent_id = fact["title"], fact["sent_id"]
            sentences = paragraphs.get(title, [])
            if sentences and sent_id < len(sentences):
                selected.append(f"{title}: {sentences[sent_id].strip()}")
        demos.append(
            dspy.Example(
                question=row["question"],
                history="\n".join(selected),
                answer=row["answer"],
            ).with_inputs("question", "history")
        )
    return demos


def build_program_demos(seed: int = 0) -> Dict[str, List[dspy.Example]]:
    """Demonstrations per operation, keyed by program constructor role."""
    return {
        "decompose": gsm8k_decompose_demos(),
        "continue": gsm8k_continue_demos(),
        "revise": gsm8k_revise_demos(seed=seed),
        "final_answer": gsm8k_final_demos() + hotpotqa_final_demos(),
        "branch": [],  # no honest source until Week-2 multi-path rollouts
    }


def build_programs(seed: int = 0) -> Dict[str, object]:
    """Construct all five programs wired with their real-data demonstrations."""
    from .branch_step import BranchStepProgram
    from .continue_step import ContinueStepProgram
    from .decompose import DecomposeProgram
    from .final_answer import FinalAnswerProgram
    from .revise_step import ReviseStepProgram

    demos = build_program_demos(seed=seed)
    return {
        "decompose": DecomposeProgram(demos=demos["decompose"]),
        "continue": ContinueStepProgram(demos=demos["continue"]),
        "revise": ReviseStepProgram(demos=demos["revise"]),
        "branch": BranchStepProgram(),
        "final_answer": FinalAnswerProgram(demos=demos["final_answer"]),
    }
