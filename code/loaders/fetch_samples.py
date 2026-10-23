"""Fetch small real samples of GSM8K, HotpotQA, and LiveCodeBench via HuggingFace datasets.

Run inside the torch-multimodal conda environment:

    conda activate torch-multimodal
    python loaders/fetch_samples.py

Writes JSONL samples to data/raw/ and prints compact examples used in the wiki.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import List

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
N_SAVE = 20  # examples per dataset saved to data/raw
N_SHOW = 2   # examples per dataset printed for the wiki
MAX_TEST_CHARS = 2000


def fetch_gsm8k() -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for rec in list(ds)[:N_SAVE]:
        answer_text, answer_final = rec["answer"].rsplit("####", 1)
        rows.append(
            {
                "question": rec["question"],
                "answer_steps": answer_text.strip(),
                "answer": answer_final.strip(),
                "raw_answer": rec["answer"],
            }
        )
    return rows


def fetch_hotpotqa() -> list[dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", trust_remote_code=True)
    rows = []
    for rec in list(ds)[:N_SAVE]:
        # each paragraph is a list of sentences; supporting-fact sent_ids index into it
        paras = dict(zip(rec["context"]["title"], rec["context"]["sentences"]))
        gold_titles = set(rec["supporting_facts"]["title"])
        rows.append(
            {
                "id": rec["id"],
                "question": rec["question"],
                "answer": rec["answer"],
                "type": rec["type"],
                "level": rec["level"],
                "supporting_facts": [
                    {"title": t, "sent_id": s}
                    for t, s in zip(rec["supporting_facts"]["title"], rec["supporting_facts"]["sent_id"])
                ],
                "gold_paragraphs": {t: paras[t] for t in gold_titles},
                "distractor_titles": [t for t in paras if t not in gold_titles],
            }
        )
    return rows


def _normalize_tests(test_cases, cap: int = 5) -> list[dict]:
    """LiveCodeBench test cases -> [{"input": str, "output": str}], capped and truncated."""
    pairs: list[dict] = []
    if isinstance(test_cases, dict):
        raw = zip(test_cases.get("inputs", []), test_cases.get("outputs", []))
        pairs = [{"input": str(i), "output": str(o)} for i, o in raw]
    elif isinstance(test_cases, list):
        for case in test_cases:
            if isinstance(case, dict) and "input" in case:
                pairs.append({"input": str(case.get("input", "")), "output": str(case.get("output", ""))})
    for pair in pairs:
        pair["input"] = pair["input"][:MAX_TEST_CHARS]
        pair["output"] = pair["output"][:MAX_TEST_CHARS]
    return pairs[:cap]


def fetch_livecodebench(n: int = 10) -> list[dict]:
    ds = load_dataset(
        "livecodebench/code_generation_lite", split="test", streaming=True, trust_remote_code=True
    )
    rows = []
    for rec in itertools.islice(iter(ds), n):
        rows.append(
            {
                "question_id": rec["question_id"],
                "question_title": rec["question_title"],
                "question_content": rec["question_content"][:6000],
                "difficulty": rec["difficulty"],
                "contest_date": str(rec["contest_date"]),
                "platform": rec["platform"],
                "starter_code": rec["starter_code"],
                "public_tests": _normalize_tests(rec.get("public_test_cases")),
            }
        )
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def main() -> None:
    gsm8k = fetch_gsm8k()
    save_jsonl(RAW_DIR / "gsm8k_train_sample.jsonl", gsm8k)

    hotpot = fetch_hotpotqa()
    save_jsonl(RAW_DIR / "hotpotqa_distractor_train_sample.jsonl", hotpot)

    lcb = fetch_livecodebench()
    save_jsonl(RAW_DIR / "livecodebench_sample.jsonl", lcb)

    print("\n===== GSM8K examples =====")
    for row in gsm8k[:N_SHOW]:
        print(json.dumps(row, ensure_ascii=False, indent=2))

    print("\n===== HotpotQA examples =====")
    for row in hotpot[:N_SHOW]:
        print(json.dumps(row, ensure_ascii=False, indent=2))

    print("\n===== LiveCodeBench examples =====")
    for row in lcb[:N_SHOW]:
        print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
