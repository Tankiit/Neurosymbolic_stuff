"""Fetch small real samples of GSM8K and HotpotQA via HuggingFace datasets.

Run inside the torch-multimodal conda environment:

    conda activate torch-multimodal
    python loaders/fetch_samples.py

Writes JSONL samples to data/raw/ and prints compact examples used in the wiki.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
N_SAVE = 20  # examples per dataset saved to data/raw
N_SHOW = 2   # examples per dataset printed for the wiki


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

    print("\n===== GSM8K examples =====")
    for row in gsm8k[:N_SHOW]:
        print(json.dumps(row, ensure_ascii=False, indent=2))

    print("\n===== HotpotQA examples =====")
    for row in hotpot[:N_SHOW]:
        print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
