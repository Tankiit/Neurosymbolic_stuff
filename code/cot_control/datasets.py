from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Example:
    problem: str
    answer: str
    metadata: Dict[str, Any]


def _demo_examples() -> List[Example]:
    return [
        Example(
            problem="If a box has 3 apples and 2 more are added, how many apples are there?",
            answer="5",
            metadata={"dataset": "demo", "split": "train"},
        ),
        Example(
            problem="Maria bought 10 pens and gave 4 to her friend. How many pens remain?",
            answer="6",
            metadata={"dataset": "demo", "split": "train"},
        ),
        Example(
            problem="A train travels 40 miles in 2 hours. What is its speed?",
            answer="20",
            metadata={"dataset": "demo", "split": "test"},
        ),
        Example(
            problem="A farmer has 12 sheep. 3 escape. How many remain?",
            answer="9",
            metadata={"dataset": "demo", "split": "test"},
        ),
    ]


def _gsm8k_examples(split: str, n: Optional[int] = None) -> List[Example]:
    examples = _demo_examples()
    filtered = [e for e in examples if e.metadata.get("split") == split or split == "all"]
    if n is not None:
        return filtered[:n]
    return filtered


def _hotpotqa_examples(split: str, n: Optional[int] = None) -> List[Example]:
    examples = [
        Example(
            problem="Which country hosted the 2016 Summer Olympics?",
            answer="Brazil",
            metadata={"dataset": "hotpotqa", "split": split},
        ),
        Example(
            problem="Who wrote the novel Pride and Prejudice?",
            answer="Jane Austen",
            metadata={"dataset": "hotpotqa", "split": split},
        ),
        Example(
            problem="What is the capital of France?",
            answer="Paris",
            metadata={"dataset": "hotpotqa", "split": split},
        ),
    ]
    if n is not None:
        return examples[:n]
    return examples


def load_dataset(dataset: str, split: str = "train", n: Optional[int] = None) -> List[Example]:
    dataset = dataset.lower()
    if dataset == "demo":
        return _demo_examples()[:n] if n is not None else _demo_examples()
    if dataset == "gsm8k":
        return _gsm8k_examples(split=split, n=n)
    if dataset == "hotpotqa":
        return _hotpotqa_examples(split=split, n=n)
    raise ValueError(f"Unsupported dataset: {dataset}")


def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
