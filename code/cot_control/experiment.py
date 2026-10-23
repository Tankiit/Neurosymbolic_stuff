from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .controller import ActionValueController
from .datasets import load_dataset, save_jsonl
from .reasoning import Action, RolloutResult, simulate_rollout


def _action_labels() -> List[str]:
    return [Action.CONTINUE.value, Action.REVISE.value, Action.BRANCH.value, Action.STOP.value]


def _safe_json_load(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def collect_rollouts(args: argparse.Namespace) -> None:
    examples = load_dataset(args.dataset, split=args.split, n=args.n)
    rows: List[Dict[str, Any]] = []

    for example in examples:
        for _ in range(max(1, args.rollouts)):
            for action in [Action.CONTINUE, Action.REVISE, Action.BRANCH, Action.STOP]:
                result = simulate_rollout(example.problem, action, step_index=0)
                rows.append(
                    {
                        "problem": result.problem,
                        "state": result.state,
                        "action": result.action.value,
                        "reward": result.reward,
                        "token_cost": result.token_cost,
                        "answer": result.answer,
                        "metadata": result.metadata,
                    }
                )

    save_jsonl(args.output, rows)
    print(f"Saved {len(rows)} rollout rows to {args.output}")


def train_controller(args: argparse.Namespace) -> None:
    rows = _safe_json_load(args.input)
    rollouts = []
    for row in rows:
        rollouts.append(
            RolloutResult(
                problem=row["problem"],
                state=row["state"],
                action=Action(row["action"]),
                reward=float(row["reward"]),
                token_cost=int(row["token_cost"]),
                answer=row.get("answer", ""),
                metadata=row.get("metadata", {}),
            )
        )

    model = ActionValueController()
    model.fit(rollouts)
    with open(args.controller, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved controller to {args.controller}")


def evaluate_controller(args: argparse.Namespace) -> None:
    with open(args.controller, "rb") as f:
        model = pickle.load(f)

    examples = load_dataset(args.dataset, split=args.split, n=args.n)
    output_rows: List[Dict[str, Any]] = []
    total = 0
    selected_actions = {label: 0 for label in _action_labels()}

    for example in examples:
        state = f"Problem: {example.problem}\nPrevious steps: []"
        scores = model.predict_scores(state)
        best_action = max(scores, key=scores.get)
        selected_actions[best_action] += 1
        total += 1
        output_rows.append(
            {
                "problem": example.problem,
                "selected_action": best_action,
                "scores": scores,
                "answer": example.answer,
            }
        )

    save_jsonl(args.output, output_rows)
    print(f"Evaluated {total} examples; action counts: {selected_actions}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal ARR reasoning controller starter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="collect counterfactual action rollouts")
    collect_parser.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    collect_parser.add_argument("--demo", action="store_true")
    collect_parser.add_argument("--dataset", default="gsm8k")
    collect_parser.add_argument("--split", default="train")
    collect_parser.add_argument("--n", type=int, default=10)
    collect_parser.add_argument("--rollouts", type=int, default=2)
    collect_parser.add_argument("--checkpoints", type=int, default=1)
    collect_parser.add_argument("--output", required=True)

    train_parser = subparsers.add_parser("train", help="train the supervised controller")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--controller", required=True)

    eval_parser = subparsers.add_parser("eval", help="evaluate controller on a dataset")
    eval_parser.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    eval_parser.add_argument("--dataset", default="gsm8k")
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument("--n", type=int, default=10)
    eval_parser.add_argument("--controller", required=True)
    eval_parser.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        collect_rollouts(args)
    elif args.command == "train":
        train_controller(args)
    elif args.command == "eval":
        evaluate_controller(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
