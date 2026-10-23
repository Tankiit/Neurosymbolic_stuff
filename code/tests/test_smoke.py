from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from cot_control.controller import ActionValueController
from cot_control.datasets import load_dataset
from cot_control.experiment import collect_rollouts, train_controller, evaluate_controller
from cot_control.reasoning import Action, simulate_rollout


class DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_load_dataset_demo():
    rows = load_dataset("demo", split="train", n=2)
    assert len(rows) == 2
    assert all(row.problem for row in rows)


def test_simulate_rollout_actions():
    row = simulate_rollout("What is 2 + 2?", Action.REVISE)
    assert row.action == Action.REVISE
    assert row.reward >= 0.0
    assert row.token_cost > 0


def test_controller_fit_and_predict():
    rollout_rows = [
        simulate_rollout("A", Action.CONTINUE),
        simulate_rollout("B", Action.REVISE),
        simulate_rollout("C", Action.BRANCH),
        simulate_rollout("D", Action.STOP),
    ]
    controller = ActionValueController().fit(rollout_rows)
    pred = controller.predict_action("Problem: A\nPrevious steps: []")
    assert pred in {Action.CONTINUE.value, Action.REVISE.value, Action.BRANCH.value, Action.STOP.value}


def test_collect_and_train_pipeline(tmp_path):
    out_path = tmp_path / "rollouts.jsonl"
    collect_args = DummyArgs(
        backend="mock",
        demo=True,
        dataset="demo",
        split="train",
        n=2,
        rollouts=1,
        checkpoints=1,
        output=str(out_path),
    )
    collect_rollouts(collect_args)

    assert out_path.exists()
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 4

    model_path = tmp_path / "controller.pkl"
    train_args = DummyArgs(input=str(out_path), controller=str(model_path))
    train_controller(train_args)
    assert model_path.exists()

    eval_path = tmp_path / "eval.jsonl"
    eval_args = DummyArgs(backend="mock", dataset="demo", split="test", n=2, controller=str(model_path), output=str(eval_path))
    evaluate_controller(eval_args)
    assert eval_path.exists()
    eval_rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(eval_rows) == 2
    assert all("selected_action" in row for row in eval_rows)
