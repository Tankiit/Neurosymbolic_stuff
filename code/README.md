# CoT Controller — ARR Starter

This repository implements the initial research scaffold for the ARR project on learning when to continue, revise, branch, or stop in language-model reasoning.

## Architecture

The codebase is organized around the core research separation described in the project plan:

- `cot_control/programs/` contains DSPy-style reasoning modules
- `cot_control/environment/` contains reasoning-state and action-transition logic
- `cot_control/controllers/` contains fixed, DSPy-style, and TorchRL-style policies
- `cot_control/optimization/` contains GEPA and controller-training logic
- `cot_control/evaluation/` contains outcomes, cost, and transfer utilities

## Directory layout

```text
cot_control/
├── programs/
│   ├── signatures.py        # natural-language DSPy signatures (requires dspy)
│   ├── demos.py             # demonstrations built from the real GSM8K / HotpotQA samples
│   ├── coding.py            # coding-domain operators + LiveCodeBench sample parser
│   ├── decompose.py
│   ├── continue_step.py
│   ├── revise_step.py
│   ├── branch_step.py
│   ├── final_answer.py
│   └── __init__.py
├── environment/
│   ├── reasoning_state.py
│   ├── transitions.py
│   ├── rollout.py
│   └── __init__.py
├── controllers/
│   ├── fixed.py
│   ├── dspy_controller.py
│   ├── torchrl_controller.py
│   └── __init__.py
├── optimization/
│   ├── gepa_programs.py
│   ├── train_controller.py
│   └── __init__.py
├── evaluation/
│   ├── action_outcomes.py
│   ├── accuracy_cost.py
│   ├── execution.py         # sandboxed stdin/stdout test runner (pilot-grade)
│   ├── transfer.py
│   └── __init__.py
├── __init__.py
├── controller.py
├── datasets.py
├── reasoning.py
├── experiment.py
└── README.md
```

## Research logic

This layout makes the separation explicit:

- `programs/`: how each action is executed
- `controllers/`: when each action should be chosen
- `optimization/`: improve instructions or train the policy
- `evaluation/`: assess action outcomes and matched-cost performance

## Quick start

Run inside the `torch-multimodal` conda environment (has dspy, gepa, and torch 2.9.0; torchrl installs at Stage 3):

```bash
conda activate torch-multimodal
cd code
PYTHONPATH=. pytest -q
PYTHONPATH=. python -m cot_control.experiment collect --demo --backend mock --n 4 --output runs/mock_rollouts.jsonl
```

## Notes

The programs are real DSPy predictors over `programs/signatures.py`, seeded with demonstrations built from the real data in `data/raw/` (see `programs/demos.py`): teacher-forced gold GSM8K solution lines for Continue/Decompose, controlled numeric corruptions paired with gold repairs for Revise, and supporting-fact-to-answer examples for FinalAnswer. Branch carries no demos until Week-2 multi-path rollouts exist. Calling a program requires a configured dspy LM; tests use `dspy.utils.DummyLM`. Controllers, optimization, and evaluation remain mock-level. See `../wiki/System-Architecture.md`.
