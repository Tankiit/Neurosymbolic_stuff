# Chain-of-Thought Control Research Wiki

This workspace includes the research vault and the runnable starter codebase for the ARR project on learning when to continue, revise, branch, or stop in language-model reasoning.

## Start here

- [wiki/Home.md](wiki/Home.md)
- [wiki/Research-Plan.md](wiki/Research-Plan.md)
- [wiki/Datasets.md](wiki/Datasets.md)
- [wiki/Controller-and-Tasks.md](wiki/Controller-and-Tasks.md)
- [wiki/System-Architecture.md](wiki/System-Architecture.md)
- [wiki/Experiments.md](wiki/Experiments.md)
- [wiki/GEPA-and-DSPy.md](wiki/GEPA-and-DSPy.md)
- [wiki/Execution-Plan.md](wiki/Execution-Plan.md)

## Research theme

The main question is whether a learned controller can improve multi-step reasoning by choosing the next operation after each partial reasoning step rather than following a fixed chain-of-thought or search schedule.

The project is organized around this three-part architecture:

- DSPy defines the reasoning operations
- GEPA improves the instructions for those operations
- TorchRL learns when to select each operation

## Core action vocabulary

- Continue
- Revise
- Branch
- Stop

## Code layout

- [code](code) — runnable project scaffold
- [code/cot_control/programs](code/cot_control/programs) — DSPy-style reasoning modules
- [code/cot_control/environment](code/cot_control/environment) — reasoning-state and action transitions
- [code/cot_control/controllers](code/cot_control/controllers) — fixed / DSPy / TorchRL controllers
- [code/cot_control/optimization](code/cot_control/optimization) — GEPA and controller training
- [code/cot_control/evaluation](code/cot_control/evaluation) — action outcomes and cost analysis
