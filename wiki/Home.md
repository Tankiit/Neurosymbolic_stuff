# Research Vault Home

This vault collects the core ideas, datasets, controller formulation, and experiment plan for the project on adaptive reasoning control.

## Core question

> Can a learned controller improve multi-step reasoning by choosing how to proceed after each step, rather than following a fixed chain-of-thought or search strategy?

## Key pages

- [Research-Plan](Research-Plan.md)
- [Datasets](Datasets.md)
- [Coding-Domain](Coding-Domain.md)
- [Controller-and-Tasks](Controller-and-Tasks.md)
- [System-Architecture](System-Architecture.md)
- [Experiments](Experiments.md)
- [GEPA-and-DSPy](GEPA-and-DSPy.md)
- [Execution-Plan](Execution-Plan.md)

## Research narrative

The paper should emphasize the decision policy itself, not simply the addition of branching or revision to chain-of-thought prompting. Methods like Tree-of-Thoughts and GEPA are important references, but the contribution is a learned state-aware controller that chooses among operations under a token budget.

## Primary paper focus

- Primary domain: coding with executable feedback — LiveCodeBench (pilot/evaluation), CodeContests (training), SWE-bench Verified (transfer)
- Non-coding generalisation anchors: HotpotQA, FOLIO
- Mathematical reasoning pilot: GSM8K and GSM-Symbolic
- CRUXEval superseded by the coding-primary plan (2026-10-23)

## Initial target

A strong first experimental claim is not merely “more reasoning works,” but that a learned controller makes qualitatively different decisions for different failure modes:

- revise for locally repairable mistakes,
- branch for plausible alternative plans,
- stop when the answer is already supported.
