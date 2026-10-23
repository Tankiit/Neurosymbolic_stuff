# Experimental Research Questions and Results Plan

## Research questions

### RQ1: Does adaptive reasoning control improve performance at matched computation?

Compare the learned controller against linear chains of thought and fixed search or repair schedules.

The central plot should be accuracy versus total generated tokens, including:

- controller calls,
- revisions,
- branch exploration,
- abandoned trajectories.

A convincing result is not simply improved accuracy. The learned controller should outperform fixed alternatives on at least part of the matched-cost frontier.

### RQ2: When should a reasoning step be revised rather than branched?

This is likely the most distinctive part of the paper.

We distinguish three reasoning states:

- Local repair: a prior inference is wrong, but most of the reasoning structure remains useful.
- Global ambiguity: multiple interpretations remain plausible, making branching valuable.
- Near completion: the answer is already supported and additional reasoning mostly wastes compute.

These states should be inferred retrospectively from counterfactual rollout outcomes rather than assigned by assumption.

The key test is whether the controller chooses different actions for these states and whether those choices improve final performance.

### RQ3: Does a controller learn transferable reasoning behavior?

Train on GSM8K and HotpotQA, then evaluate on FOLIO and GSM-Symbolic without updating the controller.

This is demanding because the domains and step structures differ. A weaker but informative setting is to allow light adaptation on the target domain.

The key distinction is whether the policy transfers directly, benefits from light adaptation, or must be retrained.

### RQ4: How does the controller change when the action executors improve?

GEPA improves the operation instructions ($\psi_a$) while the controller ($\phi$) is held fixed or retrained.

- Does a controller trained with default executors transfer to GEPA-improved executors without retraining?
- Do learned action frequencies shift once, for example, Revise actually repairs errors instead of rewording them?

This tests whether the controller learned general reasoning decisions or overfitted to particular action implementations, and complements the cross-dataset transfer of RQ3. See [System-Architecture](System-Architecture.md).

## Baselines and comparison methods

| Method | Continue | Revise | Branch | Learned control |
| --- | --- | --- | --- | --- |
| Greedy CoT | ✓ | — | — | — |
| Self-consistency | ✓ | — | Independent traces | — |
| Self-Refine / fixed reflection | ✓ | ✓ | — | — |
| Tree-of-Thoughts | ✓ | Backtracking | ✓ | — |
| Fixed search | ✓ | ✓ | ✓ | — |
| Process-verifier-guided search | ✓ | ✓ | ✓ | — |
| GEPA-optimised reasoning program | ✓ | Configurable | Configurable | Offline program optimisation |
| Proposed controller | ✓ | ✓ | ✓ | State-dependent policy |

GEPA and DSPy can serve four distinct roles in this project, not just this one baseline row; see [GEPA-and-DSPy](GEPA-and-DSPy.md). The most interesting second role is running GEPA over the controller's own action-selection prompt on the Week-2 counterfactual rollouts, giving an interpretable natural-language controller to compare against the neural action-value head. The full three-layer architecture (DSPy executors, GEPA instruction optimisation, TorchRL selection) is specified in [System-Architecture](System-Architecture.md).

## Central factorial experiment: controller × action instructions

The main comparison crosses controller design with instruction quality (see [System-Architecture](System-Architecture.md)):

| Controller | Action instructions | Isolates |
| --- | --- | --- |
| Fixed rules | Default DSPy | baseline |
| Fixed rules | GEPA-optimised | value of better execution |
| GEPA-optimised LLM policy | Default DSPy | value of reflective selection |
| GEPA-optimised LLM policy | GEPA-optimised | interaction at the prompt level |
| TorchRL policy | Default DSPy | value of learned sequential selection |
| TorchRL policy | GEPA-optimised | full combination |

Fairness rules:

- GEPA optimisation and controller training use the same training pool; optimisation costs are reported separately and counted in the total budget.
- Disclose the reflector model whenever it is stronger than the deployment model.
- Held-out test sets are used only for final evaluation, never for reflective optimisation or controller training.

## Indispensable ablations

The paper should include the following ablations:

- Continue/Stop
- Continue/Revise/Stop
- Continue/Branch/Stop
- Full four-action policy
- Random controller with the same action frequencies and token budget

These ablations are essential for showing that each operation contributes meaningfully.

## What the paper should contain

### Main result plots

- Accuracy vs token cost curve
- Compact table of overall performance
- Revise-vs-branch intervention plot
- Action distribution breakdown by reasoning phase
- Domain-transfer table across train/test settings

### Qualitative analysis

One reasoning trajectory diagram should show a real HotpotQA example in which:

- an intermediate entity identification is wrong,
- the controller chooses revise or branch,
- the final answer is resolved successfully or not.

The qualitative comparison should contrast the selected action against an alternative action at the same state, not just depict a successful trajectory.

## Recovery vs damage

The paper should report both:

- recovery: how often the model repairs an initially wrong trajectory,
- damage: how often it turns an initially correct trajectory into a wrong one.

Recovery alone can be misleading.

## Engineering defaults to validate

Initial defaults for the pilot should include:

- Reasoning models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct
- Controller: frozen text encoder with small action-value head
- Action budget: at most two active branches and two revisions per problem in pilot
- Evaluation: three decoding seeds
- Paired comparisons on identical problem instances

These are defaults to be validated, not methodological claims.

## Initial pilot results (2026-10-23)

First run of the real pipeline (llama3.1:8b via local Ollama, temperature 0, k=3 demonstrations, CONTINUE-only trajectories, 20 real GSM8K train problems; details in `code/runs/pilot_gsm8k_summary.md`):

- Demonstrations dominate execution quality: demo-seeded programs score 85% versus 25% demo-free, while generating about half the tokens per problem. Execution quality (the GEPA lever) must be pinned down before controller comparisons, otherwise it swamps the control signal.
- The first matched-state action study (REVISE vs CONTINUE at corrupted prefixes) returned a null result with 10/10 per-problem agreement: the final-answer stage recomputes from the question, so the intermediate action has no causal effect on the outcome.
- Design fix required: the final-answer module must extract from the reasoning history rather than re-solve from the question (or intermediate steps must be scored directly). Otherwise action effects are unmeasurable — an architecture correction, not evidence against the control problem.

## Coding-domain instantiation (2026-10-23)

Coding is now the primary domain because executable feedback makes intermediate rewards objective; full plan in [Coding-Domain](Coding-Domain.md). Consequences for this page:

- The central factorial and RQ1-RQ4 carry over, with pass rate versus total execution cost as the headline metric (not pass@1 alone).
- Coding baselines must include plain generation, fixed self-debugging, best-of-N, and tree search, plus agent-style systems (e.g., SWE-agent) as reference points; the novelty claim is the learned action-selection policy at matched compute, not the combination of execution feedback with revision.
- The counterfactual first experiment runs on LiveCodeBench: execute Continue / Revise / Branch / Stop from identical program states, complete under matched budget, and record test success, patch size, runtime, and tokens. Public sample tests are development feedback; held-out tests decide final scores.
