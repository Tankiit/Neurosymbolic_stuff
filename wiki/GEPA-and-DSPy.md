# GEPA and DSPy as Optimizers

## Why this page exists

GEPA (Genetic-Pareto prompt evolution) and DSPy are the strongest current tools for optimizing programs that run on frozen LLMs. The wiki already lists "GEPA-optimised reasoning program" as one baseline row, but for this project the optimizers can play four distinct roles, and conflating them weakens both the experiments and the story.

> Positioning: the contribution is the state-conditioned decision problem, not offline prompt evolution. Offline optimizers and the learned controller are compatible; the paper should compare them under matched budgets rather than argue one replaces the other.

These four roles are situated inside the full three-framework architecture (DSPy executors, GEPA instruction optimisation, TorchRL selection) specified in [System-Architecture](System-Architecture.md). Role 2 below is controller Design A there.

## Four distinct roles

| Role | What is optimized | Relation to the controller |
| --- | --- | --- |
| 1. Strongest offline baseline | A static reasoning program | Competitor |
| 2. Controller as evolved prompt | The action-selection prompt | Alternative implementation |
| 3. Base step-generator optimization | Step-generation prompt, controller fixed | Ablation axis |
| 4. DSPy as execution harness | Nothing (engineering substrate) | Infrastructure |

### Role 1 — Strongest offline baseline (already planned)

GEPA evolves one reasoning program offline against a dev set. It cannot condition on the runtime partial state; the controller can.

Expected pattern, worth stating as a hypothesis:

- The GEPA program should win on problems with homogeneous failure modes.
- The controller should win where failure modes differ across states (local arithmetic slips vs wrong entity links vs already-solved problems).

### Role 2 — The controller itself as an evolved prompt

Instead of (or beside) the neural action-value head, express the controller as a DSPy module and let GEPA evolve its instructions:

```
Input:  partial reasoning state, subgoal status, budget remaining, branch lineage
Output: action ∈ {continue, revise, branch, stop}, target step, one-line rationale
```

The Week-2 counterfactual rollout data is exactly the evaluation set this needs: for each candidate prompt, replay it on sampled partial states, apply the actions it chooses, complete the rollouts, and score.

This yields a natural-language controller whose policy text is directly interpretable, at a fraction of the engineering cost of the neural head. Either comparison outcome is publishable:

- If the evolved prompt matches the neural head, the control problem is real and the policy is interpretable.
- If the neural head wins, that is evidence the state signal matters beyond what instructions capture.

GEPA maintains a Pareto frontier over metric components, which fits the multi-objective framing:

| Component | Direction | Meaning |
| --- | --- | --- |
| accuracy | maximize | final-answer correctness on replayed states |
| recovery | maximize | wrong-prefix trajectories repaired |
| neg_tokens | maximize (negated cost) | completion cost under the prompt's choices |
| neg_damage | maximize (negated cost) | correct trajectories made wrong |

GEPA maximizes every component, so costs must be negated when building the metric dict.

### Role 3 — Optimizing the base step generator

Run MIPROv2 or GEPA on the step-generation prompt with the controller held fixed. This gives the 2x2 ablation (base program x controller) and answers whether controller gains persist over a stronger base program. Without this ablation, a reviewer can attribute controller gains to a weak base prompt.

### Role 4 — DSPy as the execution harness

Use DSPy modules for step generation and action selection independent of any optimizer:

- typed signatures give structured action outputs without hand-rolled parsing,
- `dspy.Evaluate` gives paired, multi-seed rollout loops,
- usage tracking gives per-call token accounting for the cost frontier,
- LM caching discounts shared prefixes across counterfactual rollouts.

DSPy does not provide trajectory trees, branch lineage, or action bookkeeping; those remain project code.

## Fair-comparison protocol

- Match total LM calls, including optimization calls, across GEPA runs and controller training.
- Evaluate on identical instances, paired per problem.
- At least three optimizer seeds on top of the three decoding seeds already planned; report variance for both.
- GEPA's dev minibatches must never touch the held-out test split.
- Log the reflector model. Reflections from 7-8B reflectors can be shallow; consider a stronger reflector while keeping the reasoner frozen, and report both.

## Risks

- Credit assignment for Role 2 is trajectory-level unless states are replayed per-action; replay is the mitigation and is already the Week-2 design.
- GEPA can overfit its dev subspace; keep the test split strict and reuse the paired-instance convention.
- Version churn: pin `dspy` and `gepa` exactly and record the API entry points used (they moved between releases).
- The same fairness question applies to the neural head: its training rollouts consume budget too, and both must be counted.

## Concrete first experiment

1. Collect Week-2 counterfactual rollouts on the GSM8K and HotpotQA samples already in `code/data/raw/`.
2. Train the neural action-value head as planned.
3. In parallel, run GEPA on the controller prompt over the same states with the four-component metric.
4. Compare on held-out instances: fixed policies, GEPA static program (Role 1), GEPA prompt-controller (Role 2), neural head.
5. Report the accuracy-vs-token frontier plus recovery/damage, and quote the evolved policy text.

This sharpens RQ1 (matched-cost frontier) and adds an interpretability angle at low extra cost.

## Environment status

Installed 2026-10-23 in the `torch-multimodal` conda environment (Python 3.11, torch 2.9.0): `dspy 3.4.1` (exposes the `dspy.GEPA` optimizer) and `gepa 0.1.4`. `torchrl 0.14.1` dry-run resolves cleanly against torch 2.9.0; install it at Stage 3 of [System-Architecture](System-Architecture.md).

Install commands for reference:

```bash
conda activate torch-multimodal
pip install dspy==3.4.1 gepa==0.1.4
```

GEPA is usable through the standalone `gepa` API or through DSPy's built-in GEPA optimizer; confirm the exact entry-point signature against the pinned version before wiring it in.
