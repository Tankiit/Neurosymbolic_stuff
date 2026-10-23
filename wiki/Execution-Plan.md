# Four-Week Initial Research Plan

The staged architecture behind these weeks is specified in [System-Architecture](System-Architecture.md): DSPy executes the reasoning operations, GEPA improves their instructions, and the controller learns when to select each action.

## Week 1 — Build the reasoning environment

- Load GSM8K and HotpotQA with disjoint pilot samples
- Implement decomposition, step generation, and four actions as DSPy operation modules with a fixed-rule controller
- Save complete trajectories, branch lineage, and token costs
- Verify deterministic answer scoring and valid action execution

## Week 2 — Counterfactual action study

- Collect partial states from frozen-model trajectories
- Roll out admissible actions from identical states
- Estimate action rewards and compare revise versus branch
- Run GEPA on the Continue/Revise/Branch executors and the SelectAction prompt over the same rollout pool
- Check whether meaningful action differences exist before training

## Week 3 — Learn the controller

- Train action-value predictor on rollout outcomes (plain PyTorch first; TorchRL policy machinery for the online stages)
- Evaluate on held-out GSM8K and HotpotQA
- Run fixed-policy and matched-budget baselines
- Analyse recovery, damage, and action distributions

## Week 4 — Expand toward ARR

- Add FOLIO and GSM-Symbolic evaluation
- Test controller transfer across datasets and models
- Run the central factorial experiment (controller × GEPA-optimised instructions) and the program-transfer test of RQ4
- Decide whether the results support the full ARR narrative

## Go / no-go decision

Before committing to the full controller-learning pipeline, the decisive experiment is:

> At identical partial reasoning states, do different actions produce meaningfully different success–cost outcomes?

### Proceed only if:

- Revise is consistently better for local errors,
- Branch is better for alternative hypotheses,
- Stop is preferred when the answer is already supported,
- the controller can predict those distinctions from reasoning history.

### If not:

- If Continue dominates nearly everything, or
- if Revise and Branch are indistinguishable,

then the four-action grammar will not support a strong paper.

## Recommended starting experiment

The most sensible initial empirical anchor is:

- GSM8K + HotpotQA
- compare Revise against Branch at matched partial states
- determine whether there is a usable action-selection problem before training the controller

This is the best route to establish whether the overall idea is genuinely viable.
