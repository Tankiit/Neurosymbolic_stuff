# Control Tasks and Controller Design

## Distinguish dataset from control task

A dataset is not the same as a control problem. The same dataset can produce multiple kinds of reasoning decisions.

### Control tasks

| Control task | Desired behavior | Most informative datasets |
| --- | --- | --- |
| Step completion | Continue while unresolved subgoals remain | GSM8K, HotpotQA |
| Error correction | Revise a previous mistaken step | FOLIO, GSM8K |
| Alternative-path exploration | Branch when competing solutions remain viable | HotpotQA, FOLIO |
| Adaptive termination | Stop when an answer is ready and further work is unnecessary | All four |
| Plan repair | Change an earlier subgoal or reasoning dependency | HotpotQA, FOLIO |

## Why revise vs branch is central

These are not interchangeable:

- A wrong arithmetic operation should often be revised.
- An ambiguous entity link may warrant branching.
- An already resolved multi-hop question should stop.

The controller should learn these distinctions from the partial reasoning state rather than simply deciding that difficult problems require more computation.

## Action vocabulary

The controller selects one of four actions at each step.

### Continue

Generate the next step toward the current subgoal.

### Revise

Choose a previous step, retain the valid prefix, and regenerate the problematic step and descendants.

### Branch

Preserve the current trajectory and explore a distinct alternative solution.

### Stop

Terminate the trajectory and return the final answer, or close one branch and select another completed candidate.

Stop invokes a shared final-answer extractor with fixed instructions, so answer-prompt quality cannot confound action-selection effects.

## Formal objective

The objective should be final-answer quality under a controlled computation budget:

$$
\max_\pi \mathbb{E}_\pi[R(\hat{y}, y^*) - \lambda C_{\mathrm{tokens}}]
$$

This keeps the controller from winning simply by producing much more text.

## Separating execution from selection

The transition factors into per-action instructions and a selection policy:

$$
a_t \sim \pi_\phi(a \mid s_t), \qquad s_{t+1} \sim T_\theta(s_{t+1} \mid s_t, a_t; \psi_a)
$$

Here $\psi_a$ are the instructions for executing action $a$ (DSPy modules, optimised offline by GEPA) and $\phi$ is the controller. GEPA improves how an action is performed; the controller learns when to perform it. See [System-Architecture](System-Architecture.md).

## First controller should be supervised

The initial paper should not rely on end-to-end RL. A stronger first-step method is the following:

1. Generate trajectories with a frozen LLM and save intermediate steps.
2. Sample partial states from early, middle, and late reasoning steps.
3. Apply each admissible action to the same state.
4. Complete the resulting trajectories.
5. Estimate rewards from final correctness and token cost.
6. Train a small action-value predictor.

The controller is trained to predict which action is best under a given partial reasoning state.

## State representation

A practical first controller uses:

- a frozen text encoder,
- a small classification or regression head,
- a structured representation of subgoals and step delimiters,
- the reasoning history and branch lineage as inputs.

This is a manageable and defensible first implementation.

## Experimental principle

Before spending effort on a full controller-learning pipeline, the crucial question is:

> At identical partial reasoning states, do different actions produce meaningfully different success–cost outcomes?

If Continue is always best, or if Revise and Branch are indistinguishable in practice, then the action vocabulary is not a strong paper topic.

If Revise is reliably better for local errors, Branch for alternative hypotheses, and Stop for completed solutions, then the learned control problem is meaningful.
