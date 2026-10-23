# System Architecture: DSPy + GEPA + TorchRL

## One framework per responsibility

The three frameworks are combined with distinct responsibilities rather than one joint optimisation loop.

| Layer | Framework | Owns | Object |
| --- | --- | --- | --- |
| Action execution | DSPy | defines the reasoning operations and their instructions | $\psi_a$ per action |
| Instruction optimisation | GEPA | improves how each operation is performed, from execution feedback | $\psi_a$ (offline) |
| Action selection | TorchRL | learns which operation to perform at each state | policy $\pi_\phi$ |

The central distinction: GEPA optimises how an action is performed; the controller learns when it should be performed.

## Formalisation

Extends the objective in [Controller-and-Tasks](Controller-and-Tasks.md):

$$
a_t \sim \pi_\phi(a \mid s_t), \qquad s_{t+1} \sim T_\theta(s_{t+1} \mid s_t, a_t; \psi_a)
$$

- $\theta$: frozen reasoner weights (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct)
- $\psi_a$: DSPy instructions for action $a$ — GEPA's optimisation target
- $\phi$: controller parameters — the learned policy's target

This factoring makes the core question precise: learn better operations ($\psi_a$), better selection ($\phi$), or both — and do the two improvements compose?

## Reasoning operations as DSPy modules

One module per operation instead of a single `ChainOfThought` over the whole problem. Real DSPy signatures live in `code/cot_control/programs/signatures.py`: Decompose, ContinueStep, ReviseStep, BranchStep, FinalAnswer, and SelectAction for the LLM-controller design.

Action semantics are enforced by the reasoning environment, not by the modules:

- Revise replaces the target step and invalidates its descendants
- Branch preserves the current trajectory and opens a new branch
- Stop is a controller decision that invokes a shared final-answer extractor

FinalAnswer is deliberately held at default instructions in all main comparisons, so improving the answer-extraction prompt cannot confound action-selection effects. It can be ablated separately.

## Three controller designs

### Design A — GEPA-only LLM controller

The SelectAction prompt chooses the action; GEPA optimises that selection prompt from completed-trajectory feedback. Interpretable rules, fast prototyping, no neural training. This is Role 2 in [GEPA-and-DSPy](GEPA-and-DSPy.md) and the natural baseline before any neural policy.

### Design B — TorchRL controller

A neural policy $\pi_\phi$ selects actions from state representations; DSPy executes with fixed instructions. Train offline from saved counterfactual rollouts first; online policy optimisation only later.

### Design C — GEPA executors + TorchRL controller

TorchRL selects among GEPA-improved operations. This tests whether improvements to execution and to selection are complementary — scientifically the most interesting cell, but not the starting point.

Recommended order: fixed controller + default DSPy first, establish that learned selection helps, then improve execution with GEPA, then combine. This preserves the existing go/no-go logic in [Execution-Plan](Execution-Plan.md).

## Staged implementation

| Stage | Content | Notes |
| --- | --- | --- |
| 1 | DSPy operation modules + fixed-rule controller; trajectory and token logging | runnable end-to-end with the mock backend |
| 2 | Counterfactual rollouts at sampled states; GEPA on Continue/Revise/Branch executors and on SelectAction | produces the RQ4 data |
| 3 | Offline action-value controller in PyTorch; TorchRL as the policy/value machinery; optional online fine-tuning | plain supervised fitting first — the vault already rejects day-one end-to-end RL |
| 4 | Full factorial comparison and program-transfer tests | RQ1 and RQ4 |

TorchRL caution: every transition invokes a language model (slow, variable-length text state), so online PPO is a later-stage tool, not the day-one training loop. Verified 2026-10-23: `dspy 3.4.1` and `gepa 0.1.4` are installed in the `torch-multimodal` conda environment, and `torchrl 0.14.1` resolves cleanly against `torch 2.9.0` (install it at Stage 3).

## Code mapping

The scaffold exists under `code/cot_control/`; current implementations are mock-level (keyword rules and string stubs) with real DSPy signatures ready in `programs/signatures.py`.

| Architecture layer | Code location | Status |
| --- | --- | --- |
| Programs (action execution) | `programs/` — real DSPy predictors over `signatures.py`, demonstrations from the real GSM8K / HotpotQA samples via `demos.py` | real (LM required to call) |
| Environment (state, transitions, rollouts) | `environment/` (`reasoning_state.py`, `transitions.py`, `rollout.py`) | mock |
| Controllers | `controllers/` (`fixed.py`, `dspy_controller.py`, `torchrl_controller.py`) | mock |
| Instruction optimisation | `optimization/gepa_programs.py` | skeleton |
| Controller training | `optimization/train_controller.py`, legacy `experiment.py` | skeleton + working mock pipeline |
| Evaluation | `evaluation/` (`action_outcomes.py`, `accuracy_cost.py`, `transfer.py`) | skeleton |

Demonstration provenance: Continue/Decompose demos are teacher-forced gold GSM8K solution lines; Revise demos pair a controlled numeric corruption with the gold repair; FinalAnswer demos use gold GSM8K solutions and HotpotQA supporting-fact sentences. Branch deliberately has no demonstrations — single gold solutions contain no alternative paths, so branch demos must come from the Week-2 multi-path rollouts.

## Where the interaction gets interesting

The controller's optimal decisions depend on the quality of the available actions. With a weak Revise module that rarely repairs the actual error, a learned controller should almost never select it. Once GEPA improves revision, Revise may become preferable in exactly those states where Branch was previously necessary.

This is RQ4 in [Experiments](Experiments.md) and the reason Design C matters even though it is not the starting point.
