# Coding as the Primary Domain

## Why coding first

Coding provides what ordinary chain-of-thought lacks: executable feedback about intermediate decisions. Compilation failures, unit-test outcomes, and behavioural changes are observable, so intermediate rewards do not depend on an LLM judging its own reasoning.

Action histories are also reproducible: branches can be saved as separate working directories or commits, and revisions as patches.

Decision (2026-10-23): coding becomes the primary experimental domain, with HotpotQA or FOLIO retained as the non-coding generalisation anchor. If the same controller framework works for both, the contribution is a general approach to learning the control flow of reasoning programs, not another specialised coding agent.

## Action semantics in code

| Action | Coding interpretation | Example |
| --- | --- | --- |
| Continue | Implement the next planned component | Add a dynamic-programming transition |
| Revise | Modify the existing implementation | Fix an incorrect boundary condition |
| Branch | Preserve the solution; explore a different algorithm | Try graph search instead of greedy |
| Stop | Return the current program | Submit when further changes are unlikely to help |

Operators are implemented as DSPy modules in `code/cot_control/programs/coding.py` (PlanSolution, ContinueImplementation, ReviseImplementation, BranchAlgorithm, SubmitProgram). Sandboxed execution lives in `code/cot_control/evaluation/execution.py`.

## How the three frameworks interact here

- DSPy defines the coding operators (plan, implement, repair, alternative, submit).
- GEPA improves operator instructions using compiler messages, test failures, and successful patches as reflection feedback.
- TorchRL learns action selection from the code state and test history.

Restriction: GEPA optimises action instructions on training problems only; the controller is evaluated on held-out problems with those instructions frozen.

## Datasets and allocation

| Dataset | Role | Initial scale |
| --- | --- | --- |
| LiveCodeBench | Pilot and primary evaluation | 100-300 problems |
| CodeContests | Controller training | 1,000-2,000 problems |
| SWE-bench Verified | Real-world transfer | 50-100 issues |

These are experimental budgets, not official split sizes. LiveCodeBench problems carry contest dates, so training and evaluation windows are separated by release date, and official test cases never become training feedback. A 10-problem real sample is saved at `code/data/raw/livecodebench_sample.jsonl` (fetched 2026-10-23 via `load_dataset("livecodebench/code_generation_lite", trust_remote_code=True)`); with CRUXEval superseded as the code anchor.

### Real example (LiveCodeBench `1873_B`, "Good Kid", easy, Codeforces, contest date 2023-08-21)

> Slavic has an array of $n$ digits; the present is the product of all digits. He wants to add 1 to exactly one digit. What is the maximum product?

Embedded sample tests (parsed by `parse_sample_tests`):

- Input `4` test cases; first: digits `2 2 1 2` → output `16`; last: `9 9 9 9 9 9 9 9 9` → `430467210`.

This is the counterfactual substrate: same problem state, different actions, executable outcomes.

## Research questions sharpened

These operationalise the vault-level RQs in [Experiments](Experiments.md) for code:

- Revise vs branch (RQ2): a DP solution failing several tests — repair the indexing logic or switch algorithm? Both actions are executed from identical program states under matched budget.
- Learned control vs fixed loops (RQ1): compare against plain generation, fixed self-debugging, best-of-N, and tree search; the metric is pass rate versus total execution cost, not pass@1.
- Operator improvement changes the policy (RQ4): train the controller on fixed operators, then on GEPA-optimised ones; watch whether useful revisions reduce unnecessary branching.
- Function-level to repository-level transfer (extends RQ3/RQ4): does a policy from small tasks transfer to SWE-bench, even partially (Stop and Revise first)? A explained negative result is informative.

## Novelty risk

Coding agents already combine execution feedback, search, patches, and adaptive interaction (e.g., SWE-agent). The defensible contribution is narrower:

> Learn a transferable policy for choosing among distinct program-development operations, and show those decisions beat fixed and prompt-optimised control procedures at equivalent compute.

That requires an action-selection experiment, not an end-to-end coding leaderboard.

## First experiment (counterfactual actions)

On LiveCodeBench with a frozen code-capable model, no GEPA and no online RL yet:

1. Generate a plan and an initial partial program.
2. Run syntax checks and training-time tests in a sandbox.
3. At each checkpoint, execute Continue, Revise, Branch, and Stop where admissible, from identical states.
4. Complete each counterfactual trajectory under the same remaining budget.
5. Record test success, patch size, runtime, and generated tokens.
6. Fit a small action-value controller; compare with fixed action-selection rules.

Test discipline: development tests (public samples) for intermediate feedback; hidden or held-out tests for final scoring. The current sandbox (`evaluation/execution.py`) is pilot-grade — temp dir, minimal env, timeouts, rlimits — not a container; scale-up needs container isolation with filesystem and network restrictions.

## Later: hierarchical control

Revise-the-plan and revise-the-implementation are different operations: repair the code when the algorithm is right, revise or branch when the implementation faithfully follows an unsuitable algorithm. A hierarchical controller (choose the level, then the action) could become the stronger contribution, but it is explicitly out of the first pilot.
