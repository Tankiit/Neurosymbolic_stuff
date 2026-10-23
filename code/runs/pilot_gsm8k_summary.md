# GSM8K Pilot — 2026-10-23

Model: llama3.1:8b via local Ollama, temperature 0, demos_k=3, max 4 CONTINUE steps per problem.
Data: 20 real GSM8K train problems (`data/raw/gsm8k_train_sample.jsonl`). Raw records: `runs/pilot_gsm8k.json`.

## E1 — Do real demonstrations help? (CONTINUE-only trajectories)

| Condition | Accuracy | Out-tokens/problem | Wall time |
| --- | ---: | ---: | ---: |
| Demo-seeded programs (k=3) | **17/20 (85%)** | 289 | 595 s |
| Demo-free programs | 5/20 (25%) | 596 | 1089 s |

- Demonstrations raise accuracy by 60 points and halve generated tokens: they fix both reasoning and output format/verbosity.
- All 5 demo-free successes are a subset of the demo-seeded successes (clean dominance).
- Demo-free failure modes: meandering derivations, format drift ("$6", "$25"), wrong final arithmetic.

## E2 — REVISE vs CONTINUE at matched corrupted states (10 problems)

| Arm | Recovery |
| --- | ---: |
| CONTINUE | 7/10 |
| REVISE | 7/10 |

- Per-problem agreement is 10/10: both arms are correct or wrong on exactly the same problems.
- On the failures the arms often produce byte-identical wrong answers (e.g., 2196 vs 2196) — the continuation and final-answer stages recompute the solution from the question and ignore the corrupted (or repaired) step.
- Interpretation: the action at the intermediate state had no causal effect on the final answer. The pipeline "recovers or fails" independently of the action because the answerer solves rather than extracts.

## Design implication (recorded in the wiki)

Action effects are unmeasurable when the final-answer module can re-derive the answer from the question. Before the revise-vs-branch study can detect anything, the answerer must depend on the reasoning history only (extract, not solve), or intermediate steps must be scored directly. This is a fixable architecture issue, not evidence that actions do not matter.

## Caveats

n=20 (E1) / n=10 (E2); one model; one seed; temperature 0; single numeric corruption per problem on the last corruptible gold line; E1 trajectories are CONTINUE-only.
