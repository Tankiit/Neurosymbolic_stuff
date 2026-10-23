# Research Plan

## Working title

Learning When to Continue, Revise, Branch, or Stop in Language Model Reasoning

## Main contribution

The main contribution is the decision policy, not just the availability of branching or revision. The goal is to show that learning when to use each operation produces better outcomes than fixed strategies under comparable computation budgets.

This is distinct from prior work that adds search or backtracking to chain-of-thought prompting. The key point is that the controller learns state-dependent choices from reasoning history and partial states, rather than following a predetermined schedule.

## Research claim

Existing systems can generate, revise, and search through reasoning trajectories, but choosing among these operations is itself a sequential decision problem. We study whether a learned controller can make those choices from partial reasoning histories and improve the accuracy–computation tradeoff across mathematical, logical, and evidence-grounded reasoning tasks.

## High-level framing

The paper should be organized around a precise question:

> Can a learned controller improve multi-step reasoning by choosing how to proceed after each step, rather than following a fixed chain-of-thought or fixed search strategy?

The central hypothesis is that adaptive control is useful when the model faces different failure modes:

- local arithmetic or inference mistakes,
- alternative plausible plans,
- unsupported intermediate claims,
- near-complete reasoning that should terminate instead of continuing.

## Why this is a strong ARR paper

The paper is strong if it shows not just that adaptive reasoning helps, but that the controller learns different actions for different reasoning states. This gives a more principled story than simply arguing that more search is better.

## Primary tasks

The project is structured around three reasoning settings:

1. Evidence-grounded multi-hop QA
2. Natural-language deductive reasoning
3. Mathematical reasoning with controlled generalization testing

These are complementary because they require different forms of reasoning and different recovery behaviors.

## Recommended paper arc

The narrative should proceed as follows:

1. Define the action space.
2. Show that action choice matters at matched partial states.
3. Learn a controller from counterfactual action rollouts.
4. Compare against fixed baselines under matched token budgets.
5. Test generalization to new problem families.

## Core idea to emphasize

The paper is not “another CoT variant.” It is a study of learned control over reasoning operations.
