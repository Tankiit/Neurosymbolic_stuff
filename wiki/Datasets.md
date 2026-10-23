# Dataset Selection and Allocation

## Main datasets

### 1. HotpotQA

- Primary dataset for the paper
- Evidence-grounded multi-hop QA
- Supports study of intermediate claim repair, evidence path changes, and stopping after answer resolution
- Use distractor setting at first to avoid retrieval confounds
- The supporting-fact annotations are especially useful for analyzing whether the controller repairs unsupported intermediate claims or switches to a new evidence path

### 2. FOLIO

- Natural-language deductive reasoning benchmark
- Tests whether the controller can recover from incorrect premises or inference steps
- Human-authored examples and first-order logic annotations make logical validity measurable beyond final answer correctness

### 3. GSM8K

- Arithmetic reasoning benchmark and initial controller pilot
- Correctness is easy to check
- Useful for isolating arithmetic mistakes from wrong decomposition strategies
- Not enough by itself to anchor the main EMNLP/ARR paper, but a strong pilot dataset

### 4. GSM-Symbolic

- Robustness and compositional generalization benchmark
- Tests whether the controller responds appropriately when numbers, entities, or irrelevant clauses change
- Especially useful for detecting policies that memorize surface patterns rather than learning control decisions

### 5. CRUXEval (optional appendix)

- Good external generalization experiment
- Focuses on code execution and input/output prediction
- Useful as an appendix to show the controller is not limited to ordinary natural-language reasoning
- Largely superseded as the code anchor now that coding is the primary domain (see below)

## Coding datasets (primary domain, added 2026-10-23)

Coding becomes the primary experimental domain because executable feedback gives objective intermediate rewards; see [Coding-Domain](Coding-Domain.md).

### LiveCodeBench

- Primary pilot and evaluation set
- Competitive programming with executable tests and contest dates for contamination-aware temporal splits
- Public samples serve as development feedback; private/held-out tests are for final scoring only

### CodeContests

- Controller training pool when more trajectories are needed
- Richer competitive-programming tasks with sandboxed test environments
- Enables cross-dataset evaluation against LiveCodeBench

### SWE-bench Verified

- Realistic repository-level transfer experiment (500 validated tasks; 50-100 issue budget)
- Distinguishes local patch revision from exploring alternative fixes
- External validation once the controller works on function-level tasks

## Recommended allocation

| Dataset | Pilot | Controller training | Main evaluation | Main question |
| --- | --- | --- | --- | --- |
| LiveCodeBench | Yes | No | Yes | Revise vs branch under executable feedback? |
| CodeContests | No | Yes | Cross-check | Does the policy transfer across coding sets? |
| SWE-bench Verified | No | No | Transfer | Does control transfer to repository repair? |
| GSM8K | Yes | Yes | Legacy pilot | Can revision fix intermediate mistakes? |
| HotpotQA | Yes | Limited | Yes (generalisation anchor) | Can control repair evidence chains? |
| FOLIO | Yes | Limited | Yes (generalisation anchor) | Does the controller preserve logical validity? |
| GSM-Symbolic | No | No | Yes | Does the policy transfer under problem changes? |
| CRUXEval | No | No | Optional | Superseded by the coding-primary plan |

## Important split conventions

### LiveCodeBench

- Separate training and evaluation windows by problem contest date
- Public sample tests are development feedback only; private or held-out tests decide final scores
- Never let the controller or GEPA optimise against the tests used for final scoring

### HotpotQA

- Start with supplied passages and supporting facts
- Do not add retrieval at the start, because retrieval quality would confound the main reasoning question

### FOLIO

- Split by shared premise set, not by individual conclusion, to prevent leakage between similar logical problems

### GSM-Symbolic

- Group examples by underlying problem template when evaluating generalization
- Prevent paraphrased versions of the same underlying question from appearing across train/test splits

## Proposed sample counts

| Dataset | Pilot | Train | Development | Held-out test |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 100 | 1,000 | 200 | 500 |
| HotpotQA | 100 | 1,500 | 300 | 500 |
| FOLIO | 50 | 400 | 100 | 300 |
| GSM-Symbolic | — | — | — | 500 |
| CRUXEval | — | — | — | 300 |

These counts are initial engineering targets, not rigid methodological claims.

## Core paper design choice

The same dataset can support multiple controller tasks. For example:

- step completion,
- error correction,
- alternative-path exploration,
- adaptive termination,
- plan repair.

The paper should distinguish these control tasks carefully instead of treating all reasoning as a single problem.

## Reference examples from the real datasets

These are verbatim records from the actual datasets (not placeholders), fetched with `load_dataset` in the `torch-multimodal` conda environment. Provenance:

- Fetch script: [code/loaders/fetch_samples.py](../code/loaders/fetch_samples.py)
- Saved samples (20 rows each): `code/data/raw/gsm8k_train_sample.jsonl`, `code/data/raw/hotpotqa_distractor_train_sample.jsonl`
- Coding sample (10 rows): `code/data/raw/livecodebench_sample.jsonl` — a real "Good Kid" example with parsed sample tests appears in [Coding-Domain](Coding-Domain.md)
- These samples also seed the DSPy program demonstrations: see `code/cot_control/programs/demos.py`
- Fetched: 2026-10-23

### GSM8K — `openai/gsm8k`, config `main`, split `train`

Schema per record: `question`, `answer`. The `answer` field is a human-written chain of reasoning whose final line is `#### <number>`. Inline `<<expr=result>>` markers are calculator annotations from the original release; strip them when using the text as prompts.

**Example 1 (train[0])**

Question:

> Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Gold reasoning (verbatim):

```text
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
```

Final answer: **72**

**Example 2 (train[1])**

Question:

> Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Gold reasoning (verbatim):

```text
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
```

Final answer: **10**

Controller relevance: each gold solution line is a natural step unit. Injecting an arithmetic error at one line creates a locally repairable state where Revise should dominate, and the `####` field gives a deterministic scorer.

### HotpotQA — `hotpotqa/hotpot_qa`, config `distractor`, split `train`

Schema per record: `id`, `question`, `answer`, `type` (bridge or comparison), `level`, `supporting_facts` (title + sentence indices), `context` (10 paragraphs: the 2 gold paragraphs plus 8 distractors).

**Example 1 (train[0]) — comparison, medium** (`id` `5a7a06935542990198eaf050`)

Question:

> Which magazine was started first Arthur's Magazine or First for Women?

Gold answer: **Arthur's Magazine**

Supporting facts: `Arthur's Magazine` sent 0; `First for Women` sent 0.

Gold paragraphs:

- **Arthur's Magazine** — "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into 'Godey's Lady's Book'."
- **First for Women** — "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies."

Distractor titles: Radio City (Indian radio station), History of Albanian football, Echosmith, Women's colleges in the Southern United States, First Arthur County Courthouse and Jail, 2014–15 Ukrainian Hockey Championship, Freeway Complex Fire, William Rast.

**Example 2 (train[1]) — bridge, medium** (`id` `5a879ab05542996e4f30887e`)

Question:

> The Oberoi family is part of a hotel company that has a head office in what city?

Gold answer: **Delhi**

Supporting facts: `Oberoi family` sent 0; `The Oberoi Group` sent 0.

Gold paragraphs:

- **Oberoi family** — "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group."
- **The Oberoi Group** — "The Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands."

Distractor titles include `Mohan Singh Oberoi` (a near-miss entity) alongside clearly unrelated paragraphs such as Ritz-Carlton Jakarta, Hotel Bond, and Glennwanis Hotel.

Controller relevance:

- The bridge example exposes the two-hop structure explicitly: the intermediate claim ("the Oberoi family is involved in The Oberoi Group") must be resolved before the second hop; `supporting_facts` makes a wrong intermediate entity link checkable against ground truth, which is exactly the failure mode for Revise-vs-Branch analysis.
- The comparison example has two symmetric evidence paths, a natural case for Branch.
- Distractors such as `Mohan Singh Oberoi` let us measure whether the controller switches evidence paths versus repairing the current one.

