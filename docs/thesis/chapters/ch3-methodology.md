# Chapter 3 — Methodology

> **DRAFT — not submission-ready.** Citations marked † have venues recalled from memory and have not been checked against the source (writer brief, §5). Verify each one, then remove the †. Every number and code detail in this chapter is traced to the writer brief, `configs/`, `logs/`, or source code (see the provenance table at the end).

This chapter describes how the search agent is built, trained, and evaluated. Section 3.1 gives an overview of the research design. Sections 3.2–3.6 describe the data, the search environment, the proposed reward (VEGR), the policy model, and the training algorithm. Section 3.7 defines the evaluation protocol, Section 3.8 lists the experiments, and Section 3.9 reports the implementation and compute details.

## 3.1 Research design overview

The study is a controlled experiment. Only the reward function (and, in two variants, the optimizer) changes between runs. The model, data, environment, hyperparameters, and evaluation protocol stay fixed. This lets differences in behaviour be attributed to the reward rather than to other changes.

Each training episode follows the same loop (Figure 3.1):

1. The agent receives a HotpotQA question.
2. It emits one action per turn: either `SEARCH[query]` or a terminal `ANSWER[...] CITATIONS[...]`.
3. The environment runs the search with BM25 over the question's own paragraphs and returns the results, or validates the answer and citations.
4. When the episode ends, the reward function scores it against HotpotQA's gold answer and gold supporting facts.
5. GRPO compares the rewards of 8 episodes on the same question and updates the LoRA adapter.

`[FIGURE 3.1: System diagram — question → agent (Qwen2.5-7B + LoRA) → SEARCH/ANSWER → SearchEnvironment (BM25) → VEGR reward → GRPO update]`

Three research questions guide the design (Chapter 1):

- **RQ1**: Does GRPO improve a 7B search agent over the un-tuned model?
- **RQ2**: Does VEGR prevent the shortcut behaviour of answer-only reward, without losing accuracy?
- **RQ3**: How do reward variants and optimizer fixes affect grounding and search behaviour?

## 3.2 Dataset and splits

The study uses **HotpotQA** (Yang et al., 2018†), version 1.1, in the *distractor* setting. Each question comes with 10 Wikipedia paragraphs: 2 gold paragraphs that contain the answer evidence, and 8 distractors. Each question is also annotated with **supporting facts**, the (title, sentence index) pairs needed to answer it. These annotations are what make a judge-free evidence reward possible: whether a citation is correct can be checked by rule.

Rows that fail validation are skipped and logged with a reason (for example, an empty sentence or a supporting-fact index out of range).

**Splits.** The split is produced by `scripts/prepare_splits.py` and recorded in `configs/evaluation/splits.json`, together with SHA-256 hashes of the source files.

- **Train / validation.** These are both drawn from the HotpotQA training file using a deterministic hash. Each question ID is mapped to a bucket in [0, 1) by taking the first 8 bytes of `sha256("7:" + question_id)`. A question goes to train if the bucket is below 0.95, and to validation otherwise. This gives **85,573 training** and **4,452 validation** questions. The split does not depend on file order and can be reproduced exactly.
- **Holdout.** The holdout is a frozen set of **300 questions**: the first 300 valid rows of the HotpotQA dev file, stored in `configs/evaluation/holdout_v1_ids.json` along with a hash of the manifest. The holdout was never used to choose settings. All settings were chosen on the validation split, and the holdout is used only for the final numbers reported in Chapter 4.

**Training sample.** Each training run draws **256 questions** from the training split. The sample depends on the run seed, so different seeds see different questions (Section 3.6).

## 3.3 Search environment

The environment (`SearchEnvironment`, `src/needle/environment/search.py`) is a bounded, resettable search session over a single question.

**Corpus and retriever.** The searchable corpus is the question's own 10 paragraphs. Retrieval uses Okapi BM25 (Robertson & Zaragoza, 2009†), via the `rank_bm25` implementation with its default parameters. Each paragraph is indexed as its title plus all its sentences, tokenized into case-folded Unicode word tokens. A search returns the **top 3** documents. Ties are broken by original document order, so retrieval is deterministic. Each returned document is shown in full, with numbered sentences.

**Budget.** An episode allows at most **3 searches**.

**Action grammar.** On each turn the agent must output exactly one action:

- `SEARCH[query]` runs a BM25 search and spends one unit of budget.
- `ANSWER[answer] CITATIONS[title|index; title|index; ...]` gives the final answer together with the cited evidence sentences. This action ends the episode.

**Validation and failure types.** An episode is **invalid** if any of the following happens:

- *Protocol failure*: the output does not match the grammar (for example, extra text, or citations attached to a SEARCH action).
- *Evidence failure*: a citation refers to a title or sentence index that never appeared in any search result in the episode. In practice this mostly means invented citations.
- *Budget failure*: the agent uses all its searches and still does not answer.

Invalid episodes receive the protocol penalty of −1 during training (Section 3.4) and count as wrong during evaluation (Section 3.7).

**Known loophole.** An answer with an empty citation list (`ANSWER[x]` or `ANSWER[x] CITATIONS[]`) is accepted as valid. The prompt itself allows `CITATIONS[]` for the case where no citation can be validated. We did not close this loophole, so that every run is trained and evaluated in the same environment. Chapter 4 reports how the answer-only reward exploited it.

## 3.4 Proposed reward: VEGR

The **Verifiable Evidence-Grounded Reward (VEGR)** is implemented in `src/needle/rewards/__init__.py`. Every term is computed by rule from HotpotQA's gold annotations, so no LLM judge is involved.

Let *G* be the set of gold supporting facts (title, sentence index) and *C* the set of sentences the agent cites. For a valid episode:

$$
R_{\text{VEGR}} = \text{EM} + \text{F1} + \text{Cov} + \text{Prec} + \text{RetR} - 0.1\cdot\text{Cost} - 0.1\cdot\text{Dup}
$$

| Term | Definition | Range |
|---|---|---|
| EM | normalized exact match between predicted and gold answer | {0, 1} |
| F1 | token-level F1 between predicted and gold answer | [0, 1] |
| Cov (evidence coverage) | \|G ∩ C\| / \|G\| | [0, 1] |
| Prec (citation precision) | \|G ∩ C\| / \|C\|, defined as 0 when C is empty | [0, 1] |
| RetR (retrieval recall) | fraction of gold documents that appeared in any search result | [0, 1] |
| Cost (search cost) | searches used / budget (3) | [0, 1] |
| Dup (duplicate-query penalty) | (number of queries − number of unique queries) / number of queries | [0, 1) |

An **invalid** episode receives **R = −1**, regardless of the other terms. This protocol penalty is what makes VEGR enforce the action grammar. Because an invented citation is an evidence failure, the agent cannot raise its precision by citing sentences it never retrieved.

The terms fall into three groups, each aimed at a different failure:

- **Answer terms (EM, F1)** reward correctness. F1 gives partial credit, so the reward is less sparse than EM alone.
- **Evidence terms (Cov, Prec)** reward citing the right sentences. Precision alone could be satisfied by citing a single safe sentence, and coverage alone by citing everything, so both are included (cf. ALCE; Gao et al., 2023†).
- **Search terms (RetR, Cost, Dup)** reward finding the gold documents and penalize wasted or repeated searches. The penalties have a small weight (0.1) so that they shape search behaviour without discouraging search altogether.

The individual terms are not new. The contribution is combining them into a judge-free reward for multi-turn search agents, together with a controlled evaluation of what the combination changes (Section 3.8). All weights are untuned defaults. This is a limitation, discussed in Chapter 5.

**Reward variants.** Three variants change one aspect of VEGR at a time:

- **R0, outcome-only (baseline reward).** R = EM. All other weights are 0, but invalid episodes still receive −1. This mirrors outcome-reward search agents such as Search-R1 (Jin et al., 2025†).
- **R2, gated penalties.** This is VEGR, except that the search-cost and duplicate penalties apply only when EM = 1. The idea is to avoid penalizing search on questions the agent has not yet solved.
- **R3, coverage ×2.** This is VEGR with the evidence-coverage weight raised from 1 to 2. It targets the "precise but incomplete" citation pattern described in Chapter 4.

## 3.5 Policy model

**Base model.** The policy is **Qwen2.5-7B-Instruct** (Qwen Team, 2024†), loaded from local weights in bf16.

**Prompt.** The prompt is deterministic and built by `build_prompt` in `src/needle/inference/runner.py`. It is wrapped in the model's chat template. The prompt:

1. Describes the role ("You are a careful evidence-seeking QA agent").
2. States the action grammar, gives a worked example (`ANSWER[United States] CITATIONS[Ursula K. Le Guin|1; The Left Hand of Darkness|1]`), and lists formatting rules: one `title|index` per citation, semicolon-separated, no text after the closing bracket, titles copied exactly, never invent a title.
3. Gives the question and the remaining search budget.
4. Lists all search results so far. For each search, it shows the query and then every returned document as `[Title]` followed by `index: sentence` lines.

The prompt is rebuilt from scratch every turn, so the model always sees the full search history. The complete prompt text is given in Appendix A. `[TODO: Appendix A — paste full build_prompt output for one example]`

**Adapter.** Training uses LoRA (Hu et al., 2022†) with rank r = 16, α = 32, and dropout 0. The adapter is applied to all attention and MLP projections (q, k, v, o, gate, up, down). The base weights stay frozen, which makes training on a single 48 GB GPU possible.

## 3.6 Training algorithm

Training uses **Group Relative Policy Optimization** (GRPO; Shao et al., 2024†), extended to multi-turn episodes. The implementation is in `scripts/train_grpo.py` and `src/needle/training/grpo.py`.

**Rollouts.** For each training question, the current policy runs **G = 8** full episodes, sampling with temperature 1.0 and top-p 1.0. Each action is limited to 128 new tokens. Each episode receives one scalar reward from the reward function of the run (R0–R3).

**Advantage.** Each reward is normalized within its group:

$$
A_i = \frac{r_i - \operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G}) + 10^{-6}}
$$

All actions in an episode share the episode's advantage. If all 8 rewards in a group are equal, every advantage is zero and the group carries no learning signal. Such **zero-variance groups are skipped**: no update is made. The number of skipped steps is logged, and it turns out to be an important diagnostic (Chapter 4).

**Objective.** The update uses the PPO-style clipped surrogate (Schulman et al., 2017†) with ε = 0.2. Each rollout's log-probability is the **mean over all tokens the agent generated in that episode**, across all its actions. Prompt and observation tokens are excluded. The ratio ρ is therefore computed once per rollout.

$$
\mathcal{L} = -\,\mathbb{E}\left[\min\left(\rho\,A,\; \operatorname{clip}(\rho,\,1-\varepsilon,\,1+\varepsilon)\,A\right)\right],
\qquad \rho = \exp\left(\log\pi_\theta - \log\pi_{\theta_\text{old}}\right)
$$

There is **no KL penalty** to a reference model. Invalid rollouts stay in the batch, so the −1 penalty actively pushes the policy away from malformed actions.

**Optimization.** AdamW with learning rate 2 × 10⁻⁵, gradient-norm clipping at 1.0, bf16, and gradient checkpointing. A run makes one pass over its 256 questions (256 steps, one question per step).

**Seeds.** Runs use seeds 7, 8, and 9. A seed fixes both the 256-question training sample and the random state used for sampling rollouts. Seed-to-seed differences therefore reflect both data and optimization noise.

**Optimizer variants.** Two variants change the optimizer instead of the reward:

- **Dr. GRPO advantages** (Liu et al., 2025†), used in R4. The advantage is computed without dividing by the standard deviation, A_i = r_i − mean(r). This removes the scaling that amplifies groups whose rewards are nearly equal.
- **DAPO-style dynamic sampling** (Yu et al., 2025†), used in R0-dyn. Zero-variance groups do not count toward the step budget. The trainer keeps drawing questions, from a pool of up to 512, until it has made 256 actual optimizer updates. This tests whether R0's shortcut comes from having too few informative updates.

**Pseudocode.**

```
for step, question in enumerate(sample(train_split, 256, seed)):
    episodes = [run_episode(policy, question) for _ in range(8)]   # multi-turn
    rewards  = [reward_fn(ep, gold(question)) for ep in episodes]  # VEGR or variant; invalid → −1
    if all rewards equal:
        log_skip(step); continue                                    # zero-variance group
    A = (rewards − mean(rewards)) / (std(rewards) + 1e-6)          # Dr. GRPO: no division
    for each episode i:
        ρ = exp(mean_token_logp_θ(actions_i) − mean_token_logp_old(actions_i))  # mean over all action tokens
        loss += −min(ρ·A_i, clip(ρ, 0.8, 1.2)·A_i)
    update LoRA with AdamW (lr 2e-5, grad clip 1.0)
```

## 3.7 Evaluation protocol

**Setting.** Every model is evaluated on the same frozen **300-question holdout**, with greedy decoding and **no retries**. If an action is malformed, the episode fails.

**Metrics.**

- **EM (all)** is the headline accuracy number. Every invalid episode counts as 0, so the denominator is always 300.
- **Completed episodes** is the number of valid episodes out of 300. Failures are broken down by type (evidence, protocol, budget).
- **Quality metrics** are F1, evidence coverage, citation precision, retrieval recall, and searches per episode. Each is averaged over completed episodes only, using the definitions in Section 3.4.
- **Grounded EM** counts an episode only if EM = 1, it has at least one citation, and every citation is a gold supporting fact. This is a strict metric: a correct answer with one non-gold citation does not count. It is the measure most directly tied to the thesis claim that answers should be correct *and* supported by evidence.
- **Behavioural diagnostics** come from the episode and training logs. They include the share of episodes that repeat a query, the distribution of citation counts and cited sentence indices, the invalid rate during training, and the number of skipped (zero-variance) training steps.

**Statistics.** Following Agarwal et al. (2021†) and Henderson et al. (2018†), single runs are not treated as conclusive.

- Differences in EM between two models are tested with a **paired bootstrap** over the 300 holdout questions: 10,000 resamples and a 95% percentile confidence interval. A difference is called significant only if the CI excludes 0.
- Each reward variant is trained with **3 seeds** (7, 8, 9). Results are reported per seed and as the mean across seeds.
- Since seeds of the same method can differ by several EM points, a method difference is only claimed when it holds across seeds, not when it shows up for a single seed.

## 3.8 Experiment matrix

| ID | Name | Reward | Optimizer | Seeds |
|---|---|---|---|---|
| B0 | Un-tuned baseline | — (no training) | — | — |
| R0 | Outcome-only | EM, invalid −1 | GRPO | 7, 8, 9 |
| R1 | **VEGR** | full VEGR | GRPO | 7, 8, 9 |
| R2 | Gated | VEGR, penalties only when EM = 1 | GRPO | 7, 8, 9 |
| R3 | Coverage ×2 | VEGR, coverage weight 2 | GRPO | 7, 8, 9 |
| R4 | VEGR + Dr. GRPO | full VEGR | GRPO, no std division | 7, 8, 9 |
| R0-dyn | Outcome-only + dynamic sampling | EM, invalid −1 | GRPO + DAPO dynamic sampling | 8 |

The comparisons map onto the research questions:

- **RQ1:** every trained run against B0.
- **RQ2:** R1 against R0. Same optimizer, same seeds, different reward.
- **RQ3:**
  - R2 and R3 against R1 test reward design.
  - R4 against R1, and R0-dyn against R0, test optimizer fixes.
  - R0-dyn is the key control. If dynamic sampling alone removes R0's shortcut, the problem was the optimizer and not the reward. If it does not, the evidence reward is necessary.

R0-dyn uses only seed 8, the seed on which R0 showed the citation shortcut. [TODO: author to confirm the reason for a single seed; it is not recorded in the plan or logs.]

**Planned vs. post hoc.** B0, R0, R1, and R2 were specified before the multi-seed batch ran (`logs/2026-10-10-r0-r1-r2-batch.md`). Only the single R1 seed-7 run existed at that point. R3, R4, and R0-dyn were added on 2026-10-11, after the R0/R1/R2 results had been analysed (`docs/research/2026-10-11-results-diagnosis-and-improvements.md`). They respond to specific observations: low evidence coverage, and R0's skipped steps and shortcuts. They are therefore **exploratory follow-ups**, and Chapter 4 reports them as such rather than as confirmatory tests.

## 3.9 Implementation and compute

**Software.** Python 3.11 (managed with `uv`), Transformers 4.57.6, PEFT 0.21.2, and `rank_bm25`. PyTorch differs between runs:

- The first runs (B0 and R1 seed 7) used **2.14.1+cu130**.
- All later runs used **2.11.0+cu128**, because the later pod's GPU driver supported only CUDA 12.8.

This is reported as a limitation when comparing against R1 seed 7. Training and evaluation run offline from local weights, with Hugging Face offline mode on.

**Hardware and cost.** One NVIDIA L40S (48 GB) on RunPod at about $1.10 per hour. One run (training plus holdout evaluation) takes about 2.05 hours, or roughly **$2.25**. The total project budget is about $45. `[TODO: final cost table after the fixes batch]`

**Reproducibility.**

- Splits are fixed by hashed manifests (Section 3.2).
- Every run is defined by a JSON config in `configs/training/` and writes `run_metadata.json` (library versions, seed, config), `summary.json`, `episodes.jsonl` (every evaluation episode), and `metrics.jsonl` (per-step training metrics) to `results/<date>/<run>/`.
- Each batch has a dated log in `logs/`.
- The step-by-step procedure for reproducing the runs on RunPod is in `docs/runbooks/runpod.md`.

---

## Provenance (remove before submission)

| Claim in this chapter | Source |
|---|---|
| Split algorithm, 0.95 fraction, 85,573 / 4,452 counts, holdout = first 300 valid dev rows | `configs/evaluation/splits.json`, `scripts/prepare_splits.py` |
| BM25 (`rank_bm25.BM25Okapi`, defaults), tokenizer, tie-breaking | `src/needle/retrieval/bm25.py` |
| Top-k 3, budget 3, failure types, no-citation loophole | writer brief §3; `src/needle/environment/search.py` |
| Prompt content, chat template | `src/needle/inference/runner.py::build_prompt`; `src/needle/training/local_policy.py` |
| Reward formula and weights | `src/needle/rewards/__init__.py`; writer brief §3 |
| GRPO settings, per-rollout sequence-mean ratio, clipping | `src/needle/training/grpo.py`, `scripts/train_grpo.py` |
| Dynamic sampling (512 pool, 256 updates) | `logs/2026-10-11-fixes-batch.md` |
| Library versions, torch change and reason | `results/*/run_metadata.json`; `logs/2026-10-10-r0-r1-r2-batch.md` |
| Cost per run, budget | writer brief §3; `logs/2026-10-11-fixes-batch.md` |
| `uv` | `uv.lock` in repo root |
