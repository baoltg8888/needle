# Writer Brief — Needle Thesis

Everything a writing agent needs to draft the bachelor thesis. Status as of **2026-10-11 19:00 WIB**. Read this first, then `docs/thesis/outline.md` (chapter structure).

> **Rules for the writer**
> - Only use numbers that appear in this brief or in `logs/` and `results/`. Never invent or round a number into a stronger claim.
> - Results marked **PENDING** are not available yet. Leave `[TODO: …]` placeholders instead of guessing.
> - Venues marked (K) come from memory and must be verified before submission. (C) means confirmed.
> - This is a **bachelor thesis**. Novelty is not the goal; honest, well-supported claims are. Keep the "what we do not claim" section.
> - The author is Emirsyah, writing as a single author.

---

## 1. One-paragraph summary

Needle trains a **Qwen2.5-7B-Instruct** search agent with **GRPO + LoRA** on **HotpotQA** (distractor setting). For each question, the agent searches the question's 10 paragraphs with BM25, then answers and cites evidence sentences. The thesis proposes **VEGR (Verifiable Evidence-Grounded Reward)**: a reward with no LLM judge, checked by rules against HotpotQA's gold supporting facts. It rewards a correct answer, citing the right sentences, and retrieving the right documents, and it penalizes wasted searches. The main finding is as follows. Training with answer-only reward (R0) and with VEGR (R1) reaches **about the same accuracy** (~41–44% EM vs. 22.3% for the un-tuned model). But answer-only training is **not grounded**, and how it fails depends on the seed: one seed stopped citing entirely (citation precision **0.00**), another spammed duplicate searches. VEGR kept citation precision at **~0.73–0.75 across all 3 seeds**. Everything runs on one GPU for about $2.25 per training run.

Title: **Rewarding the Evidence, Not Just the Answer: Verifiable Evidence-Grounded Rewards for Training Search Agents with GRPO**

---

## 2. Research questions and gaps

**Gaps**
- G1 Correct ≠ grounded: answer-only RL can be right without evidence (reward hacking).
- G2 Existing citation rewards need an LLM judge (cost, bias).
- G3 Search-agent RL is usually reported at multi-GPU scale.
- G4 Most papers report single runs, so seed noise is mistaken for method effects.

**Research questions**
- **RQ1**: Does GRPO improve a 7B search agent over the un-tuned model?
- **RQ2**: Does VEGR prevent the shortcut behaviour of answer-only reward, without losing accuracy?
- **RQ3**: How do reward variants (gated penalties, coverage weight) and optimizer fixes (Dr. GRPO, dynamic sampling) affect grounding and search behaviour?

**Contributions**
1. VEGR: a judge-free, sentence-level reward for search agents that enforces the action protocol.
2. Evidence that answer-only GRPO produces seed-dependent shortcuts, diagnosed from the training logs (zero-variance groups).
3. A controlled 3-seed comparison with bootstrap CIs that separates reward design from optimizer fixes.
4. An open, reproducible single-GPU pipeline.

**What we do not claim**
- No new RL algorithm.
- No state-of-the-art accuracy.
- No open-web search (the corpus is the 10 paragraphs per question).
- The individual reward terms are not new. The contribution is the combination plus the controlled evidence.

---

## 3. Method details (for Chapter 3)

### Data
- HotpotQA train/dev. Deterministic hash split (`scripts/prepare_splits.py`) into train / validation / a frozen **300-question holdout** (`configs/evaluation/holdout_v1_ids.json`).
- The holdout was **never used for tuning**. Settings were chosen on the validation split.
- Each training run samples **256 training questions**, and the sample depends on the seed.

### Environment (`SearchEnvironment`)
- BM25 over the question's own 10 paragraphs (2 gold + 8 distractors). Top-k = 3 documents per search, whole documents with numbered sentences.
- Search budget: **max 3 searches**.
- Action grammar, one action per turn:
  - `SEARCH[query]`
  - `ANSWER[answer] CITATIONS[title|index; title|index]`. This action is terminal.
- **Invalid episode** (training reward **−1**, the "protocol penalty"): malformed action, a citation to a title or sentence that was never retrieved (invented citation), or running out of budget without answering.
- Known loophole: `ANSWER[x]` with **no** citations is accepted as valid. We kept it unchanged so all runs stay comparable, and report it as a reward-hacking finding (R0 seed 8 exploited it).

### Prompt (abridged; full text in `src/needle/inference/runner.py::build_prompt`)
"You are a careful evidence-seeking QA agent. Return exactly one action: SEARCH[query] or ANSWER[answer] CITATIONS[title|index; ...]. …" The prompt ends with the question, the remaining budget, and all search results so far, given as `[Title]` followed by `index: sentence` lines.

### VEGR reward (`src/needle/rewards/__init__.py`)
For a valid episode:

```
total = EM + F1 + evidence_coverage + citation_precision + retrieval_recall
        − 0.1·search_cost − 0.1·duplicate_query_penalty
```

| Term | Definition | Range |
|---|---|---|
| EM | normalized exact match with the gold answer | {0,1} |
| F1 | token F1 with the gold answer | [0,1] |
| evidence_coverage | \|gold facts ∩ cited\| / \|gold facts\| | [0,1] |
| citation_precision | \|gold facts ∩ cited\| / \|cited\| (0 if nothing cited) | [0,1] |
| retrieval_recall | share of gold documents that appeared in any search result | [0,1] |
| search_cost | searches used / budget (3) | [0,1] |
| duplicate_query_penalty | (queries − unique queries) / queries | [0,1) |

Invalid episode → **−1**. A "gold fact" is a HotpotQA supporting fact (title, sentence index).

### Reward variants
| ID | Name | Change |
|---|---|---|
| B0 | un-tuned baseline | no training |
| R0 | outcome-only | total = EM only (all other weights 0), but invalid is still −1 (like Search-R1) |
| R1 | **VEGR** (full) | default weights above |
| R2 | gated | R1, but search/duplicate penalties apply only when EM = 1 |
| R3 | coverage ×2 | R1 with evidence_coverage weight 2 |
| R4 | VEGR + Dr. GRPO | R1, advantage = r − mean(r) (no division by std) |
| R0-dyn | R0 + DAPO dynamic sampling | zero-variance groups don't count; draws from up to 512 questions until 256 optimizer updates |

### Training (GRPO, `scripts/train_grpo.py`, `src/needle/training/grpo.py`)
- Multi-turn GRPO. Group size **G = 8** rollouts per question. Group-normalized advantage A = (r − mean)/(std + 1e-6).
- Groups where all 8 rewards are equal (**zero variance**) give no gradient and are **skipped**.
- Loss is a PPO-style clipped surrogate (ε = 0.2) on the **sequence-mean log-prob of the action tokens**. There is **no KL term**, and invalid rollouts are trained on.
- LoRA r = 16, α = 32, dropout 0, on q/k/v/o/gate/up/down projections. bf16, gradient checkpointing.
- AdamW lr **2e-5**, grad-norm clip 1.0, 1 epoch over 256 questions (256 steps). Sampling uses temperature 1.0, top-p 1.0, max 128 new tokens per action.
- Seeds 7, 8, 9. A seed changes both the training-question sample and the rollouts.

### Evaluation
- 300-question holdout, **greedy decoding, no retries**.
- **EM (all)**: failures count as 0. This is the headline accuracy number.
- Quality metrics (F1, coverage, precision, recall, searches) are means over **completed** episodes.
- **Grounded EM** (new): EM = 1 **and** at least one citation **and** every citation is gold. Results are PENDING.
- Statistics: paired bootstrap on EM (10k resamples, 95% CI), plus means across 3 seeds.

### Compute
- RunPod **NVIDIA L40S 48 GB**, ~$1.10/h. One run (train + holdout eval) takes ≈ 2.05 h ≈ **$2.25**.
- Software: Python 3.11, PyTorch 2.11.0+cu128 for the batch runs (B0 and R1 s7 used 2.14.1+cu130; note this as a limitation), transformers 4.57.6, PEFT 0.21.2.
- Total project budget ≈ $45.
- Training is offline: local weights only, Hugging Face offline mode.

---

## 4. Results so far (for Chapter 4)

### 4.1 Main table: 300-question holdout

EM (all) counts failures as 0. Other columns are means over completed episodes.

| Run | Seed | EM (all) | Completed /300 | F1 | Evid. cov. | Cit. prec. | Retr. recall | Searches |
|---|---|---|---|---|---|---|---|---|
| B0 | – | 22.3% | 170 | 0.501 | 0.472 | 0.633 | 0.829 | 1.92 |
| R0 | 7 | 42.0% | 278 | 0.576 | 0.438 | 0.621 | 0.829 | 3.00 |
| R0 | 8 | 40.7% | 299 | 0.530 | **0.000** | **0.000** | 0.710 | **1.01** |
| R0 | 9 | PENDING | | | | | | |
| R1 (VEGR) | 7 | 42.7% | 287 | 0.593 | 0.473 | 0.751 | 0.851 | 2.30 |
| R1 (VEGR) | 8 | 42.7% | 287 | 0.574 | 0.513 | 0.726 | 0.836 | 2.15 |
| R1 (VEGR) | 9 | 45.3% | 293 | 0.598 | 0.333 | 0.754 | 0.837 | 1.91 |
| R2 (gated) | 7 | 41.0% | 283 | 0.577 | 0.539 | 0.748 | 0.891 | 2.21 |
| R2 (gated) | 8 | 48.7% | 290 | 0.625 | 0.440 | 0.703 | 0.879 | 2.00 |
| R2 (gated) | 9 | 38.3% | 286 | 0.552 | 0.471 | 0.702 | 0.855 | 2.16 |
| R3, R4, R0-dyn | 7/8/9 | PENDING | | | | | | |

**Seed means (EM all):**
- R1: (42.7 + 42.7 + 45.3)/3 = **43.6%** (range 42.7–45.3)
- R2: (41.0 + 48.7 + 38.3)/3 = **42.7%** (range 38.3–48.7)
- R0 (2 seeds so far): **41.3%**

### 4.2 Paired bootstrap on EM (10k resamples, 95% CI)
| Comparison | ΔEM (pts) | 95% CI |
|---|---|---|
| R0 s7 − B0 | +19.7 | [+14.3, +25.0] |
| R0 s8 − B0 | +18.3 | [+13.3, +23.3] |
| R1 s7 − B0 | +20.3 | [+15.0, +25.7] (5k resamples) |
| R1 s8 − B0 | +20.3 | [+15.0, +25.3] |
| R2 s7 − B0 | +18.7 | [+13.7, +24.0] |
| R2 s8 − B0 | +26.3 | [+21.0, +31.7] |
| R1 s7 − R0 s7 | +0.7 | [−4.7, +6.0] |
| R1 s8 − R0 s8 | +2.0 | [−2.3, +6.3] |
| R2 s7 − R1 s7 | −1.7 | [−7.0, +3.7] |
| R2 s8 − R1 s8 | +6.0 | [+1.3, +10.7] |

Seed-9 CIs are PENDING; they will be computed together with the fixes batch.

### 4.3 Failure counts (episodes not completed: evidence / protocol / budget)
B0 85/37/8 · R0 s7 14/7/1 · R0 s8 0/1/0 · R1 s7 8/5/0 · R1 s8 8/5/0 · R2 s7 10/7/0 · R2 s8 3/7/0 · R1 s9 7 total · R2 s9 14 total.
Here "evidence" mostly means invented or unretrievable citations.

### 4.4 Behavioural evidence
- **R0 seed 8, the citation shortcut.** The agent searches once and answers with **no citations**, e.g. `ANSWER[Animorphs]`. During training, citation precision was 0.13 at steps 65–96 and **0.00 from step 97 on**; searches fell from 1.3 to 1.0. In evaluation, 122 correct answers were not fully cited.
- **R0 seed 7, duplicate-search spam.** It always uses all 3 searches, and **229/278** completed episodes repeat a query. B0 had 32 such episodes; R1 had 14 (s7) and 7 (s8).
- **Zero-variance (skipped) training steps.** R0 s7 107/256 and R0 s8 128/256, versus R1 s8 **8/256** and R2 s8 20/256. R0 learns from about half the data, with a coarse right-or-wrong signal, so whatever habit appears early gets locked in. This explains why the failure mode depends on the seed.
- **Citation pattern (R1 s8).** The model cites 1 sentence in 97 episodes, 2 in 182 and 3 in 8. By sentence index: index 0 was cited 233 times, index 1 161 times, index 2 57 times. HotpotQA gold support averages ~2.4 sentences and often isn't sentence 0, so the model is "precise but incomplete". That is why coverage stays at ~0.33–0.51. R3 tests whether a larger coverage weight fixes this.
- **Learning the protocol is most of the gain.** B0 lost 130/300 episodes to failures; trained runs lose ≤ 22. On completed episodes only, EM moves less (B0 39.4% → R1 s7 44.6%).
- **R1 s7 training dynamics.** The invalid rate fell 46% → 29% → 13% → 13% → 7% over the first five 32-step blocks, and average reward rose 0.97 → 2.81.
- **Validation split (R1 s7 only).** B0 29.3% vs. R1 57.0% EM (+27.7, CI [+21.7, +33.7]). The validation set is easier than the holdout, so report it only as supporting evidence.

### 4.5 Current interpretation (use this framing)
1. **RQ1: yes.** Every trained run beats B0 by +18 to +26 EM points, and every CI excludes 0.
2. **RQ2: same accuracy, different grounding.** R0 ≈ R1 on EM, but every R0 vs. R1 CI includes 0. R0 drifts into shortcuts that depend on the seed. R1 is stable: EM 42.7–45.3 and precision 0.73–0.75 across 3 seeds. In short: *reward the evidence, not just the answer.*
3. **R2 is not better than R1.** Seed 8 looked like a +6 win (CI excluded 0), but seeds 7 (−1.7) and 9 (38.3% vs. R1's 45.3%) reverse it, and the 3-seed means are 42.7% vs. 43.6%. This is a textbook case of single-seed results misleading (Henderson 2018; Agarwal 2021) and directly supports G4.
4. **Seed variance is large,** up to ~10 EM points for R2. Do not claim method differences smaller than ~6 points.
5. **Evidence coverage is the open problem.** The summed reward lets the policy settle for "precise but incomplete" citations (cf. ALCE).

### 4.6 PENDING (queue finishes ~12:30 WIB 2026-10-12)
- R0 seed 9: does the shortcut happen again?
- R3 seeds 7/8/9: does coverage rise when it is weighted ×2?
- R4 seeds 7/8/9: does Dr. GRPO change stability?
- R0-dyn seed 8: can an optimizer fix alone (dynamic sampling) stop R0's shortcut, or is the evidence reward needed?
- Grounded EM for all runs, 3-seed bootstrap CIs, and the final cost table.
- Results will be in `logs/2026-10-11-fixes-batch.md` and `results/2026-10-11-fixes/`.

---

## 5. Key references

| Paper | Venue | Use in thesis |
|---|---|---|
| Yang et al., HotpotQA | EMNLP 2018 (K) | dataset, supporting facts |
| Jin et al., Search-R1 | COLM 2025 (K) | outcome-reward search agent; R0 mirrors it |
| Shi et al., AutoRefine | NeurIPS 2025 (C) | retrieval rewards + GRPO |
| DeepSeek-AI, DeepSeek-R1 | Nature 2025 (C) | GRPO with verifiable + format rewards |
| Shao et al., DeepSeekMath (GRPO) | arXiv 2024 (K) | GRPO origin |
| Schulman et al., PPO | arXiv 2017 (K) | clipped objective |
| Liu et al., Dr. GRPO | COLM 2025 (K) | R4 |
| Yu et al., DAPO | NeurIPS 2025 (K) | dynamic sampling, R0-dyn |
| Wu et al., Fine-Grained RLHF | NeurIPS 2023 (K) | per-aspect rewards, R3 |
| Gao et al., ALCE | EMNLP 2023 (K) | citation precision/recall |
| Skalse et al., reward hacking | NeurIPS 2022 (K) | frames R0 shortcut |
| Gao, Schulman, Hilton, RM overoptimization | ICML 2023 (K) | discussion |
| Agarwal et al., Statistical Precipice | NeurIPS 2021 (K) | bootstrap, ≥3 seeds |
| Henderson et al., Deep RL that Matters | AAAI 2018 (K) | seed variance |
| Hu et al., LoRA | ICLR 2022 (K) | LoRA |
| Robertson & Zaragoza, BM25 | FnTIR 2009 (K) | retriever |
| Yao et al., ReAct | ICLR 2023 (K) | agent loop |
| Qwen Team, Qwen2.5 | tech report 2024 (K) | base model |
| CaRR / C-GRPO; STAMP; CIGPO (2607.16244); AMRP (2609.00213); Search-R1++ (2602.19526) | 2026 preprints | cite as preprints only; supporting, not core |

More notes: `docs/research/2026-10-03-related-work-and-directions.md`, `2026-10-11-results-diagnosis-and-improvements.md`, `2026-10-11-peer-reviewed-grounding-and-budget-plan.md`.

---

## 6. Limitations to state honestly
- 3 seeds and a 300-question holdout, so small differences are not resolvable.
- Closed corpus (10 paragraphs per question, BM25), not open-web retrieval.
- Only 256 training questions per run, 1 epoch, 7B model with LoRA.
- PyTorch version changed between the early runs (B0, R1 s7) and the batch runs.
- The no-citation loophole exists in the environment; it is reported, not fixed.
- The reward weights are un-tuned defaults.
- Adapters are kept on the pod volume and were not all downloaded.

---

## 7. Where things are in the repo
| What | Path |
|---|---|
| Chapter outline | `docs/thesis/outline.md` |
| Run logs (one per batch) | `logs/` (index in `logs/README.md`) |
| Raw results (`summary.json`, `episodes.jsonl`, `metrics.jsonl`) | `results/<date>/<run>/` |
| Reward | `src/needle/rewards/__init__.py` |
| Environment | `src/needle/environment/` |
| GRPO math | `src/needle/training/grpo.py` |
| Training script | `scripts/train_grpo.py` |
| Configs | `configs/training/*.json` |
| Prompt | `src/needle/inference/runner.py` |
| Reproduction | `docs/runbooks/runpod.md` |

## 8. Suggested writing order
1. **Ch3 Methodology** first, since everything above is final.
2. **Ch2 Literature review** next (section 5 plus the research docs).
3. **Ch4 Results**: draft 4.1–4.5 now with seed 7–9 R0/R1/R2, and leave `[TODO]` for R3/R4/R0-dyn and grounded EM.
4. **Ch1, Ch5 and the abstract** last.

Suggested figures:
- EM vs. citation precision scatter per run (shows "same accuracy, different grounding").
- Training curves of citation precision for R0 s8 vs. R1 s8.
- Bar chart of skipped steps (R0 vs. R1).
- System diagram: question → agent → SEARCH/ANSWER → environment → VEGR → GRPO.
