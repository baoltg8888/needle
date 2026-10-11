# 2026-10-11 — Peer-reviewed grounding for the next experiments + budget tiers

Follow-up to [results diagnosis](2026-10-11-results-diagnosis-and-improvements.md). That note leaned on 2026 arXiv preprints. This one anchors every planned change in **peer-reviewed** work (conference or journal), so the thesis can cite accepted papers.

Venue status: (C) = confirmed via Consensus peer-reviewed search in this session; (K) = venue from prior knowledge, double-check in the bibliography before submitting.

## 1. Peer-reviewed papers and what each supports

| Paper | Venue | What it says | What it means for Needle |
|---|---|---|---|
| Yang et al., *HotpotQA* | EMNLP 2018 (K) | Supporting-fact annotations and joint answer + evidence metrics | Our evidence coverage / citation precision follow its supporting-fact evaluation |
| Jin et al., *Search-R1* | COLM 2025 (K) | RL with outcome-only (EM) reward trains search agents | Our R0 baseline is this setup |
| Shi et al., *AutoRefine* | NeurIPS 2025 (C) | Adding retrieval-specific rewards to the answer reward under GRPO improves multi-hop QA and search quality | Supports R1 (retrieval + evidence terms) over R0 |
| DeepSeek-AI, *DeepSeek-R1* | Nature 2025 (C) | GRPO with verifiable rewards; format reward plus accuracy reward | Justifies GRPO and our format/invalid penalty |
| Wu et al., *Fine-Grained Human Feedback Gives Better Rewards* | NeurIPS 2023 (K) | Separate per-aspect rewards (relevance, factuality, completeness) beat one holistic reward, and their weights control the trade-off | Supports R3 (re-weighting coverage) and R4 (per-dimension handling) |
| Gao et al., *ALCE: Enabling LLMs to Generate Text with Citations* | EMNLP 2023 (K) | Citation recall and citation precision are distinct, and models are often precise but incomplete | Exactly our pattern: precision 0.75, coverage 0.47 |
| Skalse et al., *Defining and Characterizing Reward Hacking* | NeurIPS 2022 (K) | Any proxy reward short of the true objective can be hacked | Frames R0 s8 (no citations) as reward hacking |
| Gao, Schulman, Hilton, *Scaling Laws for Reward Model Overoptimization* | ICML 2023 (K) | Optimizing a proxy too hard degrades the true objective | Same framing, for the discussion chapter |
| Liu et al., *Understanding R1-Zero-Like Training (Dr. GRPO)* | COLM 2025 (K) | GRPO's division by group std over-weights low-variance (too easy or too hard) questions; removing it is more stable | Cheap fix for our seed instability: one flag |
| Yu et al., *DAPO* | NeurIPS 2025 (K) | Dynamic sampling drops zero-variance groups and refills the batch; clip-higher prevents entropy collapse | Explains R0's 107–128/256 skipped steps; a cheap fix |
| Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* | NeurIPS 2021, outstanding paper (K) | With few seeds, report interval estimates (bootstrap CIs, IQM), never single runs | Justifies our paired bootstrap; argues for ≥3 seeds |
| Henderson et al., *Deep RL that Matters* | AAAI 2018 (K) | Seeds alone can produce "significant" differences | Our R2 41% vs 49% seed gap is a textbook example |

## 2. How this changes the plan
The diagnosis is the same, but the fixes are now each backed by an accepted paper:

| Fix | Backed by | Change | Cost |
|---|---|---|---|
| F1 Close the no-citation loophole | Skalse 2022, HotpotQA 2018 | Answer without citations → invalid | code |
| F2 "Grounded EM" metric (EM=1 and cit. prec=1 and ≥1 gold sentence cited) | ALCE 2023, HotpotQA joint metrics | analysis | free |
| F3 **R3**: coverage weight ×2 | Fine-Grained RLHF 2023, ALCE 2023 | config | 1 run per seed |
| F4 **R4**: Dr. GRPO advantages (no std division) | Dr. GRPO 2025 | 1 flag in `group_advantages` | 1 run per seed |
| F5 Dynamic sampling (resample zero-variance groups) | DAPO 2025 | small loop change | 1 run per seed |
| F6 ≥3 seeds plus IQM/bootstrap reporting | Agarwal 2021, Henderson 2018 | more runs | $$ |
| F7 More data (512 q) | Search-R1, AutoRefine scale | config | 2× per run |

*Swapped from the previous note:* per-dimension adaptive weighting (AMRP) is a 2026 preprint, so it's replaced by Dr. GRPO (F4), which is peer-reviewed and a one-line change.

## 3. Budget tiers
Real cost from the last batch: **1 run (256 q train + 300 q holdout eval) ≈ 2.05 h ≈ $2.25** on L40S Secure at $1.10/h. The pod will auto-stop at the end of each queue, so no idle cost. About $6.80 remains after seed 9.

| Tier | Runs | Total compute | **Credit to add** | What you can claim |
|---|---|---|---|---|
| **Small** | R3 × 2 seeds, R4 × 2 seeds (F1, F2 included) | 4 runs ≈ $9 + $1.50 buffer | **+$5** | "Here is the failure, here are two literature-backed fixes, tested on 2 seeds" |
| **Medium** | Small + 3rd seed for R0, R3, R4 + F5 (DAPO dynamic sampling) on R0 × 2 seeds | 9 runs ≈ $20 + $2 | **+$15** | Every main reward on 3 seeds, so IQM/bootstrap claims meet the Agarwal 2021 standard; shows whether R0's collapse is fixable by the optimizer alone |
| **Large** | Medium + best variant and R0/R1 retrained on 512 questions × 2 seeds (2× cost each) | 9 + 6×2 = 21 run-units ≈ $47 + $3 | **+$45** | Results hold at 2× data; strongest Chapter 4; closest to workshop-paper quality |

## 4. Recommendation
**Medium (+$15)** is the best value for a bachelor thesis. Three seeds per method is the minimum that reviewers and examiners accept for RL claims (Agarwal 2021). It also tests both kinds of fix: the reward (R3) and the optimizer (R4 Dr. GRPO, DAPO dynamic sampling). Large mainly adds confidence, not new findings.
