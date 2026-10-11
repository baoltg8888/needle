# 2026-10-11 — Why the results look like this, and how to improve them

Based on the R0/R1/R2 batch ([log](../../logs/2026-10-10-r0-r1-r2-batch.md)) and new literature (alphaXiv, Oct 2026). Complements [related work](2026-10-03-related-work-and-directions.md).

## 1. What we observed
| Observation | Data |
|---|---|
| A. All rewards give the same EM gain (+18–26 pts) | R0 41.3%, R1 42.7%, R2 44.8% (seed means) |
| B. R0 shortcuts: no citations (s8) or duplicate-query spam (s7) | R0 s8 cit. prec 0.00, 1.0 search; R0 s7 229/278 episodes repeat a query |
| C. Evidence coverage barely moves | B0 0.47 → R1 0.47–0.51 |
| D. Large seed variance | R2 41.0% vs 48.7% |

## 2. Why — diagnosis from our own data

**A. The gain is mostly "learning the protocol".** B0 lost 130/300 episodes to invented citations, format errors, or the budget. Every trained run has ≤22 such failures. The −1 invalid penalty exists in all three rewards, so all three learn it equally. EM on *completed* episodes moves less (B0 39% → ~45%).

**B. R0 had almost no learning signal half the time.** EM-only reward is 0/1, so a group where all 8 rollouts are wrong (or all right) has zero variance and is skipped:

| Run | Skipped (zero-variance) steps |
|---|---|
| R0 s7 | 107 / 256 |
| R0 s8 | 128 / 256 |
| R1 s8 | 8 / 256 |
| R2 s8 | 20 / 256 |

R0 learns from about half the data, with a coarse signal ("right or wrong"). Nothing in it says "cite" or "don't repeat queries", so whichever habit happens to appear early gets locked in. This is seed-dependent, which explains the two opposite failure modes. CIGPO (2607.16244) calls this "zero-advantage lock-in" in HotpotQA GRPO. Aggregation-induced hacking (2609.00213) describes the general mechanism: whichever dimension is easiest gets optimized first.

**Environment loophole:** `ANSWER[x]` with no `CITATIONS` is accepted as valid. R0 s8 exploited this. It's worth reporting as a reward-hacking example.

**C. Why evidence coverage stays flat.** The model cites only 1–2 sentences, and is biased toward sentence 0 of each paragraph:

| R1 s8 | Count |
|---|---|
| Citations per answer | 1 → 97 episodes, 2 → 182, 3 → 8 |
| Cited sentence index | 0 → 233, 1 → 161, 2 → 57 |

HotpotQA gold support averages about 2.4 sentences, often not the first sentence. So the policy learned a safe default: cite the first sentence of the two titles. That's precise enough to avoid the penalty, but it misses the bridge sentence. Coverage is also only one of five summed terms (weight 1 of about 5), so it's a weak gradient. This is the same aggregation problem as in 2609.00213.

**D. Seed variance.** There are only 256 questions × 1 epoch, and seed changes both the question sample and the rollouts. Small-data GRPO is known to be noisy; Xu et al. (2602.19526) report GRPO as the least stable of PPO/GRPO/REINFORCE for Search-R1-style agents.

## 3. Related work that explains or supports this
- **How to Train Your Deep Research Agent (Search-R1++), 2602.19526.** GRPO is the least stable optimizer for search agents. F1-only reward causes "answer avoidance", fixed by small action-level penalties (F1+). A short "fast thinking" prompt is more stable than long `<think>`. *Supports:* our action-level penalties (R1/R2) are what keep behaviour stable, versus R0.
- **Aggregation-induced reward hacking (AMRP), 2609.00213.** Summing multiple rewards with fixed weights lets the policy max out easy terms (format, citation precision) and neglect hard ones (coverage). The fix is adaptive per-dimension weights. *Explains:* precision rises but coverage stays flat.
- **CIGPO, 2607.16244.** Same task (HotpotQA, Qwen2.5). Outcome-only GRPO collapses via zero-variance groups. Per-turn information-gain rewards keep variance up. *Explains:* R0's 50% skipped steps.
- **Beyond Outcome Rewards: retrieval credit, 2610.10179.** Intermediate retrieval rewards help, and where credit is assigned (whole trajectory vs. the specific search turn) matters. *Supports:* our retrieval-recall term. Per-turn credit is the next step.
- **CaRR / C-GRPO and STAMP** (already in the related-work doc): outcome-only RL shrinks search and produces shortcuts. *Confirmed* by R0.

## 4. How we could improve it (ranked by cost vs. value for a bachelor thesis)

| # | Change | Fixes | Effort | Needs GPU? |
|---|---|---|---|---|
| 1 | **Close the loophole:** an answer with no citations is invalid (or gets 0 evidence credit) | B | 1 line + test | eval only |
| 2 | **Report EM on completed episodes and a "grounded-correct" metric** (EM=1 and cit. prec=1) | A, makes R0 vs R1 visible in one number | analysis only | no |
| 3 | **Increase the evidence-coverage weight** (e.g., 2×) or use recall-style coverage, so citing the bridge sentence pays | C | config | yes (1 run) |
| 4 | **Per-dimension normalization** (normalize each reward term within the group before summing, as in GDPO/AMRP and CIGPO's separate normalization) | C | small code change | yes |
| 5 | **More data / epochs** (512 q or 2 epochs) and 3+ seeds | D | config | yes, $$ |
| 6 | **Per-turn retrieval credit** (STAMP / 2610.10179 / CIGPO information gain) | B, C | larger | yes |
| 7 | **Try REINFORCE (leave-one-out baseline)** instead of GRPO | D | medium | yes |

**Recommendation with the remaining budget (~$2–3 after seed 9):** do #1 and #2 now. They're free, strengthen Chapter 4, and #2 gives a single headline number showing why R1 beats R0. Put #3–#7 in "Future work". #3 is the most natural one-line follow-up if more credit becomes available.

## 5. Thesis framing
- The −1 invalid penalty does most of the accuracy work, so "learning the protocol" is the main RL gain at this scale.
- Rewarding evidence doesn't raise accuracy, but it is what makes the agent show its work. Without it, behaviour depends on the seed and drifts into shortcuts.
- Evidence coverage is the open problem: the summed reward lets the policy settle for "precise but incomplete" citations. That's a known multi-reward failure mode, with known fixes.
