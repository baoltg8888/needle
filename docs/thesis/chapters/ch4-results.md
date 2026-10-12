# Chapter 4 — Results and Discussion

> **DRAFT — not submission-ready.** Status as of 2026-10-12, with all 16 trained runs complete. Per-run numbers come from the holdout episode files (`results/2026-10-10-*/`, `results/2026-10-11-fixes/results/*/episodes.jsonl`) and training logs (`metrics.jsonl`), summarised in `logs/2026-10-11-fixes-batch.md`. Every figure, and every confidence interval in this chapter, is recomputed by `scripts/make_thesis_figures.py` (values in `docs/thesis/figures/figures_data.json`). The batch log reports a few CI endpoints that differ from these by 0.1–0.4 points because it used a different random generator. This chapter uses the script's values.

This chapter reports the results in the order of the research questions:

- **4.1** gives the experimental setup.
- **4.2** describes the un-tuned baseline.
- **4.3 to 4.6** answer RQ1, RQ2 and RQ3, with a diagnosis of training dynamics in between (4.5).
- **4.7** shows error analysis and example traces.
- **4.8** discusses the findings and threats to validity.

Throughout, **EM (all)** and **grounded EM** count invalid episodes as wrong, so the denominator is always 300. Quality metrics (F1, coverage, precision, recall, searches) are averaged over **completed** episodes only (Section 3.7).

## 4.1 Experimental setup

| Setting | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct + LoRA (r = 16, α = 32) |
| Training data per run | 256 HotpotQA training questions (seed-dependent sample), 1 epoch; R0-dyn: up to 512 |
| Group size / learning rate | G = 8 / 2 × 10⁻⁵ |
| Search budget / top-k | 3 searches / 3 documents |
| Evaluation | 300-question frozen holdout, greedy decoding, no retries |
| Seeds | 7, 8, 9 (R0-dyn: 8 only) |
| Hardware | 1 × NVIDIA L40S 48 GB (RunPod, ~$1.10/h) |
| Time and cost per run | ≈ 2.05 h ≈ $2.25 (training + holdout evaluation) |
| Total runs and cost | 16 trained runs + B0; ≈ $49 in total (Table 3.2) |

## 4.2 Baseline (B0)

The un-tuned model reaches **22.3% EM** and **13.0% grounded EM** on the holdout. Its main problem is not wrong answers but invalid episodes: only **170 of 300** episodes complete. The 130 failures break down into 85 evidence failures, 37 protocol failures, and 8 budget failures. Most evidence failures are invented or unretrievable citations, meaning the model cites a title or sentence it never retrieved.

When the model does complete an episode, it is reasonably capable. EM over completed episodes is 39.4%, citation precision 0.633, and retrieval recall 0.829 (Table 4.1). This already suggests that much of what training can gain lies in following the protocol, which Section 4.3 confirms.

## 4.3 RQ1: Does GRPO improve the agent over the un-tuned model?

**Table 4.1.** Holdout results for every run. EM (all) and grounded EM count failures as 0; the other columns are means over completed episodes. "Dup." is the number of completed episodes that repeat a search query.

| Run | Seed | EM (all) | Grounded EM | Completed /300 | F1 | Evid. cov. | Cit. prec. | Retr. recall | Searches | Dup. |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | – | 22.3% | 13.0% | 170 | 0.501 | 0.472 | 0.633 | 0.829 | 1.92 | 32 |
| R0 (outcome-only) | 7 | 42.0% | 24.7% | 278 | 0.576 | 0.438 | 0.621 | 0.829 | 3.00 | 229 |
| R0 (outcome-only) | 8 | 40.7% | 0.0% | 299 | 0.530 | 0.000 | 0.000 | 0.710 | 1.01 | 0 |
| R0 (outcome-only) | 9 | 45.0% | 0.0% | 298 | 0.577 | 0.000 | 0.000 | 0.860 | 2.27 | 55 |
| R1 (VEGR) | 7 | 42.7% | 32.3% | 287 | 0.593 | 0.473 | 0.751 | 0.851 | 2.30 | 14 |
| R1 (VEGR) | 8 | 42.7% | 28.0% | 287 | 0.574 | 0.513 | 0.726 | 0.836 | 2.15 | 7 |
| R1 (VEGR) | 9 | 45.3% | 37.7% | 293 | 0.598 | 0.333 | 0.754 | 0.837 | 1.91 | 27 |
| R2 (gated) | 7 | 41.0% | 30.7% | 283 | 0.577 | 0.539 | 0.748 | 0.891 | 2.21 | 34 |
| R2 (gated) | 8 | 48.7% | 32.3% | 290 | 0.625 | 0.440 | 0.703 | 0.879 | 2.00 | 4 |
| R2 (gated) | 9 | 38.3% | 26.3% | 286 | 0.552 | 0.471 | 0.702 | 0.855 | 2.16 | 36 |
| R3 (coverage ×2) | 7 | 41.3% | 26.3% | 289 | 0.570 | 0.607 | 0.701 | 0.854 | 2.01 | 12 |
| R3 (coverage ×2) | 8 | 41.0% | 18.7% | 284 | 0.574 | 0.545 | 0.633 | 0.820 | 1.61 | 5 |
| R3 (coverage ×2) | 9 | 44.7% | 29.0% | 282 | 0.602 | 0.557 | 0.738 | 0.882 | 2.35 | 34 |
| R4 (Dr. GRPO) | 7 | 39.3% | 23.7% | 288 | 0.525 | 0.469 | 0.636 | 0.796 | 1.89 | 35 |
| R4 (Dr. GRPO) | 8 | 46.3% | 30.3% | 293 | 0.620 | 0.509 | 0.693 | 0.865 | 2.30 | 44 |
| R4 (Dr. GRPO) | 9 | 46.3% | 39.0% | 292 | 0.609 | 0.346 | 0.757 | 0.867 | 2.41 | 25 |
| R0-dyn | 8 | 48.7% | 0.0% | 300 | 0.632 | 0.000 | 0.000 | 0.875 | 2.29 | 15 |

**Table 4.2.** Mean ± standard deviation across seeds 7, 8 and 9 (percentage points for EM, grounded EM).

| Method | EM | Grounded EM | Cit. prec. | Evid. cov. |
|---|---|---|---|---|
| R0 outcome-only | 42.6 ± 2.2 | 8.2 ± 14.2 | 0.207 ± 0.358 | 0.146 ± 0.253 |
| **R1 VEGR** | 43.6 ± 1.5 | **32.7 ± 4.8** | **0.744 ± 0.015** | 0.440 ± 0.095 |
| R2 gated | 42.7 ± 5.4 | 29.8 ± 3.1 | 0.718 ± 0.026 | 0.483 ± 0.051 |
| R3 coverage ×2 | 42.3 ± 2.0 | 24.7 ± 5.4 | 0.690 ± 0.053 | **0.569 ± 0.033** |
| R4 VEGR + Dr. GRPO | 44.0 ± 4.0 | 31.0 ± 7.7 | 0.695 ± 0.060 | 0.441 ± 0.085 |

**Table 4.3.** Paired bootstrap differences (points, 10,000 resamples, 95% CI). Method scores are averaged over seeds 7–9 per question before differencing (Section 3.7).

| Comparison | ΔEM | ΔGrounded EM |
|---|---|---|
| R0 − B0 | +20.2 [+15.9, +24.6] | −4.8 [−8.4, −1.2] |
| R1 − B0 | +21.2 [+17.0, +25.4] | +19.7 [+15.8, +23.7] |
| R2 − B0 | +20.3 [+16.1, +24.7] | +16.8 [+13.0, +20.6] |
| R3 − B0 | +20.0 [+15.4, +24.6] | +11.7 [+7.9, +15.6] |
| R4 − B0 | +21.7 [+17.3, +26.0] | +18.0 [+14.1, +22.0] |
| **R1 − R0** | +1.0 [−1.9, +3.8] | **+24.4 [+20.9, +28.1]** |
| R2 − R1 | −0.9 [−4.0, +2.1] | −2.9 [−5.9, +0.1] |
| R3 − R1 | −1.2 [−4.2, +1.8] | −8.0 [−11.1, −4.9] |
| R4 − R1 | +0.4 [−2.6, +3.3] | −1.7 [−4.4, +1.0] |
| R0-dyn s8 − R0 s8 | +8.0 [+3.3, +12.7] | 0.0 [0.0, 0.0] |

![Figure 4.5](../figures/fig4_5_bootstrap.png)

**Figure 4.5.** The paired differences of Table 4.3. Left: EM; right: grounded EM. Above the dashed line, each method against B0 (RQ1); below it, comparisons between methods (RQ2/RQ3).

**Answer to RQ1: yes.** Every method beats B0 by about **+20 EM points**, and every interval excludes zero (Table 4.3). This holds run by run as well: compared individually with B0, the 16 trained runs gain between +16.0 (R2 s9, CI [+10.7, +21.3]) and +26.3 points (R2 s8 and R0-dyn s8, CI [+21.0, +31.7]), and no interval comes near zero.

Grounded EM tells a more mixed story. Every evidence-rewarded method (R1–R4) also improves grounded EM over B0, by +11.7 to +19.7 points. The outcome-only reward does the opposite: R0 is **4.8 points below the un-tuned model** in grounded EM (CI [−8.4, −1.2]), although it answers correctly almost twice as often.

**Where the EM gain comes from.** Most of it comes from learning the protocol:

- B0 loses 130 of 300 episodes to failures, while every trained run loses at most 22.
- Measured over completed episodes only, accuracy moves much less: from 39.4% (B0) to 44.6% (R1 seed 7).

So GRPO mostly teaches the model to produce valid, verifiable actions, and only modestly improves its reasoning on questions it could already complete. The training logs of R1 seed 7 show this directly. The invalid-rollout rate falls from 46% to 29%, 13%, 13% and 7% over the first five 32-step blocks, while average reward rises from 0.97 to 2.81.

**Validation split (supporting evidence only).** On the validation split, R1 seed 7 reaches 57.0% EM against B0's 29.3% (+27.7 points, 95% CI [+21.7, +33.7]). The validation questions appear easier than the holdout, so this number is not used for any claim. It only shows that the gain is not specific to the holdout.

## 4.4 RQ2: Outcome-only reward vs. VEGR

**Accuracy is the same.** R1 − R0 is +1.0 EM points with CI [−1.9, +3.8], and the 3-seed means are 42.6% (R0) and 43.6% (R1). **No accuracy difference can be claimed** between the two rewards.

**Grounding is very different.** R1's grounded EM is **24.4 points higher** than R0's (CI [+20.9, +28.1]), the largest between-method effect in the thesis. The difference comes from how the outcome-only policies behave: each R0 seed drifted into a shortcut, while R1 behaved consistently.

**R0 seeds 8 and 9: the citation shortcut.** Both policies learned to answer **without any citations**.

- Holdout citation precision and evidence coverage are **0.000** for both, so their grounded EM is 0.
- All 122 correct answers of seed 8 and all 135 of seed 9 come with no citation.
- This exploits the known loophole in the environment (Section 3.3): an answer with no citations is valid, and the outcome-only reward gives it full credit.
- The two seeds differ in how they search. Seed 8 searches once (1.01 searches per episode). Seed 9 still searches 2.27 times on average and often finds the right paragraphs (retrieval recall 0.860), but then does not cite them (Trace 2, Section 4.7). Searching and citing came apart.

**R0 seed 7: duplicate-search spam.** This policy did not drop citations. Instead, it always used all 3 searches (mean 3.00), and **229 of its 278** completed episodes repeat a query. By comparison, B0 repeated a query in 32 episodes and the R1 seeds in 7 to 27.

So **all three outcome-only seeds** ended in a degenerate behaviour, two in the same one.

**R1 (VEGR) is stable.** Across all three seeds:

- EM stays between 42.7% and 45.3%, the narrowest range of any method.
- Citation precision stays between 0.726 and 0.754 (sd 0.015).
- Searches per episode stay between 1.91 and 2.30, and at most 27 episodes repeat a query.

Neither shortcut appeared.

Figure 4.3 puts both dimensions together. Every trained run sits in the same accuracy band, roughly 38–49% EM. Along the grounding axis, the evidence-rewarded runs cluster at precision 0.63–0.76, while R0 seeds 8 and 9 and R0-dyn lie at 0.

![Figure 4.3](../figures/fig4_3_em_vs_precision.png)

**Figure 4.3.** Holdout EM (all episodes, failures counted as 0) against citation precision (completed episodes), one point per run. Shape and colour encode the method.

**Answer to RQ2: yes.** VEGR matches the accuracy of outcome-only reward and avoids its shortcuts on all three seeds, raising grounded EM by 24.4 points. The finding fits reward hacking theory (Skalse et al., 2022): the outcome-only reward is a proxy that does not include grounding, and on three of three seeds the policy found a way to satisfy the proxy that the intended objective would reject.

One caveat applies. R1's grounded EM (32.7%) is still about 11 points below its EM (43.6%): roughly a quarter of its correct answers cite at least one non-gold sentence. VEGR prevents the shortcut, but it does not make every correct answer fully grounded.

## 4.5 Diagnosis: why outcome-only reward is unstable

**Zero-variance groups.** GRPO learns nothing from a group in which all 8 rollouts receive the same reward (Section 3.6), and such steps are skipped. Under the outcome-only reward this happens often, because the reward is binary and many questions are either always solved or never solved within a group.

**Table 4.4.** Skipped (zero-variance) training steps out of 256.

| Method | Seed 7 | Seed 8 | Seed 9 |
|---|---|---|---|
| R0 outcome-only | 107 | 128 | 124 |
| R1 VEGR | 8 | 8 | 18 |
| R2 gated | 19 | 20 | 10 |
| R3 coverage ×2 | 11 | 15 | 10 |
| R4 Dr. GRPO | 8 | 10 | 14 |
| R0-dyn | – | 262 of 512 drawn | – |

R0 therefore learns from only about half of its questions, and each of those gives only a coarse right-or-wrong signal. The evidence-rewarded methods skip 8–20 steps, because their continuous terms (F1, coverage, precision, recall, penalties) almost always separate the 8 rollouts.

![Figure 4.1](../figures/fig4_1_skipped_steps.png)

**Figure 4.1.** Number of GRPO training steps skipped because all 8 rollouts in the group received the same reward. Standard runs have 256 steps; R0-dyn drew 512 questions and skipped 262 of them. Source: `metrics.jsonl` of each run; values in `docs/thesis/figures/figures_data.json`.

The training curve for seed 8 (Figure 4.2) shows when the R0 shortcut appeared. Over the first 64 steps, the two rewards produce similar citation precision during training (R0 0.568 and 0.499; R1 0.550 and 0.520). R0 then falls to 0.126 in steps 65–96 and stays at 0.000 for every block from step 97 on, while R1 rises to between 0.62 and 0.76. R0 seed 9 follows the same path (0.714, 0.410, 0.192, then 0.000 from step 97 on). R0 seed 7, the run that did not drop citations, stays between 0.55 and 0.73 throughout.

![Figure 4.2](../figures/fig4_2_citation_precision.png)

**Figure 4.2.** Mean citation precision of valid training rollouts per 32-step block, seed 8. A valid answer with no citations has precision 0 by definition (Section 3.4). Each point averages 128–256 valid rollouts. These are training rollouts sampled at temperature 1.0, not holdout results. Source: `metrics.jsonl`; script `scripts/make_thesis_figures.py`.

**Is the optimizer the cause?** A natural explanation is that, with so few informative updates, whatever habit emerges early in R0 training gets reinforced and locked in. The R0-dyn run tests this (Section 4.6). It removes the skipped steps' effect by drawing new questions until a group has reward variance, and it still collapses on the same schedule: training citation precision is 0.550 and 0.596 over the first 64 drawn questions, 0.250 over the next 32, and 0.000 for every later block but one (0.004). More informative updates did not stop the shortcut. The skipped steps are a symptom of the binary reward, not the cause of the shortcut.

**"Precise but incomplete" citations.** VEGR raises citation precision (about 0.73–0.75, against 0.633 for B0) but not evidence coverage, which ranges from 0.333 to 0.513 across R1 seeds. The citation pattern of R1 seed 8 shows why:

- It cites 1 sentence in 97 episodes, 2 sentences in 182, and 3 in 8.
- By sentence position, sentence 0 is cited 233 times, sentence 1 161 times, and sentence 2 57 times.

![Figure 4.7](../figures/fig4_7_cited_index.png)

**Figure 4.7.** Position of every sentence cited by R1 seed 8 in its completed holdout episodes (485 citations). 451 of them (93%) point to one of the first three sentences of a paragraph.

The policy prefers a few early sentences, which are usually correct but often not the full gold evidence. HotpotQA's gold support often includes sentences beyond index 0. Under a summed reward, citing one safe sentence already earns full precision and partial coverage, so the policy has little reason to risk adding more. This matches ALCE's observation that citation precision and citation recall are distinct, and that models tend to be precise but incomplete (Gao et al., 2023).

Coverage also varies between seeds. R1 seed 9 has the best grounded EM of the R1 runs (37.7%) but the lowest coverage (0.333), and it cites only 1.03 sentences per completed episode against 1.49 and 1.69 for seeds 7 and 8. Coverage is therefore not a fixed property of the reward.

## 4.6 RQ3: Reward variants and optimizer fixes

R2 belongs to the planned design. **R3, R4 and R0-dyn are exploratory follow-ups**, designed after the R0/R1/R2 results were seen (Section 3.8).

**R2 (gated penalties) is not better than R1.**

- On seed 8 alone, R2 looked like a clear win: 48.7% against 42.7%, a +6.0-point difference with 95% CI [+1.3, +11.0].
- It **did not replicate**. On seed 7, R2 is 1.7 points below R1; on seed 9, it reaches 38.3% against R1's 45.3%.
- Over three seeds, R2 − R1 is −0.9 EM (CI [−4.0, +2.1]) and −2.9 grounded EM (CI [−5.9, +0.1]).
- R2 has the widest seed range of any method, from 38.3% to 48.7% (sd 5.4; Figure 4.4).

This is the kind of result Henderson et al. (2018) and Agarwal et al. (2021) warn about: a single-seed comparison can look significant and still not hold across seeds.

**R3 (coverage ×2) trades precision for coverage.** Doubling the coverage weight does what it was meant to do on coverage: R3 has the highest coverage of any method (0.569 ± 0.033, against 0.440 for R1), and it cites more sentences (1.83–2.06 per completed episode against 1.03–1.69 for R1). But the extra citations are less often gold. Precision falls to 0.690, and grounded EM is **8.0 points below R1** (CI [−11.1, −4.9]), the only between-method difference among R1–R4 whose interval excludes zero. EM is unchanged (−1.2, CI [−4.2, +1.8]). Under a metric that requires every citation to be gold, R1's weights are the better choice; under one that rewards finding all the evidence, R3's are.

**R4 (Dr. GRPO advantages) is indistinguishable from R1.** Removing the standard-deviation normalisation changes EM by +0.4 (CI [−2.6, +3.3]) and grounded EM by −1.7 (CI [−4.4, +1.0]). It does not make training more stable across seeds; the opposite tendency appears (EM sd 4.0 against 1.5, grounded EM sd 7.7 against 4.8), but three seeds are too few to call this a difference. At this scale the normalisation is not the problem.

**R0-dyn (R0 + dynamic sampling, seed 8) improves accuracy but not grounding.** This was the key control (Section 3.8).

- It reaches **48.7% EM**, tied with R2 s8 for the best of any run and **+8.0 points** above R0 s8 (CI [+3.3, +12.7]). It completes all 300 episodes.
- Its grounded EM is **0.0%**, the same as R0 s8: all 146 correct answers have no citation.
- During training it collapses to uncited answers on the same schedule as R0 s8 (Section 4.5).

So the optimizer fix makes the outcome-only agent *more accurate*, but only the reward makes it *grounded*. This is a single-seed case study on the seed where R0 had collapsed; it shows that dynamic sampling is not sufficient to prevent the shortcut there, not that it never helps.

![Figure 4.4](../figures/fig4_4_em_per_seed.png)

**Figure 4.4.** Holdout EM (left) and grounded EM (right) per seed for each method. Dots are seeds 7, 8 and 9 (left to right); the horizontal bar and its label give the seed mean. The dotted line is the un-tuned baseline B0. R0-dyn has one seed.

**Answer to RQ3.** None of the variants beats plain VEGR. Gated penalties (R2) and Dr. GRPO advantages (R4) make no measurable difference. A larger coverage weight (R3) buys coverage at the price of precision and grounded EM. Dynamic sampling (R0-dyn) raises accuracy but leaves the outcome-only shortcut intact. Given seed ranges of up to about 10 EM points, **this thesis does not claim method differences in EM**; the claimed differences are in grounded EM and hold in the seed-averaged comparison.

## 4.7 Error analysis and example traces

**Failure categories.** Table 4.5 lists the episodes that did not complete.

**Table 4.5.** Failed episodes out of 300, by type.

| Run | Evidence | Protocol | Budget | Total |
|---|---|---|---|---|
| B0 | 85 | 37 | 8 | 130 |
| R0 s7 | 14 | 7 | 1 | 22 |
| R0 s8 | 0 | 1 | 0 | 1 |
| R0 s9 | 0 | 2 | 0 | 2 |
| R1 s7 | 8 | 5 | 0 | 13 |
| R1 s8 | 8 | 5 | 0 | 13 |
| R1 s9 | 4 | 3 | 0 | 7 |
| R2 s7 | 10 | 7 | 0 | 17 |
| R2 s8 | 3 | 7 | 0 | 10 |
| R2 s9 | 10 | 4 | 0 | 14 |
| R3 s7 | 7 | 4 | 0 | 11 |
| R3 s8 | 8 | 8 | 0 | 16 |
| R3 s9 | 10 | 4 | 4 | 18 |
| R4 s7 | 9 | 3 | 0 | 12 |
| R4 s8 | 4 | 3 | 0 | 7 |
| R4 s9 | 4 | 3 | 1 | 8 |
| R0-dyn s8 | 0 | 0 | 0 | 0 |

![Figure 4.6](../figures/fig4_6_failures.png)

**Figure 4.6.** Failed holdout episodes per run, stacked by failure type. Training removes most failures of every type. Evidence failures (invented or unretrievable citations) shrink from 85 to between 3 and 14 in the runs where the agent still cites.

The uncited R0 runs (s8, s9, R0-dyn) have almost no failures, but for a degenerate reason. An answer without citations can never produce an evidence failure, so the citation shortcut removes that failure type entirely.

**Example traces.** The traces below are taken verbatim from the holdout episode logs.

*Trace 1: same question, three behaviours* (question `5adbf0a255429947ff17385a`). All three policies answer correctly (EM = 1).

```
R1 s8:  SEARCH[Laleli Mosque Esma Sultan Mansion location]
        SEARCH[Laleli neighborhood]
        ANSWER[no] CITATIONS[Laleli Mosque|0; Esma Sultan Mansion|0]
R0 s8:  SEARCH[Laleli Mosque and Esma Sultan Mansion location]
        ANSWER[no]
R0 s7:  SEARCH[neighborhood of Laleli Mosque and Esma Sultan Mansion]
        SEARCH[neighborhood of Laleli Mosque and Esma Sultan Mansion]
        SEARCH[neighborhood of Laleli Mosque and Esma Sultan Mansion]
        ANSWER[no] CITATIONS[Esma Sultan Mansion|0; Laleli Mosque|0]
```

- **R1 (VEGR)** searches twice with different queries and cites one sentence from each gold paragraph. In this episode coverage and precision are both 1.0.
- **R0 seed 8** answers after one search with no citations: the citation shortcut.
- **R0 seed 7** repeats the same query three times before answering: duplicate-search spam.

*Trace 2: finding the evidence but not citing it* (`5abd94525542992ac4f382d2`, the question of Table 3.1 and Appendix A).

```
R1 s8:      SEARCH[S/S 2014 debut South Korean boy group]
            SEARCH[WINNER YG Entertainment]
            ANSWER[YG Entertainment] CITATIONS[2014 S/S|0; Winner (band)|0]
R0 s8:      SEARCH[2014 S/S debut album South Korean boy group formed by]
            ANSWER[YG Entertainment]
R0 s9:      SEARCH[2014 S/S debut album South Korean boy group formed by]
            SEARCH[WINNER YG Entertainment connection]
            SEARCH[WINNER YG Entertainment]
            ANSWER[YG Entertainment]
R0-dyn s8:  SEARCH["2014 S/S debut album South Korean boy group formed by"]
            SEARCH["2014 S/S WINNER YG Entertainment"]
            ANSWER[YG Entertainment]
```

R1 uses its second search to follow the bridge entity ("WINNER") into the second paragraph, which is the multi-hop behaviour the task is meant to require, and cites both gold sentences. R0 s8 gives the same answer with nothing to support it. R0 s9 and R0-dyn are more telling: they also follow the bridge entity and retrieve the gold paragraphs, but then answer without citing them. Their search behaviour is multi-hop; their answers are unsupported.

*Trace 3: more citations, fewer of them gold (R3 s8)* (`5ae7a8175542993210983ed8`). The answer is correct, but only one of the three cited sentences is a gold fact (precision 0.33, coverage 0.33):

```
R3 s8:  ANSWER[Pedro Rodríguez] CITATIONS[Formula One drivers from Mexico|1;
        2014 Bahrain Grand Prix|3; 2009 Singapore Grand Prix|5]
```

This is the trade-off behind R3's lower grounded EM: a heavier coverage weight encourages the policy to add citations, and some of the additions are not gold.

*Trace 4: an evidence failure (R1 s8).* The final action `ANSWER[yes] CITATIONS[Ambroise Thomas|0; Provesi|1]` (question `5ab29c24554299449642c932`) was marked invalid as an evidence failure: "citation references unknown document 'Provesi'". The model cited a document title that had not appeared in any of its search results. Even after training, invented or unretrievable citations remain the most common failure type for R1–R4 (Table 4.5).

*Trace 5: a protocol failure caused by the citation format (R1 s9)* (`5a8a3e745542996c9b8d5e70`). After three searches (two of them identical), the agent answered `ANSWER[Adventure] CITATIONS[Tunnels &amp; Trolls|0]`. The model wrote the title's "&" as the HTML entity `&amp;`, and the `;` inside the entity was read as the separator between two citations. The parser then rejected "Tunnels &amp" as a citation without an index ("each citation must have the form title|sentence_index"). Some protocol failures therefore come from the action format itself, not from the agent's reasoning. A separator that cannot appear in titles would remove this case.

## 4.8 Discussion

**Comparison with the literature.**

- **Shortcut behaviour.** The R0 shortcuts are consistent with CaRR (Zhang et al., 2026), which reports that outcome-only GRPO agents learn shortcut answers and use fewer tool calls. R0 seeds 8 and 9 show the same pattern in a much smaller setting (7B, LoRA, 256 questions, one GPU). Because HotpotQA's supporting facts are exact, the shortcut can be detected by rule, without a judge.
- **Seed dependence.** R0 seed 7 failed in the *opposite* direction: it searched more, not less. R0 seed 9 kept searching but stopped citing. So "fewer searches" is not the only way outcome-only training goes wrong; what the three seeds share is that the outcome reward never asked for evidence.
- **Optimizer fixes.** Dynamic sampling (DAPO; Yu et al., 2025) and Dr. GRPO (Liu et al., 2025) were proposed to make GRPO training more efficient and less biased. In this setting both behave as designed with respect to accuracy (R0-dyn gains 8 EM points over R0 s8), but neither changes *what* the policy is rewarded for, and so neither addresses grounding.
- **Accuracy.** The results agree with Search-R1 (Jin et al., 2025) that outcome-only RL improves accuracy. They add that the accuracy number alone hides whether the answers are grounded: R0 is as accurate as R1 and less grounded than the un-tuned model.

**What the results support:**

1. GRPO with LoRA on one GPU improves a 7B search agent by about 20 EM points over the un-tuned model. Most of the gain comes from learning to produce valid, verifiable actions.
2. Outcome-only and evidence-grounded rewards reach statistically indistinguishable accuracy, but VEGR's grounded EM is 24.4 points higher.
3. The outcome-only reward led to a degenerate behaviour on all three seeds (uncited answers on two, duplicate searches on one), while VEGR stayed consistent in grounding across three seeds.
4. The shortcut is a reward problem, not only an optimizer problem: on the seed where it occurred, dynamic sampling raised accuracy but did not prevent it.

**What the results do not support:**

- That VEGR is more *accurate* than outcome-only reward.
- That gated penalties (R2) or Dr. GRPO advantages (R4) help.
- That VEGR solves evidence coverage. It does not; raising the coverage weight (R3) increases coverage only at the cost of precision, and coverage remains the main open problem.

**Threats to validity.**

- **Few seeds, small holdout.** Three seeds and 300 holdout questions cannot resolve EM differences of a few points, and the seed-averaged bootstrap does not capture seed variance (Section 3.7). Seed standard deviations are therefore reported next to every method mean.
- **Closed corpus.** Retrieval is BM25 over each question's own 10 paragraphs, not an open corpus or the web. The search problem is much easier than in deployed agents.
- **Small training scale.** Each run uses 256 questions for one epoch with a LoRA-adapted 7B model. The results may not hold at larger scale.
- **Software change.** The PyTorch version changed between B0/R1 seed 7 (2.14.1) and the later runs (2.11.0) because of a driver constraint.
- **Environment loophole.** The environment accepts answers without citations. This was kept deliberately for comparability, but it makes the R0 citation shortcut possible. Closing it would likely change R0's behaviour, though perhaps into another shortcut (as seed 7 suggests).
- **Untuned weights.** The reward weights are untuned defaults; R3 shows that changing one weight moves the coverage–precision balance.
- **Exploratory runs.** R3, R4 and R0-dyn were designed after seeing the R0/R1/R2 results and are reported as exploratory.
- **R0-dyn has one seed and fell short of its target.** Budget allowed only a single run, on the seed where R0 collapsed (Section 3.8), and it reached 250 of the 256 intended updates. Its result is evidence about mechanism on that seed, not a method comparison.
