# 2026-10-13: CPU re-analysis for the peer-review issues

These are answers to the simulated IEEE review (summary: GitHub issue #25). Everything here is computed from files already in `results/`. There were no new GPU runs and no cost.

- Script: `scripts/peer_review_reanalysis.py`. Run it with `python -I -X utf8 scripts/peer_review_reanalysis.py results`; it takes seconds.
- Full output: `results/2026-10-13-peer-review-reanalysis/output.md`.

## 1. Seed-aware confidence intervals (issue #4)
**Method:** a hierarchical paired bootstrap with 10k resamples. Each resample draws 3 seeds with replacement per method and 300 questions with replacement, with the questions shared across methods. The earlier CIs resampled only questions after averaging seeds.

| Comparison | ΔEM | ΔGrounded EM | ΔJoint-EM |
|---|---|---|---|
| R0 − B0 | +20.2 [+15.1, +25.6] | −4.8 [−16.0, +10.7] | −2.8 [−8.0, +3.4] |
| R1 − B0 | +21.2 [+16.4, +26.0] | +19.7 [+13.3, +26.1] | +1.1 [−5.1, +6.9] |
| R2 − B0 | +20.3 [+13.8, +27.4] | +16.8 [+11.7, +21.9] | +4.0 [+0.7, +7.2] |
| R3 − B0 | +20.0 [+14.8, +25.2] | +11.7 [+5.0, +18.2] | +7.4 [+3.6, +11.6] |
| R4 − B0 | +21.7 [+15.7, +27.6] | +18.0 [+10.0, +26.7] | +1.4 [−4.0, +7.3] |
| **R1 − R0** | +1.0 [−3.6, +5.4] | **+24.4 [+8.8, +37.3]** | +3.9 [−4.0, +11.4] |
| R2 − R1 | −0.9 [−7.1, +5.8] | −2.9 [−9.8, +3.6] | +2.9 [−2.9, +9.3] |
| R3 − R1 | −1.2 [−5.7, +3.3] | −8.0 [−16.1, −0.2] | +6.3 [+0.2, +13.1] |
| R4 − R1 | +0.4 [−5.2, +5.7] | −1.7 [−10.9, +7.7] | +0.3 [−7.0, +8.1] |

R0 stopped citing on **2 of 3 seeds**. The Clopper–Pearson 95% CI for that rate is [0.09, 0.99], so with 3 seeds the collapse frequency is essentially unknown.

**Reading:**
- The main result survives. R1 − R0 grounded EM is still clearly positive, but the honest interval is [+8.8, +37.3], not [+20.9, +28.1].
- R0 − B0 grounded EM is no longer significant.
- R3 − R1 only just excludes 0.

## 2. HotpotQA supporting-fact and joint metrics (issue #6)
- **Sup-EM:** the cited set equals the gold set (precision 1 and coverage 1).
- **Joint-EM:** EM and Sup-EM both hold.
- **Joint-F1:** follows HotpotQA, with P = answer F1 × citation precision and R = answer F1 × coverage.

3-seed means ± sd:

| Method | EM | Grounded EM | Lenient gEM | Sup-EM | Joint-EM | Joint-F1 |
|---|---|---|---|---|---|---|
| B0 | 22.3 | 13.0 | 20.3 | 11.3 | 6.0 | 17.5 |
| R0 | 42.6 ± 2.2 | 8.2 ± 14.2 | 12.9 ± 22.3 | 5.0 ± 8.7 | 3.2 ± 5.6 | 10.5 ± 18.1 |
| R1 | 43.6 ± 1.5 | 32.7 ± 4.8 | 39.9 ± 1.6 | 11.9 ± 10.1 | 7.1 ± 5.8 | 33.7 ± 3.9 |
| R2 | 42.7 ± 5.4 | 29.8 ± 3.1 | 40.0 ± 4.5 | 15.9 ± 4.8 | 10.0 ± 1.9 | 35.2 ± 2.3 |
| R3 | 42.3 ± 2.0 | 24.7 ± 5.4 | 40.1 ± 2.0 | **24.3 ± 4.2** | **13.4 ± 2.2** | **38.0 ± 2.2** |
| R4 | 44.0 ± 4.0 | 31.0 ± 7.7 | 39.7 ± 3.3 | 12.4 ± 8.7 | 7.4 ± 5.3 | 33.7 ± 4.7 |

*Lenient gEM: a correct answer with at least one gold citation.*

**Reading:**
- On HotpotQA's own joint metric, the ranking changes. **R3 (coverage ×2) is best**, Joint-EM 13.4 vs. R1 7.1, with R3 − R1 = +6.3 [+0.2, +13.1]. So R3 is not simply "worse": it trades precision for completeness, and which method wins depends on the metric.
- R1 s9 has Sup-EM 0.7, because it almost always cites exactly one sentence.
- R1 still beats R0 on Joint-F1 (33.7 vs. 10.5), but its Joint-EM advantage over R0 is not significant (+3.9 [−4.0, +11.4]).
- Thesis framing: VEGR reliably makes the agent cite *correct* sentences, but it does not make it cite *all* of them. Grounded EM is a precision-based measure.

## 3. Protocol learning, citation rate and selection bias (issues #9, #16)
| Method | EM | EM on completed only | Citation rate | Precision when cited |
|---|---|---|---|---|
| B0 | 22.3 | 39.4 | 98.8% | 0.640 |
| R0 | 42.6 | 43.8 | s7 99.6%, s8 0%, s9 0% | 0.623 (s7) |
| R1 | 43.6 | 45.2 | 99.9% | 0.744 |
| R2 | 42.7 | 44.7 | 99.5% | 0.721 |
| R3 | 42.3 | 44.6 | 99.9% | 0.691 |
| R4 | 44.0 | 45.3 | 99.8% | 0.697 |

**Reading:**
- On completed episodes, EM moves only from 39.4 to about 45. Most of RQ1's +20 points comes from completing the protocol.
- On the **136 questions that every run completed**, the precision ordering is unchanged: B0 0.627, R0 s7 0.646, R1 0.750–0.790. So the precision gain is not a selection artifact.

## 4. Why R0 drops citations: the training rollouts (issue #2)
Training rollouts for R0 seed 8, in 32-step blocks of 256 rollouts each:

| Steps | invalid: evidence (reward −1) | valid, cited | valid, uncited | mean reward: valid + cited | mean reward: valid, uncited |
|---|---|---|---|---|---|
| 1–32 | 90 | 86 | 42 | 0.50 | 0.36 |
| 33–64 | 79 | 94 | 75 | 0.48 | 0.20 |
| 65–96 | 17 | 34 | 204 | 0.53 | 0.50 |
| 97–256 | 0 | 0 | ~255 per block | – | 0.40–0.52 |

R0 seed 9 shows the same pattern: 48 and 45 evidence failures in the first two blocks, and no cited rollouts from step 97 on.

**Reading:** this refines the reviewers' claim that "citations have no upside under R0".
- A *valid* cited rollout actually earned **more** reward than an uncited one (0.50 vs. 0.36), because cited answers were more often correct.
- But about half of all attempts to cite were invalid (90 of 176 in the first block), and each invalid one costs −1.
- So the expected reward of *trying* to cite is about (86 × 0.50 − 90) / 176 ≈ **−0.27**, against +0.36 for not citing.
- The policy drops citations because invented or unretrievable citations are punished and nothing in R0 rewards getting them right. The −1 validity penalty is the driver, not EM alone.
- VEGR changes the trade: a valid cited rollout earns about 3.4 vs. about 1.3 uncited (R1 s8), which outweighs the −1 risk.

## 5. Parser collisions with '&' (issue #24)
This counts holdout protocol failures whose final output contains `&`. It is an upper bound on separator collisions from titles like "Simon & Garfunkel".

| | protocol failures | containing '&' |
|---|---|---|
| R1 (s7/s8/s9) | 5 / 5 / 3 | 3 / 3 / 2 |
| R2 | 7 / 7 / 4 | 4 / 3 / 2 |
| R3 | 4 / 8 / 4 | 4 / 5 / 4 |
| R4 | 3 / 3 / 3 | 2 / 2 / 3 |
| R0 (s7/s8/s9) | 7 / 1 / 2 | 3 / 0 / 0 |
| B0 | 37 | 2 |

**Reading:** for trained runs, about 2–5 episodes per run (≤ 1.7 EM points) may be lost to this parser limitation. The effect is similar across R1–R4, so it does not change any comparison. It slightly penalises the methods that cite more (R3).
