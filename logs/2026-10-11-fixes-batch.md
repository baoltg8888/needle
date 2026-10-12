# 2026-10-11 — Seed 9 + literature-backed fixes batch

Plan: [peer-reviewed grounding and budget tiers](../docs/research/2026-10-11-peer-reviewed-grounding-and-budget-plan.md), **Medium tier** (+$15, balance $28.52 at start).

Same pod `9eh9hp0rd9pu3r` (L40S Secure, $1.10/h, torch 2.11.0+cu128). Each run trains on 256 questions with G=8 and lr 2e-5, then evaluates on the 300-question holdout (greedy).

## Queue
| # | Config | Change vs R1 | Backed by |
|---|---|---|---|
| a | `r1_full_seed9`, `r2_gated_seed9` | 3rd seed for R1/R2 (`/workspace/batch2.log`) | Agarwal 2021 |
| 1 | `r3_coverage2_seed7/8/9` | evidence_coverage_weight 1 → 2 | Fine-Grained RLHF (NeurIPS 2023), ALCE (EMNLP 2023) |
| 2 | `r4_drgrpo_seed7/8/9` | advantage = r − mean (no std division) | Dr. GRPO (COLM 2025) |
| 3 | `r0_outcome_seed9` | 3rd seed for R0 | Agarwal 2021 |
| 4 | `r0_dynsample_seed8` | R0 + dynamic sampling: zero-variance groups don't count; up to 512 questions until 256 updates | DAPO (NeurIPS 2025) |

The queue (`/workspace/queue3.sh`, log `/workspace/batch3.log`) waits for seed 9 to finish, pulls commit `1ebe62c`, runs items 1–4 in the order r3 s7, r4 s7, r3 s8, r4 s8, r0 s9, r3 s9, r4 s9, r0-dyn s8, then **stops the pod automatically** via `runpodctl`, so there is no idle cost.

Estimate: seed 9 done ~18:20 WIB 2026-10-11; queue ≈ 18 h → done ~12:30 WIB 2026-10-12; total ≈ $23.

The no-citation loophole is **not** changed in the environment, so new runs stay comparable with R0–R2. Instead, evaluation adds a "grounded EM" metric (EM = 1, at least one citation, and every citation a gold fact).

### Why R0-dyn has one seed (recorded 2026-10-11)
The plan's Medium tier listed "F5 on R0 × 2 seeds". When the queue was built, this was cut to **one seed**:
- **Cost.** Dynamic sampling doesn't count zero-variance groups. R0 s8 skipped 128/256 steps, so reaching 256 updates is expected to need up to ~512 questions, about 2× a normal run (~$4.50).
- **Priority.** The +$15 had to cover 3 seeds for every main method (R0–R4) first.
- **Seed choice.** Seed 8 was chosen on purpose, because R0 collapsed there (citation precision 0.00, 128/256 skipped). The run tests whether an optimizer fix alone stops the shortcut where it actually happened.
- **Framing in the thesis.** It is a single-seed diagnostic case study, not a method comparison, and it is listed under limitations.

## Results

The queue finished at 14:26 WIB on 2026-10-12. All 10 runs (2 from batch2, 8 from queue3) exited 0, and the pod stopped itself. When restarting the pod to download, its GPU host was full, so it was started CPU-only and then stopped again.

Raw files are in `results/2026-10-11-fixes/`:
- `results/<run>/` holds the holdout `episodes.jsonl` and `summary.json`
- `runs/<run>/` holds the training `metrics.jsonl` and `run_metadata.json`
- the queue scripts and logs are `batch*.sh`, `queue3.sh` and `batch*.log`

R1 s7 is `results/2026-10-10-grpo-run2/`. R0 s7/s8, R1 s8 and R2 s7/s8 are in `results/2026-10-10-batch/`. The LoRA adapters (~470 MB per run) were not downloaded.

**Grounded EM** = EM 1, at least one citation, and citation precision 1. "Updates/skipped" count the training steps; a skipped step is a zero-variance group.

### Per run (300-question holdout, failures count as 0)
| Run | EM | grounded EM | completed | F1 | evid. cov | cit. prec | retr. recall | searches | dup-query eps | correct with no citation | updates | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 22.3% | 13.0% | 170 | 0.501 | 0.472 | 0.633 | 0.829 | 1.92 | 32 | 0 | – | – |
| R0 s7 | 42.0% | 24.7% | 278 | 0.576 | 0.438 | 0.621 | 0.829 | 3.00 | 229 | 0 | 149 | 107 |
| R0 s8 | 40.7% | 0.0% | 299 | 0.530 | 0.000 | 0.000 | 0.710 | 1.01 | 0 | 122 | 128 | 128 |
| R0 s9 | 45.0% | 0.0% | 298 | 0.577 | 0.000 | 0.000 | 0.860 | 2.27 | 55 | 135 | 132 | 124 |
| R1 s7 | 42.7% | 32.3% | 287 | 0.593 | 0.473 | 0.751 | 0.851 | 2.30 | 14 | 0 | 248 | 8 |
| R1 s8 | 42.7% | 28.0% | 287 | 0.574 | 0.513 | 0.726 | 0.836 | 2.15 | 7 | 0 | 248 | 8 |
| R1 s9 | 45.3% | 37.7% | 293 | 0.598 | 0.333 | 0.754 | 0.837 | 1.91 | 27 | 0 | 238 | 18 |
| R2 s7 | 41.0% | 30.7% | 283 | 0.577 | 0.539 | 0.748 | 0.891 | 2.21 | 34 | 0 | 237 | 19 |
| R2 s8 | 48.7% | 32.3% | 290 | 0.625 | 0.440 | 0.703 | 0.879 | 2.00 | 4 | 0 | 236 | 20 |
| R2 s9 | 38.3% | 26.3% | 286 | 0.552 | 0.471 | 0.702 | 0.855 | 2.16 | 36 | 0 | 246 | 10 |
| R3 s7 | 41.3% | 26.3% | 289 | 0.570 | 0.607 | 0.701 | 0.854 | 2.01 | 12 | 0 | 245 | 11 |
| R3 s8 | 41.0% | 18.7% | 284 | 0.574 | 0.545 | 0.633 | 0.820 | 1.61 | 5 | 0 | 241 | 15 |
| R3 s9 | 44.7% | 29.0% | 282 | 0.602 | 0.557 | 0.738 | 0.882 | 2.35 | 34 | 0 | 246 | 10 |
| R4 s7 | 39.3% | 23.7% | 288 | 0.525 | 0.469 | 0.636 | 0.796 | 1.89 | 35 | 0 | 248 | 8 |
| R4 s8 | 46.3% | 30.3% | 293 | 0.620 | 0.509 | 0.693 | 0.865 | 2.30 | 44 | 0 | 246 | 10 |
| R4 s9 | 46.3% | 39.0% | 292 | 0.609 | 0.346 | 0.757 | 0.867 | 2.41 | 25 | 0 | 242 | 14 |
| R0-dyn s8 | 48.7% | 0.0% | 300 | 0.632 | 0.000 | 0.000 | 0.875 | 2.29 | 15 | 146 | 250 | 262 |

R0-dyn used all 512 questions and reached 250 updates, 6 short of the 256 target.

### 3-seed means (mean ± sd across seeds 7/8/9)
| Method | EM | grounded EM | cit. prec | evid. cov |
|---|---|---|---|---|
| R0 outcome-only | 42.6 ± 2.2 | 8.2 ± 14.2 | 0.207 ± 0.358 | 0.146 ± 0.253 |
| **R1 VEGR** | 43.6 ± 1.5 | **32.7 ± 4.8** | **0.744 ± 0.015** | 0.440 ± 0.095 |
| R2 gated | 42.7 ± 5.4 | 29.8 ± 3.1 | 0.718 ± 0.026 | 0.483 ± 0.051 |
| R3 coverage ×2 | 42.3 ± 2.0 | 24.7 ± 5.4 | 0.690 ± 0.053 | **0.569 ± 0.033** |
| R4 VEGR + Dr. GRPO | 44.0 ± 4.0 | 31.0 ± 7.7 | 0.695 ± 0.060 | 0.441 ± 0.085 |

### Paired bootstrap (10k resamples, 95% CI, points)
For each question, the score is averaged over the 3 seeds, and the difference is bootstrapped over the 300 questions. These CIs capture question sampling, not seed variance; use the sd above for seed variance.

| Comparison | ΔEM | ΔGrounded EM |
|---|---|---|
| R0 − B0 | +20.2 [+15.8, +24.8] | −4.8 [−8.6, −1.2] |
| R1 − B0 | +21.2 [+17.0, +25.4] | +19.7 [+15.8, +23.7] |
| R2 − B0 | +20.3 [+16.1, +24.7] | +16.8 [+13.0, +20.7] |
| R3 − B0 | +20.0 [+15.6, +24.6] | +11.7 [+7.9, +15.6] |
| R4 − B0 | +21.7 [+17.3, +26.0] | +18.0 [+14.1, +22.0] |
| **R1 − R0** | +1.0 [−1.8, +3.9] | **+24.4 [+20.9, +28.1]** |
| R2 − R1 | −0.9 [−3.8, +2.1] | −2.9 [−5.9, +0.0] |
| R3 − R1 | −1.2 [−4.2, +1.8] | −8.0 [−11.1, −4.8] |
| R4 − R1 | +0.4 [−2.4, +3.3] | −1.7 [−4.4, +1.1] |
| R0-dyn s8 − R0 s8 | +8.0 [+3.7, +12.3] | +0.0 [+0.0, +0.0] |

## Interpretation
1. **The R0 shortcut replicates.** R0 s9 drops citations completely, just like s8: precision 0.00, and 135 correct answers have no citation. So 2 of 3 outcome-only seeds stop citing, and the third (s7) spams duplicate searches instead. Without the evidence reward, the agent learns that citations are optional.
2. **The main result: VEGR keeps accuracy and adds grounding.** R1 and R0 have the same EM (+1.0, CI includes 0). But R1's grounded EM is **+24.4 points** higher (CI [+20.9, +28.1]), and its precision stays at 0.73–0.75 on every seed.
3. **Dynamic sampling (DAPO) alone does not fix the shortcut.** R0-dyn s8 reaches the best EM of any run (48.7%, +8.0 vs. R0 s8, and it completes all 300 episodes). But it still cites nothing (grounded EM 0.0%, 146 correct answers with no citation). Fixing the optimizer makes the answers better; only the reward makes them grounded. This is a single-seed case study.
4. **R3 (coverage ×2) trades precision for coverage.** Coverage rises to 0.569 (vs. 0.440), the only clear coverage gain in the batch. But precision falls and grounded EM drops by **−8.0** vs. R1 (CI excludes 0), because the agent cites more sentences, some of them wrong. Under a strict "all citations gold" metric, R1's weights are the better choice.
5. **R4 (Dr. GRPO) ≈ R1.** EM is +0.4 and grounded EM is −1.7, both with CIs that include 0. Seed variance is larger (EM sd 4.0 vs. 1.5). Removing the std normalisation is not needed at this scale.
6. **R2 is not better than R1,** as the seed-9 run already showed (−0.9 EM, −2.9 grounded EM).
7. Every trained run beats B0 on EM by about +20 points. On grounded EM, R0 is *below* B0 (−4.8).

## Cost
The RunPod balance went from $28.52 at the start of the batch to $2.05, so this batch cost about **$26.5**: 10 runs, plus the CPU-only restart (~$0.55/h) used to download the results.
