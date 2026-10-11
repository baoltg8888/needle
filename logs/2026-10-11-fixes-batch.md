# 2026-10-11 — Seed 9 + literature-backed fixes batch (in progress)

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
_To be filled when the queue finishes._
