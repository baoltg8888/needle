# Thesis Outline — Needle

Title: **Rewarding the Evidence, Not Just the Answer: Verifiable Evidence-Grounded Rewards for Training Search Agents with GRPO**

Formal alternative: *Evidence-Grounded Reward Design for Multi-Hop Question Answering Agents Trained with Group Relative Policy Optimization*

Method name: **VEGR (Verifiable Evidence-Grounded Reward)**

Standard 5-chapter structure; adapt chapter names and numbering to the university template.

---

## Framing: existing work → gaps → proposal → evaluation

### Existing work
- **Search-R1** (COLM 2025): RL-trained search agents with answer-only (EM) reward.
- **AutoRefine** (NeurIPS 2025): adds retrieval-specific rewards to answer rewards under GRPO.
- **CaRR / C-GRPO** (2026 preprint): citation-aware rubric rewards scored by an LLM judge, at large scale.
- **DeepSeek-R1** (Nature 2025): GRPO with verifiable rewards plus a format reward.

### Gaps
| # | Gap | Evidence it is real |
|---|---|---|
| G1 | **Correct ≠ grounded.** Answer-only RL can be right without showing evidence (reward hacking). | Skalse et al. NeurIPS 2022; CaRR; our R0 runs (seed 8: citation precision 0.00) |
| G2 | **Citation rewards are expensive.** Existing citation rewards need an LLM judge (cost, latency, judge bias). | CaRR; rubric reward-hacking literature |
| G3 | **Compute barrier.** Search-agent RL is reported at multi-GPU scale. | Search-R1, AutoRefine setups |
| G4 | **Weak evidence on what matters.** Most papers report single runs, so reward vs. optimizer effects are not separated. | Agarwal et al. NeurIPS 2021; Henderson et al. AAAI 2018 |

### Proposal: VEGR
A judge-free, sentence-level reward for GRPO search agents. Every term is checked by rules against HotpotQA's gold supporting facts:
- answer correctness: exact match + F1
- evidence coverage: fraction of gold supporting sentences cited
- citation precision: fraction of citations that are gold sentences
- retrieval recall: gold documents exposed by the agent's searches
- efficiency: search cost and duplicate-query penalty
- protocol enforcement: −1 for invented citations, malformed actions or exhausted budget

Variants studied: R2 (penalties gated on correct answers), R3 (coverage weight ×2), R4 (VEGR + Dr. GRPO advantages).

### How each gap is bridged
| Gap | How the thesis bridges it | Experiment |
|---|---|---|
| G1 | Show the shortcut happens with answer-only reward, and that VEGR prevents it at equal accuracy | R0 vs. R1 (3 seeds) |
| G2 | Rule-checked rewards from gold labels: free, instant, reproducible | All VEGR runs; no judge in the loop |
| G3 | Whole pipeline on 1 L40S, LoRA, 256 questions, ≈$2.25 per run | All runs; cost table |
| G4 | 3 seeds and paired bootstrap CIs; separate reward fixes (R1–R3) from optimizer fixes (Dr. GRPO, DAPO dynamic sampling) | R0 vs. R0 + dynamic sampling; R1 vs. R2/R3/R4 |

### Differences from existing work
| | Search-R1 | AutoRefine | CaRR / C-GRPO | **Needle (VEGR)** |
|---|---|---|---|---|
| Reward | Answer (EM) | Answer + retrieval | Answer + citation rubric | Answer + sentence-level evidence + retrieval + search cost |
| Citation checking | none | none | LLM judge | **rules vs. gold labels** |
| Citation granularity | – | – | pages / claims | **sentence (title + index)** |
| Hallucinated citations | not handled | not handled | judged | **hard −1 (invalid)** |
| Compute | multi-GPU | multi-GPU | large scale | **1 GPU, LoRA, ≈$2.25 per run** |
| Statistics | mostly single run | mostly single run | single run | **3 seeds + bootstrap CIs** |

### What we do not claim
- No new RL algorithm (GRPO, Dr. GRPO and DAPO are existing methods).
- No state-of-the-art accuracy and no open-web search: the agent searches the 10 HotpotQA distractor paragraphs per question with BM25.
- The individual reward terms are not new. The contribution is the judge-free, sentence-level, protocol-enforcing combination, plus controlled evidence of why it is needed and which parts matter.

---

## Abstract
- Problem: RL-trained search agents rewarded only for correct answers learn to answer without showing evidence; existing citation rewards rely on costly LLM judges.
- Approach: VEGR, a rule-checked evidence-grounded reward; an agent that issues `SEARCH[...]` / `ANSWER[...] CITATIONS[...]` actions over HotpotQA paragraphs (BM25), trained with GRPO + LoRA on Qwen2.5-7B-Instruct on one GPU.
- Results (fill in final numbers): all rewards +18–26 EM over the un-tuned model; answer-only reward matches VEGR on EM but collapses citations (precision 0.00 on one seed) or spams duplicate searches; VEGR keeps citation precision ≈0.75 with stable accuracy.
- Takeaway: to train grounded search agents, reward the evidence, not just the answer, and this can be done cheaply with verifiable rewards.

## Chapter 1 — Introduction
1.1 Background: LLMs, hallucination, retrieval-augmented generation (RAG), agentic search, RL for LLMs.
1.2 Problem statement: answer-only rewards let agents be correct without being grounded (G1); citation rewards are expensive (G2); evidence comes from large compute (G3) and single runs (G4).
1.3 Research questions
- RQ1: Does GRPO training improve a 7B search agent on HotpotQA over the un-tuned model?
- RQ2: Does a verifiable evidence-grounded reward (VEGR) prevent the shortcut behaviour of answer-only reward, without losing accuracy?
- RQ3: How do reward variants (gated penalties, coverage weight) and optimizer fixes (Dr. GRPO, dynamic sampling) affect grounding and search behaviour?
1.4 Objectives
1.5 Contributions
1. VEGR: a judge-free, sentence-level, protocol-enforcing reward for search agents.
2. Evidence that answer-only GRPO produces seed-dependent shortcuts (no citations / duplicate-search spam), with a training-log diagnosis (zero-variance groups).
3. A controlled comparison across 3 seeds with bootstrap CIs, separating reward design from optimizer fixes.
4. An open, reproducible single-GPU pipeline (≈$2.25 per run).
1.6 Scope and limitations: HotpotQA distractor setting, BM25 over per-question paragraphs (not open web), 7B LoRA, 256 training questions per run, 300-question holdout, ≈$45 total compute.
1.7 Thesis structure

## Chapter 2 — Literature Review
2.1 Large language models and instruction tuning (Qwen2.5).
2.2 Retrieval-augmented generation; BM25.
2.3 Multi-hop QA and HotpotQA (supporting facts, distractor setting, EM/F1); ALCE citation metrics.
2.4 LLM agents and tool use (ReAct-style action loops).
2.5 RL for LLMs: policy gradient, PPO, GRPO (DeepSeekMath, DeepSeek-R1), Dr. GRPO, DAPO; LoRA/PEFT.
2.6 RL for search agents: Search-R1, AutoRefine, "How to Train Your Deep Research Agent".
2.7 Grounding and evidence rewards: Fine-Grained RLHF, CaRR / C-GRPO, STAMP.
2.8 Reward hacking and RL evaluation methodology: Skalse 2022, Gao 2023, Agarwal 2021, Henderson 2018.
2.9 Research gaps G1–G4 and positioning (table above).

Sources: `docs/research/2026-10-03-related-work-and-directions.md`, `2026-10-11-results-diagnosis-and-improvements.md`, `2026-10-11-peer-reviewed-grounding-and-budget-plan.md`.

## Chapter 3 — Methodology
3.1 Research design overview (diagram: question → agent → SEARCH/ANSWER → environment → VEGR → GRPO update).
3.2 Dataset and splits: HotpotQA train/dev, deterministic hash split, validation split, frozen 300-question holdout (`prepare_splits.py`).
3.3 Environment: `SearchEnvironment`, BM25 top-k=3, max 3 searches, action grammar, citation validation, failure types (invalid, exhausted).
3.4 **Proposed reward (VEGR)**: formula and weights from `RewardConfig`; protocol penalty −1; variants R2 (gated), R3 (coverage ×2). Baseline reward R0 (EM only).
3.5 Policy model: Qwen2.5-7B-Instruct, chat prompt (`build_prompt`), LoRA (r=16, α=32, all projection layers).
3.6 Training algorithm: multi-turn GRPO, G=8, group-normalized advantage, sequence-mean log-prob over action tokens, clipped objective (ε=0.2), no KL; options for Dr. GRPO advantages and DAPO dynamic sampling; pseudocode.
3.7 Evaluation protocol: greedy decoding, no retries, holdout only; metrics: EM, F1, evidence coverage, citation precision, retrieval recall, searches/episode, invalid rate, grounded EM (correct and fully cited); paired bootstrap 95% CIs; 3 seeds.
3.8 Experiment matrix: B0, R0, R1 (VEGR), R2, R3, R4, R0 + dynamic sampling.
3.9 Implementation and compute: Python/uv, PyTorch, Transformers, PEFT; RunPod L40S 48 GB; reproducibility (seeds, manifests, run metadata, logs/ and results/).

## Chapter 4 — Results and Discussion
4.1 Experimental setup table (hyperparameters, data sizes, GPU hours, cost).
4.2 Baseline B0: 22.3% EM; failure breakdown (invented citations dominate).
4.3 RQ1: GRPO vs. B0 (all rewards +18–26 EM, bootstrap CIs).
4.4 RQ2: R0 vs. VEGR: same EM, very different grounding; the two R0 shortcut modes; grounded EM.
4.5 Diagnosis: zero-variance groups (R0 107–128/256 skipped vs. VEGR ~8), first-sentence citation bias, precision-vs-coverage trade-off.
4.6 RQ3: R2/R3/R4 and dynamic sampling; does an optimizer fix alone stop the shortcut?
4.7 Error analysis: failure categories (`analyze_eval.py`), 5–10 qualitative traces (grounded, shortcut, duplicate search, format failure).
4.8 Discussion: comparison to literature; threats to validity (3 seeds, 300-question holdout, closed BM25 corpus, torch version change between runs, no-citation answers accepted by the environment).

## Chapter 5 — Conclusion
5.1 Summary per RQ.
5.2 Contributions (1–4 above).
5.3 Limitations.
5.4 Future work: step-level credit (STAMP, information gain), adaptive multi-reward weighting, open-web retrieval, more datasets (2Wiki, MuSiQue), larger training sets.

## References
HotpotQA, Search-R1, AutoRefine, DeepSeek-R1, DeepSeekMath/GRPO, PPO, Dr. GRPO, DAPO, Fine-Grained RLHF, ALCE, CaRR, STAMP, Skalse 2022, Gao 2023, Agarwal 2021, Henderson 2018, LoRA, BM25, ReAct, Qwen2.5. Verify every venue before submission.

## Appendices
- A. Prompt template and action grammar.
- B. Full hyperparameters (`configs/training/*.json`).
- C. Example trajectories.
- D. Reproduction commands (`docs/runbooks/runpod.md`).

---

## Writing checklist
- [ ] Ch2 and Ch3 can be written now.
- [ ] Ch4 after the fixes batch finishes (~12 Oct midday).
- [ ] Ch1, Ch5, abstract last.
