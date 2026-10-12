# Chapter 5 — Conclusion

> **DRAFT — not submission-ready.** Every number below is taken from Chapter 4 and has the same sources.

## 5.1 Summary

This thesis asked whether a search agent trained with reinforcement learning can be rewarded for *grounded* answers, rather than only correct ones, without an LLM judge and on a student budget. A Qwen2.5-7B-Instruct agent with LoRA was trained with GRPO on HotpotQA. It searched each question's paragraphs with BM25 and answered with sentence-level citations, which a rule could check against HotpotQA's gold supporting facts. Sixteen runs were trained on one GPU for about $49 in total: five reward and optimizer configurations on three seeds each, plus one diagnostic run. Each was evaluated on a frozen 300-question holdout.

**RQ1: Does GRPO training improve a 7B search agent over the un-tuned model?** Yes. Every method gains about +20 EM points over the un-tuned model (22.3%), and every confidence interval excludes zero. Most of the gain comes from learning the action protocol: the un-tuned model fails 130 of 300 episodes, while trained runs fail at most 22.

**RQ2: Does VEGR prevent the shortcuts of answer-only reward without losing accuracy?** Yes. The two rewards reach the same accuracy (43.6% vs. 42.6%, a difference of +1.0 points with CI [−1.9, +3.8]). VEGR's grounded EM is **24.4 points higher** (CI [+20.9, +28.1]). All three outcome-only seeds ended in a degenerate behaviour. Two stopped citing entirely, and the third repeated the same search in most episodes. As a result, the outcome-only agent is *less* grounded than the un-tuned model (−4.8 grounded EM points). VEGR kept citation precision at 0.73–0.75 on every seed.

**RQ3: How do reward variants and optimizer fixes affect grounding and search behaviour?** None improved on plain VEGR.

- Gated penalties (R2) and Dr. GRPO advantages (R4) made no measurable difference. R2's apparent single-seed gain of +6 EM points did not replicate.
- Doubling the coverage weight (R3) raised evidence coverage from 0.44 to 0.57. Precision fell, and grounded EM dropped 8.0 points.
- Dynamic sampling (R0-dyn, one seed) made the outcome-only agent more accurate (+8.0 EM points). It still cited nothing and collapsed during training on the same schedule as the plain outcome-only run.

The citation shortcut is therefore caused by what the reward asks for, not only by how GRPO uses it.

## 5.2 Contributions

1. **VEGR**, a judge-free, sentence-level evidence reward that also enforces the action protocol. Its individual terms are not new; the contribution is the combination and its controlled evaluation.
2. **Evidence that answer-only GRPO produces shortcuts** in a small, reproducible setting, detectable by rule. The thesis also diagnoses the shortcut: outcome-only runs skip about half their training steps for lack of reward variance, but restoring those updates does not prevent the shortcut.
3. **A controlled multi-seed comparison** with paired bootstrap confidence intervals and seed standard deviations. It separates reward design from optimizer fixes, and shows how a single-seed result can mislead.
4. **An open single-GPU pipeline.** One run costs about $2.25. Splits, configurations, logs, per-episode results and the figure script are released in the repository.

## 5.3 Limitations

- **Setting.** Retrieval is BM25 over each question's own 10 paragraphs, on one dataset, so the search problem is far easier than open-web search.
- **Scale.** Training uses 256 questions per run for one epoch, on a 7B model with LoRA.
- **Statistics.** Three seeds and 300 holdout questions resolve large effects, such as the grounded-EM gaps. They cannot resolve EM differences of a few points.
- **Single seed for R0-dyn.** R0-dyn has one seed and reached 250 of its 256 target updates.
- **Exploratory runs.** R3, R4 and R0-dyn were designed after seeing the first results.
- **Loophole and weights.** The environment accepts uncited answers. The reward weights were not tuned.
- **Coverage.** Coverage remains low. Even with VEGR, about a quarter of correct answers include at least one non-gold citation.

## 5.4 Future work

- **Close the no-citation loophole and fix the citation format.** Require at least one citation, and use a separator that cannot occur in titles (Trace 5, Section 4.7). Then test whether outcome-only training finds a different shortcut.
- **Reward complete evidence without losing precision.** Examples are a coverage term that counts only gold citations, or a reward based on the F1 of the citation set, instead of a larger weight on coverage alone (R3).
- **Step-level credit.** Reward individual searches by the information they add, in the direction of STAMP (Xu et al., 2026), rather than scoring only the final episode.
- **Harder retrieval and more data.** Extend to an open corpus, to other multi-hop datasets with gold evidence such as 2WikiMultiHopQA and MuSiQue, and to larger training sets.
- **More seeds for the diagnostic runs.** Run R0-dyn on all three seeds, so that the mechanism finding becomes a method comparison.
