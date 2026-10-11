# Appendix A — Full Agent Prompt

This appendix gives the complete prompt the agent receives, for one real holdout question. The text was produced by `build_prompt` (`src/needle/inference/runner.py`), the same function used in training and evaluation. It is passed to Qwen2.5-7B-Instruct as a single user message through the model's chat template.

## How the example was reproduced

- **Question:** `5abd94525542992ac4f382d2` from the frozen holdout (`configs/evaluation/holdout_v1_ids.json`). The row was taken from the HotpotQA distractor validation split on Hugging Face (`hotpotqa/hotpot_qa`, `distractor/validation-00000-of-00001.parquet`) and converted to the native HotpotQA format.
- **History:** a `SearchEnvironment` (top-k = 3, budget = 3) replayed the two searches that R1 seed 8 made in its holdout episode (`results/2026-10-10-batch/r1_full_seed8/episodes.jsonl`). `build_prompt` was then called at the point where the agent chose its final action.
- **Gold supporting facts:** `2014 S/S|0` and `Winner (band)|0`.
- **Agent's action at this point:** `ANSWER[YG Entertainment] CITATIONS[2014 S/S|0; Winner (band)|0]`. The answer is correct and both gold sentences are cited (Table 3.1, episode A).

## Prompt text

The prompt is reproduced verbatim. The leading spaces on some sentences are present in the HotpotQA source text.

```text
You are a careful evidence-seeking QA agent.
Return exactly one action: SEARCH[query] or ANSWER[answer] CITATIONS[title|index; ...].
SEARCH[query] is search-only: never append CITATIONS[...] to a SEARCH action.
ANSWER[...] is terminal; when the search budget reaches 0, answer immediately.
Example: ANSWER[United States] CITATIONS[Ursula K. Le Guin|1; The Left Hand of Darkness|1].
Use one title|index citation per entry; separate entries with semicolons, never comma-separated indices.
Do not add a period, explanation, or any text after the final closing bracket.
Copy citation titles exactly as shown in the search results, including punctuation and accents.
Never invent a document title; use CITATIONS[] when no citation can be validated.
Use only the supplied search results for citations.
Question: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
Remaining search budget: 1
Search results so far:
Search 1: S/S 2014 debut South Korean boy group
[2014 S/S]
0: 2014 S/S is the debut album of South Korean group WINNER.
1:  It was released on August 12, 2014 by the group's record label, YG Entertainment.
2:  The members were credited for writing the lyrics and composing the majority of the album's songs.
[History (band)]
0: History (Korean: 히스토리 ) was a South Korean boy group formed by LOEN Entertainment in 2013.
1:  They debuted on April 26, 2013 with "Dreamer", featuring the narration of their labelmate IU.
2:  They were LOEN Entertainment's first boy group.
3:  They officially disbanded on May 12, 2017.
[SF9 (band)]
0: SF9 (Korean: 에스에프나인 ; shortened from Sensational Feeling 9) is a South Korean boy group formed by FNC Entertainment.
1:  SF9 is the company's first dance boy group to ever debut.
2:  SF9 debuted on October 5, 2016 with the release of their first single album "Feeling Sensation".
Search 2: WINNER YG Entertainment
[Winner (band)]
0: Winner (Hangul: 위너), often stylized as WINNER, is a South Korean boy group formed in 2013 by YG Entertainment and debuted in 2014.
1:  It currently consists of four members, Jinwoo, Seunghoon, Mino and Seungyoon.
2:  Originally a five-piece group with Taehyun, who later departed from the group in November 2016.
[2014 S/S]
0: 2014 S/S is the debut album of South Korean group WINNER.
1:  It was released on August 12, 2014 by the group's record label, YG Entertainment.
2:  The members were credited for writing the lyrics and composing the majority of the album's songs.
[History (band)]
0: History (Korean: 히스토리 ) was a South Korean boy group formed by LOEN Entertainment in 2013.
1:  They debuted on April 26, 2013 with "Dreamer", featuring the narration of their labelmate IU.
2:  They were LOEN Entertainment's first boy group.
3:  They officially disbanded on May 12, 2017.
```

## Notes on the prompt

- **First turn.** Before any search, the history section reads `(none)` and the remaining budget is 3.
- **Repeated documents.** Each search lists its full top-3 results, even when a document was already returned by an earlier search ("2014 S/S" and "History (band)" appear twice above). The prompt grows with every search, which is one reason the budget is limited to 3.
- **Why the format rules exist.** The rules about citation formatting (one `title|index` per entry, exact titles, no trailing text) target the protocol failures observed in the un-tuned model, which lost 37 of 300 holdout episodes to malformed actions (Section 4.2).
- **No-citation loophole.** The line "use CITATIONS[] when no citation can be validated" is what makes an empty citation list a valid action (the loophole discussed in Section 3.3).
