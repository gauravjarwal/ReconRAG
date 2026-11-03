# Defending the eval harness

Claude wrote this code. Answer these aloud before accepting it, then check the file named.

## What it does

1. Why is relevance judged on text, and not on chunk ids? (`evals/harness/matching.py`)
2. What does "recall@5 = 0.500 [0.427, 0.573]" mean, in one sentence?
3. Why is the comparison a paired bootstrap, and not two separate intervals?
   (`evals/harness/metrics.py`, `paired_bootstrap`)

## Why this design

4. Why freeze v1's files, when the baseline could simply import `backend/`?
   (`evals/baselines/v1_frozen/README.md`)
5. Why split dev and test by a hash of the item id, and not randomly at build time?
   (`evals/corpus/golden.py`, `split_of`)
6. Why does ingestion use a fixed tolerance in the gate, while retrieval uses a confidence interval?
   (`evals/compare.py`)
7. Why are there 24 near-identical sibling plans in the corpus? (`evals/corpus/world.py`)
8. Why does the stub never record results? (`evals/generation.py`)

## What breaks if a line is removed

9. Remove the coverage check in `contains_evidence`. Which numbers go wrong, and in which direction?
10. Compare numbers against the whole chunk instead of the matched passage. What goes wrong?
11. Let `calibrate_gate` see test items. Why would the reported precision become worthless?
12. Remove `same_golden` from the gate. How could a regression pass CI?

## Interview pressure

13. "Your data is synthetic. Why should I believe any of this?"
14. "Hybrid beat vector by 0.058. Is that real, and does it matter?"
15. "You have no faithfulness number. How do you know the answers are any good?"
16. "How do you trust your judge?"
