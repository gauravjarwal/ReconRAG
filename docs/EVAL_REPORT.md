# ReconRAG eval report: v1 baseline

Measured 2026-11-01 on the frozen golden sets. Every number comes from `evals/results.jsonl`.
Baseline run: `20261101T000929-v1_hybrid-0ce10a`. Reproduce with `make eval`.

## What this report can and can't claim

- **Scope:** v1's real parser, chunker, BM25 and RRF, frozen from commit `f5cf1b3`.
- **Different from production v1:** local `bge-small-en-v1.5` embeddings instead of Gemini, exact search
  instead of Chroma, and PHI and the LLM reranker off. So v1's reranker and its confidence gate are
  **not measured here**.
- **Synthetic data:** I generated the corpus and its labels with code. These numbers show how v1 handles
  known messiness, not how it would do on real users' data.
- **No answer quality yet:** generation evals exist but haven't run on a live model, because no budget
  was agreed. There are no correctness or faithfulness numbers.
- **Test split only,** unless stated. 172 answerable items. Intervals are 95% bootstrap.

## Headline

| Stage | Metric | v1 |
|---|---|---|
| Parsing | Files handled correctly (216 files) | 0.454 |
| Parsing | Files that crashed ingestion | 0 |
| Parsing | Required text recovered | 0.595 |
| Parsing | Table cells kept on one line with their row | 0.046 |
| Cleaning | Boilerplate left in chunks | 0.189 |
| Dedup | Duplicates recognised (15 files) | 0.000 |
| Versioning | Superseded editions recognised (10 files) | 0.000 |
| Versioning | Questions with a stale edition in the top 5 | 0.733 [0.467, 0.933], 15 items |
| Chunking | Evidence intact in one chunk | 0.700 |
| Retrieval | recall@5, hybrid | 0.500 [0.427, 0.573] |
| Retrieval | MRR, hybrid | 0.448 |

Disguised and corrupt files were mostly rejected correctly (0.889). The miss was binary garbage named
`.txt`, which v1 indexed.

## Retrieval modes

| Pipeline | recall@1 | recall@5 | recall@10 | all_found@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 only | 0.288 | 0.392 [0.323, 0.462] | 0.451 | 0.442 | 0.346 | 0.369 |
| Vector only | 0.384 | 0.442 [0.372, 0.517] | 0.485 | 0.483 | 0.413 | 0.430 |
| v1 no-reranker path (3 + 3) | 0.390 | 0.474 [0.404, 0.547] | 0.474 | 0.465 | 0.430 | 0.439 |
| Hybrid RRF (10 + 10) | 0.401 | 0.500 [0.427, 0.573] | 0.523 | 0.517 | 0.448 | 0.466 |

Paired differences in recall@5:

| Comparison | Delta | 95% CI |
|---|---|---|
| Hybrid over BM25 | +0.108 | +0.052 to +0.160 |
| Hybrid over vector | +0.058 | +0.029 to +0.090 |
| Fetching 10 + 10 over 3 + 3 | +0.026 | +0.006 to +0.052 |

Hybrid search is a real gain on this data, but a small one next to what parsing loses.

Local retrieval takes about 6 ms at p95. That excludes any network embedding call, so it says nothing
about production latency.

## Hybrid by category

| Category | Items | recall@1 | recall@5 | recall@10 | all_found@10 |
|---|---|---|---|---|---|
| docx | 11 | 1.00 | 1.00 | 1.00 | 1.00 |
| markdown | 6 | 1.00 | 1.00 | 1.00 | 1.00 |
| clause | 10 | 0.80 | 1.00 | 1.00 | 1.00 |
| encoding | 9 | 0.89 | 0.89 | 0.89 | 0.89 |
| paraphrase | 19 | 0.47 | 0.79 | 0.79 | 0.79 |
| fact | 32 | 0.47 | 0.53 | 0.53 | 0.53 |
| adversarial | 4 | 0.50 | 0.50 | 0.75 | 0.75 |
| version | 9 | 0.11 | 0.44 | 0.44 | 0.44 |
| identifier | 10 | 0.20 | 0.40 | 0.40 | 0.40 |
| table | 21 | 0.33 | 0.33 | 0.33 | 0.33 |
| multi_hop | 12 | 0.00 | 0.17 | 0.42 | 0.33 |
| html | 10 | 0.00 | 0.00 | 0.00 | 0.00 |
| ocr | 19 | 0.00 | 0.00 | 0.00 | 0.00 |

Small categories have wide uncertainty. The adversarial row is only 4 items.

## Why evidence was missed

Across both splits at k = 10, hybrid had 326 pieces of evidence to find:

| Result | Count | Share | Fix belongs in |
|---|---|---|---|
| Found | 161 | 49% | none |
| File never indexed | 102 | 31% | Parsing: HTML, EML, XLSX, CSV, images, scans |
| Indexed but not in the top 10 | 54 | 17% | Ranking, near-duplicates, versions |
| Indexed but not in any chunk | 9 | 3% | Parsing: mixed PDFs, UTF-16 text |

I read a sample of the last row and every case was a real extraction failure, not a scorer error.

## Refusal

- **v1 without the reranker never refuses:** 0 of 334 questions.
- **A threshold on top cosine similarity is weak.** Tuned on dev (threshold 0.656, F1 0.53), it reached
  precision 0.33 and recall 0.35 on test, with 6 of 17 refusal items caught and 12 false refusals.
- The fall from dev to test is why the threshold is never tuned on test. It also means cosine alone
  won't be a good enough gate.

## Other findings

- **v1 understates its costs.** `backend/pipelines/usage.py` prices Flash-Lite at $0.075/$0.30 per 1M
  tokens. Google's pricing page lists `gemini-3.1-flash-lite` at $0.25/$1.50 (read 2026-11-01), so
  v1's reported per-query costs are too low.
- **One bad file fails a whole v1 upload.** Unsupported types raise a 400 for the entire request, so a
  mixed real-world batch can't be loaded in one call.

## What I'd build next, from these numbers

1. **Parsers first.** They account for 111 of 165 misses. HTML, EML, XLSX, CSV and OCR are the biggest
   wins, and no ranking work can recover those.
2. **Versioning and dedup.** A stale edition appears in the top 5 for most version questions.
3. **Structure-aware chunking.** Table cell recall is 0.046 and 30% of evidence is split across chunks.
4. **Ranking and a better refusal signal,** measured last, once the text they rank is right.

## Limits of the harness

- `is_refusal` matches phrases. It needs checking against real model output on the first live run.
- The judge is unvalidated. Seeded-error calibration and your 50 labels are both still to do.
- The human queue uses seeded answers, not real system output. Real answers should replace it after
  the first live run.
- The CI workflow has been checked locally, but has never run on GitHub.
