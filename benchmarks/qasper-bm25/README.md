# Frozen QASPER BM25 baseline

Measured 2026-10-02 with the repository's document-scoped retrieval evaluator.
This evaluates English structured paragraphs, not PDF extraction or generated answers.

| Subset | Questions | Papers | Macro evidence recall@5 | Macro evidence recall@10 |
| --- | ---: | ---: | ---: | ---: |
| Train development | 50 | 23 | 35.70% | 57.78% |
| Validation | 200 | 68 | 48.58% | 64.13% |

The fixed seed is 42. Selection happens before scoring, by shuffled sorted paper
IDs followed by sorted question IDs. No parameter tuning was performed for this
baseline. The 50 train questions are the development subset; validation metrics
are now published, so this is a frozen comparison subset rather than a claim of
future unseen performance. Questions within the same paper are correlated.

Recall is the fraction of distinct gold paragraphs present in the first k
retrieved paragraphs, averaged over questions. Gold is the union of complete,
uniquely aligned text-only annotations. This is not the official QASPER Evidence
F1 metric. Zero-retrieval or runtime failures receive zero credit; there were
zero runtime failures in this run.

At k=10, validation has 105 fully covered questions, 52 partially covered
questions, and 43 with no retrieved evidence. At k=5 those counts are 78, 48,
and 74. The baseline therefore still misses substantial evidence.

The source train split contains 2,593 questions: 1,855 are eligible and 738
excluded. Validation contains 1,005: 798 eligible and 207 excluded. Figure/table
evidence, unmatched text, ambiguous duplicate paragraphs, missing evidence,
unanswerable questions, and empty papers are accounted for in `summary.json`.
An invalid annotation is excluded in full; a question can remain eligible via
another valid annotation. Issue counts are annotation-level and can overlap;
question exclusions use one documented primary reason. Unselected eligible
questions are also counted. These results do not apply to excluded categories.

`results.jsonl` contains all 250 question identifiers, gold/retrieved paragraph
identifiers, scores, timings, and recall values. It contains no question text,
answer text, or source paragraphs. Paragraph IDs use zero-based section and
paragraph coordinates. `summary.json` records source checksums, code/version
provenance, selection hash, denominator counts, and resource timings.

Query p50/p95 cover tokenization, scoring, and result construction only.
Index building and snapshot loading are separate; CLI startup and output are
excluded. Do not compare these query-only timings to end-to-end LLM latency.
The original run built and loaded 91 document indexes for 250 queries.

Reproduce from a source checkout:

```bash
uv sync --locked --group dev --group eval
uv run --locked --group eval page-cite-rag eval download-qasper
uv run --locked --group eval page-cite-rag eval qasper
# If that run already exists:
uv run --locked --group eval page-cite-rag eval qasper --resume
```

The run ID hashes configuration, code hashes, and selection. It can differ
across dependency/Python versions; the reported
selection hash and policy allow checking whether the same questions were used.

Data attribution: Dasigi et al. (2021), *A Dataset of Information-Seeking Questions
and Answers Anchored in Research Papers*.
[AllenAI QASPER](https://huggingface.co/datasets/allenai/qasper),
revision `16ac8fdf81e7e3ad213ead4b60fdcc9dc8ca40e9`,
dataset license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes: whitespace normalization for evidence alignment, deterministic subset
selection, and conversion of textual evidence to paragraph identifiers.
This publication does not redistribute source PDFs or the full corpus.
