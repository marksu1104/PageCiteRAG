# Frozen BM25, dense and hybrid QASPER comparison

Initial unoptimized comparison on the same 50 development and 200 validation
questions as [the BM25 baseline](../qasper-bm25/README.md). The validation subset
contains 68 known papers and 375 distinct per-question gold paragraphs. These
are retrieval results, **not answer accuracy** or the official QASPER leaderboard.

| Validation method | Macro recall@5 | Macro recall@10 | Complete @10 | Partial @10 | No evidence @10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 48.58% | 64.13% | 105/200 | 52/200 | 43/200 |
| Qwen dense | 55.52% | 70.95% | 120/200 | 48/200 | 32/200 |
| BM25 + dense RRF | 57.21% | 72.55% | 121/200 | 52/200 | 27/200 |

Hybrid gains 8.42 percentage points in macro recall@10 over BM25. This does not
mean every question improves or that hybrid dominates dense:

| Validation evidence group | BM25 recall@10 / complete | Dense recall@10 / complete | Hybrid recall@10 / complete |
| --- | ---: | ---: | ---: |
| Single paragraph (119 questions) | 73.95% / 88 | 78.15% / 93 | 83.19% / 99 |
| Multiple paragraphs (81 questions) | 49.69% / 17 | 60.38% / 27 | 56.91% / 22 |
| Three or more (42-question subset) | 42.27% / 6 | 50.96% / 6 | 46.67% / 6 |

Dense is better on the multi-evidence subset; hybrid is better on single-evidence
questions. All methods completely recover evidence for only 6/42 questions in
the three-or-more group. At @10 micro recall, dense (63.20%) also exceeds hybrid
(61.87%); macro gives each question equal weight, whereas micro weights questions
by gold evidence count. This leaves substantial detail-coverage work unresolved.
No statistical significance or universal superiority is claimed; questions
within a paper are correlated and the evidence groups overlap.

## Protocol and costs

- QASPER revision: `16ac8fdf81e7e3ad213ead4b60fdcc9dc8ca40e9`; seed 42.
- Selection checksum: `da54def8692ff302abed07851311bba123de5597c38c38dd1687da9696c9df0f`.
- All methods use identical source paragraphs, selection, alignment and @5/@10
  recall definitions. Indexes never contain question/answer/evidence labels.
- BM25S Lucene k1=1.5, b=0.75; no tuning based on validation scores.
- Qwen/Qwen3-Embedding-0.6B revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; query instruction from its snapshot,
  no document instruction, normalized 1024-dimensional vectors, CPU float32.
- Max input 512 tokens, overlap 64; long original paragraphs use character-offset
  subchunks and maximum cosine per parent. Compare this **whole retrieval
  pipeline**, not a claim that embedding alone caused every difference.
- RRF uses up to 20 unique parent paragraphs per branch and constant 60. Raw
  lexical and cosine scores are not added. No translation, reranker or LLM.
- Gold is the union of complete, uniquely aligned text-only annotations.
  Exclusions are reported in summary.json. It is not official best-reference
  Evidence F1. Incomplete gold coverage is a risk signal, not proof of a wrong answer.
- Three runs completed 250 queries each with zero execution errors. BM25 scores
  reproduce the earlier baseline. This is English structured text, not PDF,
  cross-language, all-document search or generation evaluation.

Observed validation query-only CPU latency:

| Method | p50 | p95 |
| --- | ---: | ---: |
| BM25 | 0.205 ms | 0.273 ms |
| Dense | 137.518 ms | 161.800 ms |
| Hybrid | 123.788 ms | 136.364 ms |

Queries include tokenization/query embedding/scoring/result construction, but
exclude source preparation, model loading and index build/load. The runs are
sequential, not a controlled latency comparison; warmup/load variation means
the observed lower hybrid latency is not evidence that adding RRF makes dense
inference faster. Model initialization took 4.634 seconds. Building dense indexes
for 91 papers (both splits) took 979.030 seconds (~16.3 minutes), including
document embedding. Hybrid reused those indexes; its zero new builds must not
hide this preparation cost. Standalone CLI processes load the model again.

## Reproduce and inspect

From the repository root, explicitly prepare public data and model files once:

```bash
uv sync --locked --group dev --group eval --extra ml-cpu
uv run --locked doc-rag eval download-qasper
uv run --locked doc-rag model prepare-embedding
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --extra ml-cpu --group eval python scripts/qasper_comparison.py
# Reuse all 750 completed method/question records:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked --extra ml-cpu --group eval python scripts/qasper_comparison.py --resume
```

`summary.json` stores configuration/provenance and the original measured runs;
`bm25.jsonl`, `dense.jsonl`, and `hybrid.jsonl` store only identifiers, rankings,
scores, timing and status. No questions, answers, corpus text, PDFs, databases,
vectors or model weights are published. Source/code/version changes produce new
run IDs; selection checksum allows comparisons across implementations.
`scripts/publish_qasper_comparison.py` checks selection/run identities, result
fields/scope and independently recomputes summary metrics before export.

Attribution: Dasigi et al. (2021), *A Dataset of Information-Seeking Questions and
Answers Anchored in Research Papers*. [AllenAI QASPER](https://huggingface.co/datasets/allenai/qasper)
is labeled [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) is labeled
Apache-2.0. The source PDFs are not redistributed.
