# PageCiteRAG

Local-first, bilingual document reading with answers linked to the original source text.
Ingest a document, ask in English or Traditional Chinese, and read the passages behind each
citation. The core workflow is implemented; current evaluation has not established semantic
answer quality. Agent reading remains experimental and opt-in.

## What it does

- Ingest one text-based PDF, TXT, or Markdown document into a local SQLite store.
- Search with BM25, dense retrieval, or hybrid retrieval when the matching embedding index is
  prepared.
- Ask in English or Traditional Chinese, or explain a selected passage; retrieve the cited
  original text and its page/line/character location.
- Run locally with a compatible local model backend; the application does not require a paid
  hosted LLM API.

Citation validation confirms that quoted text exists in the stored source and belongs to the
requested document. It does **not** prove that the quotation supports the answer or that the
answer preserves every important detail. Inspect citations before relying on an answer.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/marksu1104/PageCiteRAG.git
cd PageCiteRAG
uv sync --locked --no-default-groups --group dev
uv run --locked page-cite-rag --help
uv run --locked page-cite-rag ingest ./paper.pdf
```

Copy the `document_id` printed by `ingest`, then build a lexical index:

```bash
uv run --locked page-cite-rag index DOCUMENT_ID --method bm25
```

The default local database is `data/page-cite-rag.sqlite3`. Use `--db PATH` to select another
location. `ask` does not download or pull a model: use the pinned local Ollama-compatible backend
(`--backend ollama`) or prepare the pinned `llama.cpp` bundle with
`python scripts/prepare_llama.py --help`. The latter requires an existing compatible local build
toolchain; it does not install or update system drivers. Check disk, model, and GPU requirements
before preparation.

Once the matching local runtime and model are prepared, ask a question. For the prepared
`llama.cpp` bundle:

```bash
uv run --locked page-cite-rag ask DOCUMENT_ID \
  "What are the main findings and limitations?" \
  --backend llama --token-budget exact --language zh-Hant
```

With the optional `agent` dependencies installed, add `--agent` to use one bounded
source-navigation agent. The same core is available through `serve --agent`.
The reader can search, inspect the document outline, and read consecutive source passages.
The shared answer workflow then produces the answer and citations. JSON responses include a
`reading_trace` with tool activity, model-call accounting, and the navigation stopping reason.
This workflow remains experimental and opt-in; additional reading does not guarantee a
correct or complete answer.

## Optional retrieval and parsing components

Dense and hybrid indexes use the pinned Qwen embedding model. Installing the optional local
embedding dependencies and preparing model files can require several gigabytes of disk space:

```bash
uv sync --locked --no-default-groups --group dev --extra ml-cpu
uv run --locked page-cite-rag model prepare-embedding
uv run --locked page-cite-rag index DOCUMENT_ID --method hybrid
```

Docling is an optional PDF parser for structured text and native source locations. OCR is
disabled in this integration; scans and image-only evidence are outside the supported path. Its parser
dependencies and model assets are optional and larger than the core install. Start with the
default text parser and inspect extracted text before indexing.

For TXT or Markdown files, supply `--tokenizer-dir TOKENIZER_DIRECTORY` during ingestion to
split long paragraphs into token-bounded source blocks. Each block retains its exact original
character offsets. This configuration creates a distinct document identity; rebuild the index
for the new document ID.

With the pinned native GPU embedding and reranking assets already prepared, both `ask` and
`serve` support optional reranking. For example:

```bash
uv run --locked page-cite-rag index DOCUMENT_ID --method hybrid --native-gpu \
  --gpu-uuid GPU_UUID --model-dir TOKENIZER_DIRECTORY
uv run --locked page-cite-rag ask DOCUMENT_ID "What are the important conditions?" \
  --method hybrid --native-gpu --rerank --backend llama --token-budget exact \
  --device cuda --gpu-uuid GPU_UUID --model-dir TOKENIZER_DIRECTORY
```

For the API, select the same index with `serve DOCUMENT_ID --method hybrid
--embedding-backend native-gpu --rerank --device cuda --gpu-uuid GPU_UUID
--model-dir TOKENIZER_DIRECTORY`. Embedding, reranking, and answer generation run sequentially.
Reranking changes candidate order while retaining complete source chunks and the same final
answer token budget. It remains opt-in; reranking does not guarantee better answers.

## Local API

The API binds only to `127.0.0.1` and requires a bearer token on every endpoint. Set a private,
nonempty secret in `PAGE_CITE_RAG_API_TOKEN` before starting `page-cite-rag serve`; the CLI does
not generate, print, or log a token. For a temporary shell session, generate one without printing
it and start the service:

```bash
uv sync --locked --no-default-groups --group dev --extra api
export PAGE_CITE_RAG_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run --locked page-cite-rag serve DOCUMENT_ID
```

Send the value in the `Authorization: Bearer …` header from the client. The API is intended for
one local user, not as a network-facing or multi-user service.

Before publishing changes, review the staged files. Ignore rules do not prevent an explicitly
added file from being committed.

## Development checks

```bash
uv sync --locked --no-default-groups --group dev --group eval --extra api --extra gpu-embed --extra agent
uv run --no-sync ruff check src scripts tests
uv run --no-sync ruff format --check src scripts tests
uv run --no-sync python -m pytest -q
```

`CPU checks` runs on pushes to `main` and pull requests. `Dependency audit` runs when dependency
or workflow files change, every Monday at 02:27 UTC, or when started manually. Security findings
fail the audit workflow independently of the CPU tests.

GPU evaluation scripts require `PAGE_CITE_RAG_GPU_UUID` to identify the selected device.
Scoring saved results does not require a GPU.

QASPER retrieval evaluation and its scope are documented in
[`benchmarks/qasper-retrieval/README.md`](benchmarks/qasper-retrieval/README.md). Its English
paper QA metrics do not establish Chinese answer quality, PDF parsing accuracy, or full-document
comprehension. Historical experiments are not directly comparable unless their data, models,
prompts, code, and budgets match.

## Current limits

- This is a development snapshot, not a security-cleared release. The dependency lock uses
  Pillow 13 with a compatible optional Docling stack. Dependency audits remain blocking,
  point-in-time checks, not proof that arbitrary hostile documents are safe to process.
  Do not expose this local application as a public upload service.
- The supported workflow is local, single-document reading. It is not an unlimited-context or
  general multi-document research system.
- OCR, image/chart understanding, robust complex-table interpretation, and production
  multi-user authentication are not complete.
- Retrieval and citation-location tests are not semantic answer-quality results. Full-document
  conceptual understanding still needs a controlled quality evaluation.

## Repository layout

- `src/page_cite_rag/`: document storage, retrieval, answering, local inference, API, and evaluation support.
- `tests/`: regression tests for the supported workflow and evaluation metrics.
- `scripts/`: local model preparation, QASPER evaluation, scorer checks, and sanitized result export.
- `benchmarks/`: published retrieval summaries and rankings.

This repository contains the maintained application and public benchmark summaries.
