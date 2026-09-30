# doc-rag

Local-first document analysis and retrieval experiments, starting with a small,
testable Python package. The first document-ingestion slice accepts UTF-8 text,
Markdown, and text-based PDFs, stores extracted source units and paragraph blocks
in a local SQLite database, and can read the stored text back. Retrieval,
generation, and agent workflows are not implemented yet.

## Development

Use Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --group dev
uv run --locked doc-rag --help
uv run --locked doc-rag doctor --json
uv run --locked pytest -q
```

The `doctor` command reports only the Python, package, and SQLite versions/status.
Ingestion is explicit; importing the package does not scan documents, download
models, or initialize a database. The default database path is
`database.sqlite3`; pass `--db` to choose another local path.

```bash
uv run --locked doc-rag ingest ./paper.pdf --language en --usage-scope personal
uv run --locked doc-rag show DOCUMENT_ID
uv run --locked doc-rag show DOCUMENT_ID --unit 1
```

PDF extraction uses pdfplumber's text layer only. Scanned pages are recorded as
having no extractable text; OCR, table reconstruction, complex reading-order
repair, search, and answer generation are out of scope for this slice. Extracted
page text is parser output, not a claim of faithful visual or semantic PDF
reconstruction. Do not ingest documents unless you are authorized to process
them, and keep document text and databases local. The current slice has no
document-level deletion or access-control feature; `--usage-scope` is only a
local metadata label, not an authorization check. Avoid sensitive documents
until those controls exist.

## Scope