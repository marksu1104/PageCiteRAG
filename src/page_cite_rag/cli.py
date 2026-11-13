"""Local-first CLI; model resources are loaded only for explicit model commands."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sqlite3
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from page_cite_rag.ingest import (
    DocumentIngestError,
    prepare_source,
)
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL, RERANK_CANDIDATES_PER_BRANCH
from page_cite_rag.store import SQLiteDocumentStore, StoreError
from page_cite_rag.text import TokenBlockFactory

_DISTRIBUTION_NAME = "page-cite-rag"
_DEFAULT_DATABASE = Path("data/page-cite-rag.sqlite3")


def _has_native_docling_source(store: SQLiteDocumentStore, document_id: str) -> bool:
    from page_cite_rag.docling_document import has_native_source

    return has_native_source(store, document_id)


def _package_version() -> str:
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "not-installed"


def _doctor_report() -> dict[str, object]:
    sqlite_available = False
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(":memory:")
        sqlite_available = connection.execute("SELECT 1").fetchone() == (1,)
    except sqlite3.Error:
        pass
    finally:
        if connection is not None:
            connection.close()

    return {
        "package": {"name": _DISTRIBUTION_NAME, "version": _package_version()},
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "sqlite": {"available": sqlite_available, "version": sqlite3.sqlite_version},
    }


def _run_doctor(as_json: bool) -> int:
    report = _doctor_report()
    if as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(f"{report['package']['name']} {report['package']['version']}")
        print(f"Python {report['python']['implementation']} {report['python']['version']}")
        sqlite = report["sqlite"]
        status = "available" if sqlite["available"] else "unavailable"
        print(f"SQLite {sqlite['version']} ({status})")
    return 0


def _run_ingest(args: argparse.Namespace) -> int:
    try:
        # PDFs default to Docling, the validated parser: it keeps the outline that
        # document navigation uses. pdfplumber stays available on request.
        parser = args.parser
        if parser == "auto":
            parser = "docling" if Path(args.source).suffix.lower() == ".pdf" else "text"
        if parser == "docling":
            from page_cite_rag.docling_document import DEFAULT_ARTIFACTS, convert_pdf
            from page_cite_rag.embedding import DEFAULT_MODEL_DIR

            if args.pdf_reading_order != "geometric" or args.pdf_x_tolerance_ratio is not None:
                raise DocumentIngestError(
                    "Docling does not use pdfplumber spacing/order options; "
                    "pass --parser pdfplumber to use them"
                )
            bundle = convert_pdf(
                args.source,
                artifacts=args.docling_artifacts or DEFAULT_ARTIFACTS,
                tokenizer_dir=args.tokenizer_dir or DEFAULT_MODEL_DIR,
                language=args.language,
                usage_scope=args.usage_scope,
            )
            store = SQLiteDocumentStore(args.db)
            result = store.ingest(
                bundle.metadata,
                bundle.units(),
                block_factory=bundle.block_factory,
                source_manifests=bundle.source_manifests(),
            )
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
            return 1 if result.document.status == "failed" else 0
        prepared = prepare_source(
            args.source,
            language=args.language,
            usage_scope=args.usage_scope,
            pdf_reading_order=args.pdf_reading_order,
            pdf_x_tolerance_ratio=args.pdf_x_tolerance_ratio,
            chunker=(
                TokenBlockFactory(args.tokenizer_dir) if args.tokenizer_dir is not None else None
            ),
        )
        store = SQLiteDocumentStore(args.db)
        result = store.ingest(
            prepared.document, prepared.iter_units(), block_factory=prepared.block_factory
        )
    except (
        DocumentIngestError,
        ImportError,
        StoreError,
        OSError,
        sqlite3.Error,
        ValidationError,
        ValueError,
    ) as exc:
        print(f"page-cite-rag: ingest failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 1 if result.document.status == "failed" else 0


def _run_show(args: argparse.Namespace) -> int:
    try:
        store = SQLiteDocumentStore(args.db)
        document = store.get_document(args.document_id)
        if document is None:
            print("page-cite-rag: document was not found", file=sys.stderr)
            return 1

        if args.unit is None:
            report = {
                "document": document.model_dump(mode="json"),
                "units": [
                    unit.model_dump(mode="json")
                    for unit in store.get_unit_summaries(args.document_id)
                ],
            }
            if document.parser_name == "docling" and _has_native_docling_source(
                store, args.document_id
            ):
                from page_cite_rag.docling_document import load_bundle

                bundle = load_bundle(store, args.document_id)
                report["coverage"] = bundle.coverage
                report["sections"] = (
                    [section.model_dump(mode="json") for section in bundle.outline.sections]
                    if bundle.outline is not None
                    else []
                )
                report["tables"] = [
                    item.get("self_ref") for item in bundle.document.get("tables", [])
                ]
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0

        unit = store.get_unit(args.document_id, args.unit)
        if unit is None:
            print("page-cite-rag: source unit was not found", file=sys.stderr)
            return 1
        if unit.status == "error":
            print(
                f"page-cite-rag: source unit could not be extracted ({unit.error_type})",
                file=sys.stderr,
            )
            return 1
        if unit.status == "no_text":
            print("page-cite-rag: this unit has no extractable text", file=sys.stderr)
            return 0

        sys.stdout.write(unit.text)
        return 0
    except (StoreError, OSError, sqlite3.Error, ValidationError, ValueError) as exc:
        print(f"page-cite-rag: read failed: {exc}", file=sys.stderr)
        return 1


def _run_source(args: argparse.Namespace) -> int:
    from page_cite_rag.workflows import WorkflowError, read_verified_source

    try:
        store = SQLiteDocumentStore(args.db, read_only=True)
        block = read_verified_source(store, args.document_id, args.block_id)
        document = store.get_document(args.document_id)
        location = None
        if document is not None and document.parser_name == "docling":
            if _has_native_docling_source(store, args.document_id):
                from page_cite_rag.docling_document import load_bundle

                location = load_bundle(store, args.document_id).source.location_for(
                    args.document_id, args.block_id
                )
            else:
                location = store.get_source_location(args.document_id, args.block_id)
        else:
            location = store.get_source_location(args.document_id, args.block_id)
        if args.format == "json":
            report = block.model_dump(mode="json")
            if location is not None:
                report[
                    "native_location"
                    if document is not None
                    and document.parser_name == "docling"
                    and _has_native_docling_source(store, args.document_id)
                    else "source_location"
                ] = location
            print(_terminal_safe(json.dumps(report, ensure_ascii=False, indent=2)))
        else:
            source_pages = location.get("pages", []) if isinstance(location, dict) else []
            page_label = (
                f"pages {', '.join(map(str, source_pages))}, "
                if len(source_pages) > 1
                else f"page {block.page_number}, "
                if block.page_number is not None
                else ""
            )
            print(
                f"Block {block.block_id}; {page_label}lines {block.line_start}-{block.line_end}; "
                f"chars [{block.char_start}, {block.char_end})"
            )
            if location is not None:
                print(_terminal_safe(json.dumps({"native_location": location}, ensure_ascii=False)))
            print(_terminal_safe(block.text))
        return 0
    except (WorkflowError, StoreError, OSError, sqlite3.Error, ValueError):
        # Invalid IDs, database paths and source content must not echo into errors.
        print("page-cite-rag: source failed: unavailable or invalid local source", file=sys.stderr)
        return 1


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _gpu_temperature_limit(args: argparse.Namespace) -> int:
    explicit = getattr(args, "gpu_temperature_limit", None)
    if explicit is not None:
        return explicit
    if (
        getattr(args, "native_gpu", False)
        or getattr(args, "device", None) == "cuda"
        or getattr(args, "embedding_backend", None) == "native-gpu"
    ):
        return 88
    return 80


def _run_retrieval(args: argparse.Namespace) -> int:
    # Keep lexical libraries out of imports, --help and doctor startup.
    from page_cite_rag.generation import ModelError
    from page_cite_rag.retrieval import BM25Retriever, RetrievalError, build_bm25_index

    try:
        if args.gpu_temperature_limit is not None and not args.native_gpu:
            raise ValueError("GPU retrieval temperature guard requires --native-gpu")
        if not args.db.expanduser().is_file():
            raise RetrievalError("database was not found; ingest a document first")
        store = SQLiteDocumentStore(args.db)
        gpu_trace = None
        document = store.get_document(args.document_id)
        if args.method is None:
            args.method = (
                "hybrid"
                if document is not None
                and document.parser_name == "docling"
                and _has_native_docling_source(store, args.document_id)
                else "bm25"
            )
        if (
            document is not None
            and document.parser_name == "docling"
            and _has_native_docling_source(store, args.document_id)
        ):
            from page_cite_rag.docling_retrieval import build_index, load_retriever

            if args.method != "hybrid":
                raise ValueError("Docling requires --method hybrid")
            if args.native_gpu and not args.gpu_uuid:
                raise ValueError("native GPU needs an explicit GPU UUID")
            if args.native_gpu:
                from page_cite_rag.embedding import DEFAULT_MODEL_DIR
                from page_cite_rag.embedding_native import NativeEmbeddingEncoder
                from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH, NativeEmbeddingServer

                with NativeEmbeddingServer(
                    args.embedding_gguf or DEFAULT_GGUF_PATH,
                    args.gpu_uuid,
                    max_gpu_temperature_c=_gpu_temperature_limit(args),
                ) as server:
                    encoder = NativeEmbeddingEncoder(server, args.model_dir or DEFAULT_MODEL_DIR)
                    result = (
                        build_index(store, args.document_id, encoder)
                        if args.command == "index"
                        else load_retriever(store, args.document_id, encoder).search(
                            args.query, top_k=args.top_k
                        )
                    )
                    gpu_trace = server.diagnostics()
            else:
                encoder = _load_encoder(args)
                result = (
                    build_index(store, args.document_id, encoder)
                    if args.command == "index"
                    else load_retriever(store, args.document_id, encoder).search(
                        args.query, top_k=args.top_k
                    )
                )
            report = result if isinstance(result, dict) else result.model_dump(mode="json")
            if gpu_trace is not None:
                report["gpu_retrieval_trace"] = gpu_trace
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.native_gpu and (args.method == "bm25" or not args.gpu_uuid):
            raise ValueError("native GPU needs dense/hybrid and an explicit GPU UUID")
        encoder = _load_encoder(args) if args.method != "bm25" and not args.native_gpu else None
        if args.command == "index":
            if args.method == "bm25":
                result = build_bm25_index(store, args.document_id)
            else:
                from page_cite_rag.dense import build_dense_index

                if args.native_gpu:
                    from page_cite_rag.embedding import DEFAULT_MODEL_DIR
                    from page_cite_rag.embedding_native import NativeEmbeddingEncoder
                    from page_cite_rag.embedding_server import (
                        DEFAULT_GGUF_PATH,
                        NativeEmbeddingServer,
                    )

                    with NativeEmbeddingServer(
                        args.embedding_gguf or DEFAULT_GGUF_PATH,
                        args.gpu_uuid,
                        max_gpu_temperature_c=_gpu_temperature_limit(args),
                    ) as server:
                        encoder = NativeEmbeddingEncoder(
                            server, args.model_dir or DEFAULT_MODEL_DIR
                        )
                        dense = build_dense_index(store, args.document_id, encoder, versioned=True)
                        gpu_trace = server.diagnostics()
                else:
                    dense = build_dense_index(store, args.document_id, encoder)
                if args.method == "hybrid":
                    lexical = build_bm25_index(store, args.document_id)
                    result = {
                        "dense": dense.model_dump(mode="json"),
                        "bm25": lexical.model_dump(mode="json"),
                    }
                else:
                    result = dense
        else:
            if args.native_gpu:
                from page_cite_rag.embedding import DEFAULT_MODEL_DIR
                from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH
                from page_cite_rag.gpu_retrieval import StagedGpuSearch

                staged_search = StagedGpuSearch(
                    store,
                    args.document_id,
                    args.method,
                    args.embedding_gguf or DEFAULT_GGUF_PATH,
                    args.gpu_uuid,
                    tokenizer_directory=args.model_dir or DEFAULT_MODEL_DIR,
                    gpu_temperature_limit_c=_gpu_temperature_limit(args),
                )
                result = staged_search(args.query, args.top_k)
                gpu_trace = staged_search.last_trace
            elif args.method == "bm25":
                retriever = BM25Retriever(store, args.document_id)
            else:
                from page_cite_rag.dense import DenseRetriever

                retriever = DenseRetriever(store, args.document_id, encoder)
                if args.method == "hybrid":
                    from page_cite_rag.hybrid import HybridRetriever

                    retriever = HybridRetriever(
                        BM25Retriever(store, args.document_id),
                        retriever,
                        candidates=max(20, args.top_k),
                    )
            if not args.native_gpu:
                result = retriever.search(args.query, top_k=args.top_k)
        report = result if isinstance(result, dict) else result.model_dump(mode="json")
        if gpu_trace is not None:
            report["gpu_retrieval_trace"] = gpu_trace
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except (
        ModelError,
        RetrievalError,
        StoreError,
        OSError,
        sqlite3.Error,
        ValidationError,
        ValueError,
    ) as exc:
        print(f"page-cite-rag: {args.command} failed: {exc}", file=sys.stderr)
        return 1


def _run_eval(args: argparse.Namespace) -> int:
    from page_cite_rag.qasper import QasperError, download_qasper

    try:
        if args.eval_action == "download-qasper":
            report = download_qasper(args.data_dir)
        else:
            from page_cite_rag.evaluation import EvaluationError, run_qasper
            from page_cite_rag.retrieval import RetrievalError

            try:
                start = time.perf_counter()
                encoder = _load_encoder(args) if args.method != "bm25" else None
                model_load_ms = (time.perf_counter() - start) * 1000 if encoder else 0
                report = run_qasper(
                    args.data_dir,
                    args.output_dir,
                    dev_questions=args.dev_questions,
                    validation_questions=args.validation_questions,
                    seed=args.seed,
                    resume=args.resume,
                    retry_errors=args.retry_errors,
                    method=args.method,
                    encoder=encoder,
                    candidates=args.candidates,
                    rrf_constant=args.rrf_constant,
                    model_load_ms=model_load_ms,
                    index_database=args.index_db,
                )
            except (EvaluationError, RetrievalError, ValueError) as exc:
                print(f"page-cite-rag: evaluation failed: {exc}", file=sys.stderr)
                return 1
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        errors = sum(row["errors"] for row in report.get("results", {}).values())
        return 1 if errors else 0
    except (QasperError, StoreError, OSError, sqlite3.Error, ValidationError) as exc:
        print(f"page-cite-rag: evaluation failed: {exc}", file=sys.stderr)
        return 1


def _embedding_options(parser: argparse.ArgumentParser) -> None:
    # No embedding/model imports for --help, doctor or BM25 commands.
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--max-tokens", type=_positive_integer, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=_positive_integer, default=4)
    parser.add_argument("--threads", type=_positive_integer, default=6)


def _load_encoder(args: argparse.Namespace):
    from page_cite_rag.embedding import DEFAULT_MODEL_DIR, QwenEmbeddingEncoder

    return QwenEmbeddingEncoder(
        args.model_dir or DEFAULT_MODEL_DIR,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _run_model(args: argparse.Namespace) -> int:
    from page_cite_rag.embedding import DEFAULT_MODEL_DIR, prepare_embedding_model
    from page_cite_rag.retrieval import RetrievalError

    try:
        if args.model_action == "prepare-docling":
            from page_cite_rag.docling_document import DEFAULT_ARTIFACTS, prepare_models

            report = prepare_models(args.artifacts or DEFAULT_ARTIFACTS)
        elif args.model_action == "prepare-native-embedding":
            from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH, prepare_native_model

            report = prepare_native_model(args.embedding_gguf or DEFAULT_GGUF_PATH)
        else:
            report = prepare_embedding_model(args.model_dir or DEFAULT_MODEL_DIR)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except (RetrievalError, OSError, ValueError) as exc:
        print(f"page-cite-rag: model preparation failed: {exc}", file=sys.stderr)
        return 1


# Plan 5.13 validated the pinned GGUF on the owned Ollama server; it is the default.
OWNED_BACKENDS = ("pinned-ollama", "llama")
# The CLI permits up to an hour; its default and the local API limit stay at 600 s.
MAX_CLI_REQUEST_SECONDS = 3600


def _generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("pinned-ollama", "llama", "ollama", "api"),
        default="pinned-ollama",
        help=(
            "pinned-ollama (default) and llama serve the pinned, hash-checked GGUF from an "
            "owned local server with exact token counting, as validated. ollama connects to "
            "your own Ollama server and --llm-model: an unvalidated configuration."
        ),
    )
    parser.add_argument("--api-base", help="Explicit OpenAI-compatible HTTPS API base URL.")
    parser.add_argument("--api-model", help="Explicit cloud model name (--backend api).")
    parser.add_argument("--api-key-env", default="PAGECITE_API_KEY")
    parser.add_argument("--api-max-calls", type=_positive_integer, default=100)
    parser.add_argument(
        "--token-budget",
        choices=("conservative", "exact"),
        default=None,
        help="Defaults to exact for the owned backends and conservative for your own Ollama.",
    )
    parser.add_argument(
        "--llm-model", help="Model name on your own Ollama server (--backend ollama)."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--gpu-uuid", help="One explicit GPU UUID; only the owned backends support CUDA."
    )
    parser.add_argument(
        "--gpu-temperature-limit",
        type=_positive_integer,
        help="GPU temperature stop threshold (max 88 C; defaults to 88 C for explicit GPU use).",
    )
    parser.add_argument("--bundle-dir", type=Path, help="Explicitly prepared llama runtime/model.")
    parser.add_argument("--llm-host", default="http://127.0.0.1:11435")
    parser.add_argument("--context-window", type=_positive_integer, default=16384)
    parser.add_argument("--output-tokens", type=_positive_integer, default=1024)
    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=600.0,
        help="Seconds per model call (default 600, at most 3600 for CPU-only machines).",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=600.0,
        help="Seconds for the whole question (default 600, at most 3600 on the CLI).",
    )
    parser.add_argument("--llm-threads", type=_positive_integer, default=2)
    parser.add_argument("--language", choices=("en", "zh-Hant", "auto"), default="zh-Hant")
    parser.add_argument("--format", choices=("text", "json"), default="text")


def _generation_config(args: argparse.Namespace):
    # Validate choices before loading a large embedding model or starting a server.
    from page_cite_rag.generation import DEFAULT_LLM, GenerationConfig

    if args.token_budget is None:
        args.token_budget = (
            "exact"
            if args.backend in OWNED_BACKENDS
            or (args.command in ("serve", "mcp") and args.backend != "api")
            else "conservative"
        )
    if args.token_budget == "exact" and args.backend not in OWNED_BACKENDS:
        raise ValueError("exact token budget requires an owned backend (pinned-ollama or llama)")
    if args.llm_model is not None and args.backend != "ollama":
        raise ValueError("--llm-model selects a model on your own Ollama (--backend ollama)")

    api_retrieval = args.backend == "api" and (
        getattr(args, "native_gpu", False)
        or getattr(args, "embedding_backend", None) == "native-gpu"
    )
    if (
        args.gpu_temperature_limit is not None
        and not api_retrieval
        and (args.backend not in OWNED_BACKENDS or args.device != "cuda")
    ):
        raise ValueError("--gpu-temperature-limit requires an owned backend on CUDA")

    if args.backend == "api":
        from page_cite_rag.api_backend import validate_api_base
        from page_cite_rag.generation import ModelError

        if not args.api_base:
            raise ModelError("--api-base is required for --backend api")
        validate_api_base(args.api_base)
        if not args.api_model or not args.api_model.strip():
            raise ModelError("--api-model is required for --backend api")
        if args.bundle_dir is not None or args.llm_host != "http://127.0.0.1:11435":
            raise ModelError("--backend api does not use --bundle-dir or --llm-host")
        return GenerationConfig(
            num_ctx=args.context_window,
            num_predict=args.output_tokens,
            timeout_seconds=args.generation_timeout,
            num_thread=args.llm_threads,
        )
    if args.backend == "ollama":
        if args.device != "cpu" or args.gpu_uuid is not None or args.bundle_dir is not None:
            raise ValueError("your own Ollama runs as configured there; CUDA/bundle need owned")
        return GenerationConfig(
            host=args.llm_host,
            model=args.llm_model or DEFAULT_LLM,
            num_ctx=args.context_window,
            num_predict=args.output_tokens,
            timeout_seconds=args.generation_timeout,
            num_thread=args.llm_threads,
            num_gpu=0,
        )
    from page_cite_rag.llama import DEFAULT_BUNDLE, LlamaConfig

    if args.llm_host != "http://127.0.0.1:11435":
        raise ValueError("owned backends start their own server; --llm-host needs --backend ollama")
    return LlamaConfig(
        bundle_dir=args.bundle_dir or DEFAULT_BUNDLE,
        device=args.device,
        gpu_uuid=args.gpu_uuid,
        num_ctx=args.context_window,
        num_predict=args.output_tokens,
        timeout_seconds=args.generation_timeout,
        num_thread=args.llm_threads,
        max_gpu_temperature_c=_gpu_temperature_limit(args),
    )


def _open_generation_client(args: argparse.Namespace, config):
    if args.backend == "api":
        from page_cite_rag.api_backend import ApiChatClient

        client = ApiChatClient(
            config,
            api_base=args.api_base,
            model=args.api_model,
            api_key_env=args.api_key_env,
            max_calls=args.api_max_calls,
        )
        print(
            "Document passages and questions will be sent to "
            f"{client.identity['api_host']} (--backend api).",
            file=sys.stderr,
        )
        return client
    if args.backend == "ollama":
        from page_cite_rag.generation import OllamaModelClient

        print(
            "page-cite-rag: using your own Ollama model; this configuration is not the "
            "validated one, so the published comparison does not apply to it.",
            file=sys.stderr,
        )
        return OllamaModelClient(config)
    if args.backend == "pinned-ollama":
        from page_cite_rag.ollama_backend import OllamaGgufClient

        return OllamaGgufClient(config, budget_policy="exact", recover_output_errors=True)
    from page_cite_rag.llama import LlamaModelClient

    if args.token_budget == "exact":
        return LlamaModelClient(config, budget_policy="exact", recover_output_errors=True)
    return LlamaModelClient(config, recover_output_errors=True)


def _answer_retrievers(args: argparse.Namespace, store: SQLiteDocumentStore):
    """Load existing snapshots only; return lexical/dense parts for explicit routing."""
    from page_cite_rag.application import document_retriever_parts

    encoder = _load_encoder(args) if args.method != "bm25" else None
    return document_retriever_parts(store, args.document_id, args.method, encoder=encoder)


def _terminal_safe(value: str) -> str:
    """Escape terminal controls for display only; stored text and citations stay exact."""
    return "".join(
        f"\\u{ord(character):04x}"
        if (
            (ord(character) < 32 and character not in "\n\t")
            or 0x7F <= ord(character) <= 0x9F
            or 0x202A <= ord(character) <= 0x202E
            or 0x2066 <= ord(character) <= 0x2069
        )
        else character
        for character in value
    )


def _closest_passages(result, store) -> list[dict]:
    if result.status != "insufficient_evidence" or store is None:
        return []
    passages = []
    for block_id in result.included_block_ids[:3]:
        block = store.get_block(result.document_id, block_id)
        if block is not None:
            passages.append(
                {
                    "block_id": block.block_id,
                    "page_number": block.page_number,
                    "line_start": block.line_start,
                    "line_end": block.line_end,
                    "text": block.text[:300],
                }
            )
    return passages


def _print_answer(result, output_format: str, store=None) -> None:
    if output_format == "json":
        report = result.model_dump(mode="json")
        if result.status == "insufficient_evidence":
            report["closest_passages"] = _closest_passages(result, store)
        print(_terminal_safe(json.dumps(report, ensure_ascii=False, indent=2)))
        return

    def print_claim(claim, label: str) -> None:
        print(f"- {_terminal_safe(claim.text)}")
        for citation in claim.citations:
            location = f"page {citation.page_number}, " if citation.page_number is not None else ""
            print(
                f"  {label}: {location}lines {citation.line_start}-{citation.line_end}; "
                f"block {citation.block_id}; chars [{citation.char_start}, {citation.char_end})"
            )
            print(f"  {_terminal_safe(json.dumps(citation.quote, ensure_ascii=False))}")

    if result.answer_claims:
        print("Answer:")
        for claim in result.answer_claims:
            print_claim(claim, "answer evidence")
    else:
        print(f"Answer: {_terminal_safe(result.answer)}")
    if result.detail_claims:
        print("\nDetails:")
        for claim in result.detail_claims:
            print_claim(claim, "detail evidence")
    print(f"\nStatus: {result.status}; stop reason: {result.stop_reason}")
    if result.omitted_block_ids:
        print(f"Context omitted {len(result.omitted_block_ids)} whole candidate blocks.")
    if result.retrieval is not None and result.retrieval.query_plan is not None:
        plan = result.retrieval.query_plan
        lexical_query = _terminal_safe(str(plan.get("lexical_query", result.retrieval.query)))
        print(f"Lexical query: {lexical_query}")
        rewrite = plan.get("rewrite", plan)
        if "status" in rewrite:
            status = _terminal_safe(str(rewrite["status"]))
            reason = _terminal_safe(str(rewrite.get("reason", "not_recorded")))
            print(f"Query rewrite: {status} ({reason})")
    for limitation in result.limitations:
        print(f"Note: {_terminal_safe(limitation)}")

    if result.status == "insufficient_evidence":
        print("Most relevant passages:")
        for passage in _closest_passages(result, store):
            page = passage["page_number"]
            location = f"page {page}, " if page is not None else ""
            print(
                f"- {location}lines {passage['line_start']}-{passage['line_end']}; "
                f"block {passage['block_id']}: {_terminal_safe(passage['text'])}"
            )


_ROUTE_REASONS = {
    "requested": "requested with --route",
    "auto": "auto",
}


def _print_route(trace: dict) -> None:
    route = trace.get("route")
    if route == "whole_document_context":
        print("Route: whole document (fits the context).")
    elif route == "reranked":
        print("Route: reranked top 20 of 200.")
    elif route == "agent":
        print(
            f"Route: evidence agent ({trace['model_calls']} model calls, "
            f"{len(trace['search_steps'])} searches)."
        )
    else:
        reason = (
            "no reranker"
            if trace.get("reason") == "rerank_unavailable"
            else _ROUTE_REASONS.get(trace.get("reason"), str(trace.get("reason")))
        )
        print(f"Route: fixed retrieval ({reason}).")


class _RouteConfigurationError(ValueError):
    """Safe, fixed route guidance that contains no user input or local paths."""


def _run_answer(args: argparse.Namespace) -> int:
    from page_cite_rag.application import (
        ask_document,
        document_retriever,
        document_workflow,
        explain_source,
    )
    from page_cite_rag.generation import ModelError, ModelTimeoutError
    from page_cite_rag.retrieval import RetrievalError
    from page_cite_rag.workflows import WorkflowConfig

    client = None
    retrieval_owner = [None]
    route = getattr(args, "route", "fixed") if args.command == "ask" else "fixed"
    native_gpu = getattr(args, "native_gpu", False)
    request_timeout = args.request_timeout_seconds
    if (
        type(request_timeout) not in (int, float)
        or not math.isfinite(request_timeout)
        or not 1 < request_timeout <= MAX_CLI_REQUEST_SECONDS
    ):
        print(
            "page-cite-rag: request timeout must be greater than 1 and at most "
            f"{MAX_CLI_REQUEST_SECONDS} seconds",
            file=sys.stderr,
        )
        return 1
    request_deadline = time.monotonic() + request_timeout
    deadline_expired = threading.Event()

    def stop_request() -> None:
        deadline_expired.set()
        cancel_retrieval = getattr(retrieval_owner[0], "cancel", None)
        if callable(cancel_retrieval):
            try:
                cancel_retrieval()
            except Exception:
                pass
        close_model = getattr(client, "close", None)
        if callable(close_model):
            try:
                close_model()
            except Exception:
                pass

    deadline_timer = threading.Timer(request_timeout, stop_request)
    deadline_timer.daemon = True
    deadline_timer.start()

    def ensure_request_time() -> None:
        if deadline_expired.is_set() or time.monotonic() >= request_deadline:
            deadline_expired.set()
            raise ModelTimeoutError("local request exceeded its total deadline")

    try:
        config = _generation_config(args)
        if not args.question.strip() or len(args.question) > 4000:
            raise ValueError("question must be nonblank and at most 4000 characters")
        args.question.encode("utf-8")
        workflow_config = WorkflowConfig(
            top_k=args.top_k if args.command == "ask" else 6,
            neighbor_blocks=args.neighbors if args.command == "explain" else 0,
        )
        if not args.db.expanduser().is_file():
            raise ValueError("database was not found; ingest a document first")
        store = SQLiteDocumentStore(args.db, read_only=True)
        document = store.get_document(args.document_id)
        if document is None:
            raise ValueError("document was not found")
        docling_source = document.parser_name == "docling" and _has_native_docling_source(
            store, args.document_id
        )
        if args.command == "ask" and args.method is None:
            args.method = "hybrid" if docling_source or native_gpu else "bm25"
        if args.command == "ask":
            # RetrievalError messages are shown to the user; they carry no input or paths.
            if route != "fixed" and (args.rewrite_query or args.rerank):
                raise _RouteConfigurationError(
                    "--rewrite-query and --rerank apply to --route fixed"
                )
            if route == "agent" and (
                not native_gpu
                or args.method != "hybrid"
                or args.backend not in ("pinned-ollama", "api")
            ):
                raise _RouteConfigurationError(
                    "agent requires --native-gpu, --method hybrid and --backend pinned-ollama"
                )
            if route == "auto" and native_gpu:
                args.method = "hybrid"
            if args.rerank and not native_gpu:
                raise ValueError("--rerank requires --native-gpu")
            if native_gpu and (
                args.backend not in (*OWNED_BACKENDS, "api")
                or (args.backend != "api" and args.device != "cuda")
                or not args.gpu_uuid
                or args.method not in ("dense", "hybrid")
            ):
                raise ValueError("--native-gpu requires dense/hybrid with an owned backend on CUDA")
            if args.rewrite_query and args.method == "dense":
                raise ValueError("query rewriting is lexical-only; use bm25 or hybrid")
            if native_gpu:
                from page_cite_rag.embedding import DEFAULT_MODEL_DIR
                from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH
                from page_cite_rag.gpu_retrieval import StagedGpuSearch
                from page_cite_rag.reranker_server import DEFAULT_MODEL_PATH

                gpu_retriever = StagedGpuSearch(
                    store,
                    args.document_id,
                    args.method,
                    args.embedding_gguf or DEFAULT_GGUF_PATH,
                    args.gpu_uuid,
                    tokenizer_directory=args.model_dir or DEFAULT_MODEL_DIR,
                    gpu_temperature_limit_c=_gpu_temperature_limit(args),
                    reranker_model_path=(
                        DEFAULT_MODEL_PATH
                        if args.rerank or (route == "auto" and DEFAULT_MODEL_PATH.is_file())
                        else None
                    ),
                    candidates=(
                        RERANK_CANDIDATES_PER_BRANCH
                        if args.rerank or (route == "auto" and DEFAULT_MODEL_PATH.is_file())
                        else FORMAL_RETRIEVAL.candidates_per_branch
                    ),
                )
                retrieval_owner[0] = gpu_retriever
                if args.rewrite_query:
                    from page_cite_rag.query import QueryRewriteConfig, QueryRewriter
                    from page_cite_rag.query_retrieval import RewritingRetriever

                    client = _open_generation_client(args, config)
                    rewriter = QueryRewriter(
                        client,
                        config,
                        QueryRewriteConfig.for_document_language(document.language),
                    )
                    retriever = RewritingRetriever(None, rewriter, search_backend=gpu_retriever)
                else:
                    retriever = gpu_retriever
                lexical, dense = None, None
            elif docling_source:
                if args.method != "hybrid":
                    raise ValueError("Docling source requires --method hybrid")
                retriever = document_retriever(
                    store,
                    args.document_id,
                    "hybrid",
                    encoder=_load_encoder(args),
                    candidates=FORMAL_RETRIEVAL.candidates_per_branch,
                )
                lexical, dense = None, None
            else:
                lexical, dense = _answer_retrievers(args, store)
        else:
            if store.get_block(args.document_id, args.block_id) is None:
                raise ValueError("explanation block was not found within this document")
            lexical, dense = None, None

        precomputed_search = None
        precomputed_blocks = ()
        initial_gpu_search_trace = None
        if native_gpu and route == "fixed":
            search_workflow = document_workflow(
                store,
                args.document_id,
                retriever,
                None,
                config,
                top_k=workflow_config.top_k,
            )
            precomputed_search, precomputed_blocks = search_workflow.search_sources(args.question)
            precomputed_blocks = search_workflow._context_blocks(
                precomputed_search, precomputed_blocks
            )
            gpu_backend = getattr(retriever, "_search_backend", retriever)
            if isinstance(getattr(gpu_backend, "last_trace", None), dict):
                initial_gpu_search_trace = dict(gpu_backend.last_trace)

        if client is None:
            client = _open_generation_client(args, config)
        set_deadline = getattr(client, "set_request_deadline", None)
        if callable(set_deadline):
            set_deadline(request_deadline)
        ensure_request_time()
        if args.command != "ask":
            retriever = None
        elif native_gpu:
            pass
        elif docling_source:
            if args.rewrite_query:
                from page_cite_rag.docling_retrieval import RewritingDoclingRetriever
                from page_cite_rag.query import QueryRewriteConfig, QueryRewriter

                rewriter = QueryRewriter(
                    client,
                    config,
                    QueryRewriteConfig.for_document_language(document.language),
                )
                retriever = RewritingDoclingRetriever(retriever, rewriter)
        else:
            if args.rewrite_query:
                from page_cite_rag.query import QueryRewriteConfig, QueryRewriter
                from page_cite_rag.query_retrieval import RewritingRetriever

                rewriter = QueryRewriter(
                    client,
                    config,
                    QueryRewriteConfig.for_document_language(document.language),
                )
                retriever = RewritingRetriever(lexical, rewriter, dense)
            elif args.method == "hybrid":
                from page_cite_rag.hybrid import HybridRetriever

                retriever = HybridRetriever(
                    lexical,
                    dense,
                    candidates=FORMAL_RETRIEVAL.candidates_per_branch,
                )
            else:
                retriever = lexical if lexical is not None else dense
        if retrieval_owner[0] is None:
            retrieval_owner[0] = retriever
        workflow = document_workflow(
            store,
            args.document_id,
            retriever,
            client,
            config,
            top_k=workflow_config.top_k,
            neighbor_blocks=workflow_config.neighbor_blocks,
            writer=(
                "quote_first"
                if route == "agent" or getattr(args, "answer_format", "general") == "short"
                else "structured"
            ),
            answer_format=(
                "short_answer"
                if getattr(args, "answer_format", "general") == "short"
                else "general"
            ),
        )
        if route != "fixed":
            from page_cite_rag.application import answer_question

            routed = answer_question(
                workflow,
                args.question,
                language=args.language,
                source_language=document.language,
                route=route,
            )
            ensure_request_time()
            result = routed.result
            if args.format == "json":
                report = {"answer": result.model_dump(mode="json"), "routing": routed.trace}
                if (
                    result.status == "insufficient_evidence"
                    and routed.trace.get("route") != "whole_document_context"
                ):
                    report["closest_passages"] = _closest_passages(result, store)
                if initial_gpu_search_trace is not None:
                    report["initial_gpu_search_trace"] = initial_gpu_search_trace
                if (
                    args.backend in OWNED_BACKENDS
                    and args.device == "cuda"
                    and callable(getattr(client, "diagnostics", None))
                ):
                    report["gpu_generation_diagnostics"] = client.diagnostics()
                print(
                    _terminal_safe(
                        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                    )
                )
            else:
                # Passages of a whole-document read are in document order, not by relevance.
                whole = routed.trace.get("route") == "whole_document_context"
                _print_answer(result, args.format, None if whole else store)
                _print_route(routed.trace)
        else:
            if args.command == "ask" and precomputed_search is not None:
                result = workflow.answer_from_context(
                    args.question,
                    precomputed_blocks,
                    language=args.language,
                    context_groups=precomputed_search.context_groups,
                )
            elif args.command == "ask":
                result = ask_document(workflow, args.question, language=args.language)
            else:
                result = explain_source(
                    workflow, args.block_id, args.question, language=args.language
                )
            ensure_request_time()
            if args.format == "json" and (initial_gpu_search_trace or args.device == "cuda"):
                report = result.model_dump(mode="json")
                if result.status == "insufficient_evidence":
                    report["closest_passages"] = _closest_passages(result, store)
                if initial_gpu_search_trace is not None:
                    report["gpu_retrieval_trace"] = initial_gpu_search_trace
                if (
                    args.backend in OWNED_BACKENDS
                    and args.device == "cuda"
                    and callable(getattr(client, "diagnostics", None))
                ):
                    report["gpu_generation_diagnostics"] = client.diagnostics()
                print(
                    _terminal_safe(
                        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                    )
                )
            else:
                _print_answer(result, args.format, store)
        ensure_request_time()
        return 0  # Evidence insufficiency is a visible result, not a transport error.
    except (ModelError, RetrievalError, StoreError, OSError, sqlite3.Error, ValueError) as exc:
        # ValidationError/OS errors can contain raw user input or paths; don't dump them.
        detail = (
            "invalid configuration or local resource"
            if not isinstance(exc, (ModelError, RetrievalError, _RouteConfigurationError))
            else str(exc)
        )
        print(f"page-cite-rag: {args.command} failed: {detail}", file=sys.stderr)
        return 1
    finally:
        deadline_timer.cancel()
        if deadline_timer is not threading.current_thread():
            deadline_timer.join(timeout=5)
        if client is not None:
            client.close()


def _create_document_service(args: argparse.Namespace):
    """Shared configuration and runtime construction for API and MCP transports."""
    store = SQLiteDocumentStore(args.db, read_only=True)
    if args.method is None:
        record = store.get_document(args.document_id)
        args.method = (
            "hybrid"
            if args.embedding_backend == "native-gpu"
            or (
                record is not None
                and record.parser_name == "docling"
                and _has_native_docling_source(store, args.document_id)
            )
            else "bm25"
        )
    config = _generation_config(args)
    if args.backend != "api" and (
        args.backend not in OWNED_BACKENDS or args.token_budget != "exact"
    ):
        raise ValueError("local API requires an owned backend with exact token budgeting")
    from page_cite_rag.service import DocumentService, ServiceConfig

    if args.rerank and args.route != "fixed":
        raise _RouteConfigurationError("--rewrite-query and --rerank apply to --route fixed")
    if args.rerank and args.embedding_backend != "native-gpu":
        raise ValueError("--rerank requires --embedding-backend native-gpu")
    if args.route == "agent" and (
        args.embedding_backend != "native-gpu"
        or args.method != "hybrid"
        or args.backend not in ("pinned-ollama", "api")
    ):
        raise _RouteConfigurationError(
            "agent requires --native-gpu, --method hybrid and --backend pinned-ollama"
        )
    if args.route == "auto" and args.embedding_backend == "native-gpu":
        args.method = "hybrid"
    if args.embedding_backend == "native-gpu" and (
        args.method == "bm25"
        or (args.backend != "api" and args.device != "cuda")
        or not args.gpu_uuid
    ):
        raise ValueError("native GPU embedding needs dense/hybrid and CUDA generation")
    encoder = (
        _load_encoder(args) if args.method != "bm25" and args.embedding_backend == "cpu" else None
    )
    staged_search = None
    if args.embedding_backend == "native-gpu":
        from page_cite_rag.embedding import DEFAULT_MODEL_DIR
        from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH
        from page_cite_rag.gpu_retrieval import StagedGpuSearch
        from page_cite_rag.reranker_server import DEFAULT_MODEL_PATH

        staged_search = StagedGpuSearch(
            SQLiteDocumentStore(args.db, read_only=True),
            args.document_id,
            args.method,
            args.embedding_gguf or DEFAULT_GGUF_PATH,
            args.gpu_uuid,
            tokenizer_directory=args.model_dir or DEFAULT_MODEL_DIR,
            gpu_temperature_limit_c=_gpu_temperature_limit(args),
            reranker_model_path=(
                DEFAULT_MODEL_PATH
                if args.rerank or (args.route == "auto" and DEFAULT_MODEL_PATH.is_file())
                else None
            ),
            candidates=(
                RERANK_CANDIDATES_PER_BRANCH
                if args.rerank or (args.route == "auto" and DEFAULT_MODEL_PATH.is_file())
                else FORMAL_RETRIEVAL.candidates_per_branch
            ),
        )
    return DocumentService(
        ServiceConfig(
            database_path=args.db,
            document_id=args.document_id,
            method=args.method,
            top_k=args.top_k,
            queue_capacity=args.queue_capacity,
            idle_seconds=args.idle_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            route=args.route,
            answer_format=args.answer_format,
        ),
        config,
        lambda: _open_generation_client(args, config),
        encoder=encoder,
        staged_search=staged_search,
    )


def _run_serve(args: argparse.Namespace) -> int:
    """One bound document and one worker; never expose a network-facing listener."""
    from page_cite_rag.generation import ModelError
    from page_cite_rag.retrieval import RetrievalError

    try:
        api_token = os.environ.get("PAGE_CITE_RAG_API_TOKEN")
        if api_token is None or not api_token.strip():
            raise ValueError("PAGE_CITE_RAG_API_TOKEN must contain a local API secret")
        if not 1 <= args.port <= 65535:
            raise ValueError("invalid loopback port")
        from page_cite_rag.api import create_app

        runtime = _create_document_service(args)
        import uvicorn

        uvicorn.run(
            create_app(runtime, api_token=api_token),
            host="127.0.0.1",
            port=args.port,
            workers=1,
            access_log=False,
            log_level="warning",
        )
        return 0
    except (ImportError, ValueError, OSError, StoreError, RetrievalError, ModelError) as exc:
        detail = (
            str(exc)
            if isinstance(exc, (_RouteConfigurationError, ModelError))
            else type(exc).__name__
        )
        print(f"page-cite-rag: local service failed: {detail}", file=sys.stderr)
        return 1


def _run_mcp(args: argparse.Namespace) -> int:
    from page_cite_rag.generation import ModelError
    from page_cite_rag.retrieval import RetrievalError

    try:
        from page_cite_rag.mcp_server import run_stdio
    except ImportError:
        print("page-cite-rag: MCP requires uv sync --extra mcp", file=sys.stderr)
        return 1
    try:
        run_stdio(_create_document_service(args))
        return 0
    except (ImportError, ValueError, OSError, StoreError, RetrievalError, ModelError) as exc:
        detail = (
            str(exc)
            if isinstance(exc, (_RouteConfigurationError, ModelError))
            else type(exc).__name__
        )
        print(f"page-cite-rag: local service failed: {detail}", file=sys.stderr)
        return 1


_ROUTE_HELP = (
    "Validated auto: use the whole document if it fits, otherwise with --native-gpu "
    "rerank 200 candidates and keep 20, otherwise fixed retrieval. "
    "agent: LangChain evidence Agent for multi-step questions; slower, requires --native-gpu."
)


def _document_service_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("document_id", help="The only document accessible to this API.")
    parser.add_argument("--db", type=Path, default=_DEFAULT_DATABASE)
    parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default=None)
    parser.add_argument("--rerank", action="store_true", help="Rerank native GPU candidates.")
    parser.add_argument(
        "--route",
        choices=("auto", "fixed", "agent"),
        default="auto",
        help=_ROUTE_HELP,
    )
    parser.add_argument("--answer-format", choices=("general", "short"), default="general")
    parser.add_argument("--top-k", type=_positive_integer, default=FORMAL_RETRIEVAL.final_k)
    parser.add_argument("--queue-capacity", type=int, default=2)
    parser.add_argument("--idle-seconds", type=float, default=300.0)
    parser.add_argument("--embedding-backend", choices=("cpu", "native-gpu"), default="cpu")
    parser.add_argument("--embedding-gguf", type=Path)
    _embedding_options(parser)
    _generation_options(parser)
    parser.set_defaults(output_tokens=3072)
    parser.set_defaults(token_budget=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="page-cite-rag",
        description="Local-first bilingual document reading with source-grounded answers.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report the Python package and SQLite runtime status.",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the allowlisted status fields as JSON.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Extract UTF-8 text or text-based PDF pages into a local SQLite store.",
    )
    ingest_parser.add_argument(
        "source", type=Path, help="Path to one .txt, .md, .markdown, or .pdf file."
    )
    ingest_parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DATABASE,
        help=f"SQLite file (default: {_DEFAULT_DATABASE}).",
    )
    ingest_parser.add_argument(
        "--language",
        help="Optional language label such as en or zh-Hant; it is not auto-detected.",
    )
    ingest_parser.add_argument(
        "--usage-scope",
        default="not-recorded",
        help="Local note describing the document's permitted use (default: not-recorded).",
    )
    ingest_parser.add_argument(
        "--pdf-reading-order",
        choices=("geometric", "text-flow"),
        default="geometric",
        help=(
            "PDF extraction order; text-flow uses the embedded character stream and "
            "font-relative spacing. Inspect output before indexing; not layout detection."
        ),
    )
    ingest_parser.add_argument(
        "--pdf-x-tolerance-ratio",
        type=float,
        help=(
            "Font-relative pdfplumber character spacing (0, 1]; defaults to 0.15 and is "
            "independent of geometric/text-flow reading order. Parser identity includes it."
        ),
    )
    ingest_parser.add_argument(
        "--parser",
        choices=("auto", "docling", "pdfplumber"),
        default="auto",
        help=(
            "auto (default) parses PDFs with Docling, which keeps the section outline and "
            "needs prepared local models; pdfplumber has no outline."
        ),
    )
    ingest_parser.add_argument(
        "--docling-artifacts", type=Path, help="Prepared offline Docling layout/table models."
    )
    ingest_parser.add_argument(
        "--tokenizer-dir", type=Path, help="Prepared Qwen embedding tokenizer directory."
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show stored source metadata or read back one source unit.",
    )
    show_parser.add_argument("document_id", help="Document ID returned by ingest.")
    show_parser.add_argument(
        "--unit",
        type=_positive_integer,
        help="Read a 1-based PDF page or the sole text-file unit.",
    )
    show_parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DATABASE,
        help=f"SQLite file (default: {_DEFAULT_DATABASE}).",
    )

    source_parser = subparsers.add_parser(
        "source", help="Read one verified original block from a stored document."
    )
    source_parser.add_argument("document_id", help="Document ID returned by ingest.")
    source_parser.add_argument("block_id", help="Block ID returned by search or a citation.")
    source_parser.add_argument("--db", type=Path, default=_DEFAULT_DATABASE)
    source_parser.add_argument("--format", choices=("text", "json"), default="text")

    index_parser = subparsers.add_parser(
        "index", help="Build the matching local index for one stored document."
    )
    search_parser = subparsers.add_parser(
        "search", help="Search one document's index and return source blocks."
    )
    for retrieval_parser in (index_parser, search_parser):
        retrieval_parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default=None)
        _embedding_options(retrieval_parser)
        retrieval_parser.add_argument("--native-gpu", action="store_true")
        retrieval_parser.add_argument("--embedding-gguf", type=Path)
        retrieval_parser.add_argument("--gpu-uuid")
        retrieval_parser.add_argument(
            "--gpu-temperature-limit",
            type=_positive_integer,
            help=(
                "GPU temperature stop threshold (max 88 C; defaults to 88 C for explicit GPU use)."
            ),
        )
        retrieval_parser.add_argument("document_id", help="Document ID returned by ingest.")
        retrieval_parser.add_argument(
            "--db",
            type=Path,
            default=_DEFAULT_DATABASE,
            help=f"SQLite file (default: {_DEFAULT_DATABASE}).",
        )
    search_parser.add_argument("query", help="Lexical query; quote queries containing spaces.")
    search_parser.add_argument("--top-k", type=_positive_integer, default=5)

    eval_parser = subparsers.add_parser(
        "eval", help="Prepare public data or run offline retrieval evaluation."
    )
    eval_commands = eval_parser.add_subparsers(dest="eval_action", required=True)
    download_parser = eval_commands.add_parser(
        "download-qasper",
        help="Explicitly download the pinned QASPER train/validation Parquet files.",
    )
    qasper_parser = eval_commands.add_parser(
        "qasper", help="Run the frozen, document-scoped BM25 QASPER baseline offline."
    )
    for command in (download_parser, qasper_parser):
        command.add_argument("--data-dir", type=Path, default=Path("data/qasper"))
    qasper_parser.add_argument("--output-dir", type=Path, default=Path("outputs/qasper-bm25"))
    qasper_parser.add_argument("--dev-questions", type=_positive_integer, default=50)
    qasper_parser.add_argument("--validation-questions", type=_positive_integer, default=200)
    qasper_parser.add_argument("--seed", type=int, default=42)
    qasper_parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default="bm25")
    qasper_parser.add_argument("--candidates", type=_positive_integer, default=20)
    qasper_parser.add_argument("--rrf-constant", type=_positive_integer, default=60)
    qasper_parser.add_argument(
        "--index-db",
        type=Path,
        help="Optional shared corpus/index database across retrieval experiments.",
    )
    _embedding_options(qasper_parser)
    qasper_parser.add_argument(
        "--resume", action="store_true", help="Reuse verified completed question records."
    )
    qasper_parser.add_argument(
        "--retry-errors", action="store_true", help="With --resume, retry recorded failures."
    )

    model_parser = subparsers.add_parser("model", help="Explicitly prepare local model files.")
    model_commands = model_parser.add_subparsers(dest="model_action", required=True)
    prepare_parser = model_commands.add_parser(
        "prepare-embedding", help="Download and verify the pinned Qwen safetensors snapshot."
    )
    prepare_parser.add_argument("--model-dir", type=Path, default=None)
    native_prepare_parser = model_commands.add_parser(
        "prepare-native-embedding", help="Download and verify the pinned public F16 GGUF."
    )
    native_prepare_parser.add_argument("--embedding-gguf", type=Path)
    docling_prepare_parser = model_commands.add_parser(
        "prepare-docling", help="Fetch and verify pinned local Docling layout/table assets."
    )
    docling_prepare_parser.add_argument("--artifacts", type=Path)

    ask_parser = subparsers.add_parser(
        "ask", help="Answer from one document's existing index with source-verified quotations."
    )
    ask_parser.add_argument("document_id", help="Document ID returned by ingest.")
    ask_parser.add_argument("question", help="Question; quote text containing spaces.")
    ask_parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default=None)
    ask_parser.add_argument("--rerank", action="store_true", help="Rerank native GPU candidates.")
    ask_parser.add_argument("--top-k", type=_positive_integer, default=FORMAL_RETRIEVAL.final_k)
    ask_parser.add_argument(
        "--rewrite-query",
        action="store_true",
        help="Opt in to lexical keyword expansion; dense keeps the original question.",
    )
    ask_parser.add_argument(
        "--route",
        choices=("auto", "fixed", "agent"),
        default="auto",
        help=_ROUTE_HELP,
    )
    ask_parser.add_argument("--answer-format", choices=("general", "short"), default="general")
    _embedding_options(ask_parser)
    ask_parser.add_argument(
        "--native-gpu",
        action="store_true",
        help=(
            "Use the pinned native GPU embedding model for one dense/hybrid retrieval, "
            "then close it before generation; bounded Agent follow-up searches may co-reside "
            "with generation under the combined 24,000 MiB limit."
        ),
    )
    ask_parser.add_argument("--embedding-gguf", type=Path)

    explain_parser = subparsers.add_parser(
        "explain", help="Explain a stored block and bounded neighbors without retrieval."
    )
    explain_parser.add_argument("document_id", help="Document ID returned by ingest.")
    explain_parser.add_argument("block_id", help="Source block ID returned by search.")
    explain_parser.add_argument(
        "--question",
        default="Explain this passage precisely, preserving its qualifications.",
    )
    explain_parser.add_argument("--neighbors", type=int, choices=(0, 1, 2), default=1)
    for answer_parser in (ask_parser, explain_parser):
        answer_parser.add_argument("--db", type=Path, default=_DEFAULT_DATABASE)
        _generation_options(answer_parser)
    ask_parser.set_defaults(output_tokens=3072)
    explain_parser.set_defaults(output_tokens=2048)

    serve_parser = subparsers.add_parser(
        "serve", help="Run one bound document's local-only API with a bounded model worker."
    )
    _document_service_options(serve_parser)
    serve_parser.add_argument("--port", type=_positive_integer, default=8765)
    mcp_parser = subparsers.add_parser("mcp", help="Serve one document over MCP stdio.")
    _document_service_options(mcp_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _run_doctor(args.json)
    if args.command == "ingest":
        return _run_ingest(args)
    if args.command == "show":
        return _run_show(args)
    if args.command == "source":
        return _run_source(args)
    if args.command in ("index", "search"):
        return _run_retrieval(args)
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "model":
        return _run_model(args)
    if args.command in ("ask", "explain"):
        return _run_answer(args)
    if args.command == "mcp":
        return _run_mcp(args)
    if args.command == "serve":
        return _run_serve(args)
    if args.command is None:
        parser.print_help()
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
