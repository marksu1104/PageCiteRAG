"""Local-first CLI; model resources are loaded only for explicit model commands."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from page_cite_rag.ingest import DocumentIngestError, prepare_source
from page_cite_rag.store import SQLiteDocumentStore, StoreError

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
        if args.parser == "docling":
            from page_cite_rag.docling_document import convert_pdf
            from page_cite_rag.embedding import DEFAULT_MODEL_DIR

            if args.pdf_reading_order != "geometric" or args.docling_artifacts is None:
                raise DocumentIngestError(
                    "Docling requires --docling-artifacts and does not use --pdf-reading-order"
                )
            bundle = convert_pdf(
                args.source,
                artifacts=args.docling_artifacts,
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
        )
        store = SQLiteDocumentStore(args.db)
        result = store.ingest(prepared.document, prepared.iter_units())
    except (
        DocumentIngestError,
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
                    max_gpu_temperature_c=args.gpu_temperature_limit or 80,
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
                        max_gpu_temperature_c=args.gpu_temperature_limit or 80,
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
                    gpu_temperature_limit_c=args.gpu_temperature_limit or 80,
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


def _generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("ollama", "llama"), default="ollama")
    parser.add_argument(
        "--token-budget",
        choices=("conservative", "exact"),
        default="conservative",
        help="Exact fixed-template token counting is opt-in and requires the llama backend.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-uuid", help="One explicit GPU UUID; only llama CUDA supports it.")
    parser.add_argument(
        "--gpu-temperature-limit",
        type=_positive_integer,
        help="Explicit llama CUDA temperature guard, at most 88 C; default 80 C.",
    )
    parser.add_argument("--bundle-dir", type=Path, help="Explicitly prepared llama runtime/model.")
    parser.add_argument("--llm-host", default="http://127.0.0.1:11435")
    parser.add_argument("--context-window", type=_positive_integer, default=16384)
    parser.add_argument("--output-tokens", type=_positive_integer, default=1024)
    parser.add_argument("--generation-timeout", type=float, default=120.0)
    parser.add_argument("--llm-threads", type=_positive_integer, default=2)
    parser.add_argument("--language", choices=("en", "zh-Hant", "auto"), default="en")
    parser.add_argument("--format", choices=("text", "json"), default="text")


def _generation_config(args: argparse.Namespace):
    # Validate choices before loading a large embedding model or starting a server.
    from page_cite_rag.generation import GenerationConfig

    if args.token_budget == "exact" and args.backend != "llama":
        raise ValueError("exact token budget requires llama")

    if args.gpu_temperature_limit is not None and (
        args.backend != "llama" or args.device != "cuda"
    ):
        raise ValueError("--gpu-temperature-limit requires explicit llama CUDA")

    if args.backend == "ollama":
        if args.device != "cpu" or args.gpu_uuid is not None or args.bundle_dir is not None:
            raise ValueError("Ollama uses the retained CPU route; CUDA/bundle options need llama")
        return GenerationConfig(
            host=args.llm_host,
            num_ctx=args.context_window,
            num_predict=args.output_tokens,
            timeout_seconds=args.generation_timeout,
            num_thread=args.llm_threads,
            num_gpu=0,
        )
    from page_cite_rag.llama import DEFAULT_BUNDLE, LlamaConfig

    if args.llm_host != "http://127.0.0.1:11435":
        raise ValueError("llama owns its loopback server; --llm-host is only for Ollama")
    return LlamaConfig(
        bundle_dir=args.bundle_dir or DEFAULT_BUNDLE,
        device=args.device,
        gpu_uuid=args.gpu_uuid,
        num_ctx=args.context_window,
        num_predict=args.output_tokens,
        timeout_seconds=args.generation_timeout,
        num_thread=args.llm_threads,
        max_gpu_temperature_c=args.gpu_temperature_limit or 80,
    )


def _open_generation_client(args: argparse.Namespace, config):
    if args.backend == "ollama":
        from page_cite_rag.generation import OllamaModelClient

        return OllamaModelClient(config)
    from page_cite_rag.llama import LlamaModelClient

    if args.token_budget == "exact":
        return LlamaModelClient(config, budget_policy="exact")
    return LlamaModelClient(config)


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


def _print_answer(result, output_format: str) -> None:
    if output_format == "json":
        print(_terminal_safe(result.model_dump_json(indent=2)))
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


def _run_answer(args: argparse.Namespace) -> int:
    from page_cite_rag.application import (
        ask_document,
        document_retriever,
        document_workflow,
        explain_source,
    )
    from page_cite_rag.generation import ModelError
    from page_cite_rag.retrieval import RetrievalError
    from page_cite_rag.workflows import WorkflowConfig

    client = None
    retrieval_session = None
    agent_enabled = getattr(args, "agent", False)
    delegate_readers = getattr(args, "delegate_readers", False)
    native_gpu = getattr(args, "native_gpu", False)
    try:
        if delegate_readers and not agent_enabled:
            raise ValueError("--delegate-readers requires --agent")
        if agent_enabled and (
            args.command != "ask"
            or args.backend != "llama"
            or args.token_budget != "exact"
            or args.rewrite_query
        ):
            raise ValueError(
                "--agent requires ask with --backend llama --token-budget exact "
                "and no query rewrite"
            )
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
            args.method = "hybrid" if docling_source else "bm25"
        if args.command == "ask":
            if native_gpu and (
                args.backend != "llama"
                or args.device != "cuda"
                or not args.gpu_uuid
                or args.method not in ("dense", "hybrid")
                or args.rewrite_query
            ):
                raise ValueError(
                    "--native-gpu requires dense/hybrid llama CUDA without query rewriting"
                )
            if args.rewrite_query and args.method == "dense":
                raise ValueError("query rewriting is lexical-only; use bm25 or hybrid")
            if native_gpu:
                from page_cite_rag.embedding import DEFAULT_MODEL_DIR
                from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH
                from page_cite_rag.gpu_retrieval import StagedGpuSearch

                retriever = StagedGpuSearch(
                    store,
                    args.document_id,
                    args.method,
                    args.embedding_gguf or DEFAULT_GGUF_PATH,
                    args.gpu_uuid,
                    tokenizer_directory=args.model_dir or DEFAULT_MODEL_DIR,
                    gpu_temperature_limit_c=args.gpu_temperature_limit or 80,
                )
                if agent_enabled:
                    retrieval_session = retriever.session()
                    retrieval_session.__enter__()
                lexical, dense = None, None
            elif docling_source:
                if args.method != "hybrid":
                    raise ValueError("Docling source requires --method hybrid")
                retriever = document_retriever(
                    store, args.document_id, "hybrid", encoder=_load_encoder(args)
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
        if native_gpu:
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
            if isinstance(getattr(retriever, "last_trace", None), dict):
                initial_gpu_search_trace = dict(retriever.last_trace)

        client = _open_generation_client(args, config)
        if args.command != "ask":
            retriever = None
        elif native_gpu:
            pass
        elif docling_source:
            if args.rewrite_query:
                from page_cite_rag.docling_retrieval import RewritingDoclingRetriever
                from page_cite_rag.query import QueryRewriter

                retriever = RewritingDoclingRetriever(retriever, QueryRewriter(client, config))
        else:
            if args.rewrite_query:
                from page_cite_rag.query import QueryRewriter
                from page_cite_rag.query_retrieval import RewritingRetriever

                retriever = RewritingRetriever(lexical, QueryRewriter(client, config), dense)
            elif args.method == "hybrid":
                from page_cite_rag.hybrid import HybridRetriever

                retriever = HybridRetriever(lexical, dense)
            else:
                retriever = lexical if lexical is not None else dense
        workflow = document_workflow(
            store,
            args.document_id,
            retriever,
            client,
            config,
            top_k=workflow_config.top_k,
            neighbor_blocks=workflow_config.neighbor_blocks,
        )
        if agent_enabled:
            from page_cite_rag.agent import local_chat_model, run_bounded_reading_agent
            from page_cite_rag.workflows import RetrievalTrace

            search_result, blocks = (
                (precomputed_search, precomputed_blocks)
                if precomputed_search is not None
                else workflow.search_sources(args.question)
            )
            blocks = workflow._context_blocks(search_result, blocks)
            with local_chat_model(client) as chat_model:
                agent_run = run_bounded_reading_agent(
                    workflow,
                    args.question,
                    args.language,
                    search_result,
                    blocks,
                    chat_model,
                    count_chat_input_tokens=client.count_chat_input_tokens,
                    delegated=delegate_readers,
                )
            if retrieval_session is not None:
                retrieval_session.__exit__(None, None, None)
                retrieval_session = None
            final_gpu_trace = getattr(retriever, "last_trace", None)
            agent_gpu_search_trace = (
                dict(final_gpu_trace) if native_gpu and isinstance(final_gpu_trace, dict) else None
            )
            result = workflow.answer_from_context(
                args.question,
                agent_run.context.blocks,
                language=args.language,
                context_groups=agent_run.context.context_groups,
            )
            result = result.model_copy(
                update={
                    "retrieval": RetrievalTrace(
                        method=search_result.method,
                        index_id=search_result.index_id,
                        query=search_result.query,
                        hit_count=len(search_result.hits),
                        elapsed_ms=search_result.elapsed_ms,
                        query_plan=workflow._query_plan(),
                        branches=search_result.branches,
                    )
                }
            )
            if args.format == "json":
                report = {
                    "answer": result.model_dump(mode="json"),
                    "agent_trace": agent_run.trace,
                }
                if initial_gpu_search_trace is not None:
                    report["initial_gpu_search_trace"] = initial_gpu_search_trace
                if agent_gpu_search_trace is not None:
                    report["agent_gpu_search_trace"] = agent_gpu_search_trace
                if (
                    args.backend == "llama"
                    and args.device == "cuda"
                    and callable(getattr(client, "diagnostics", None))
                ):
                    report["gpu_generation_diagnostics"] = client.diagnostics()
                print(
                    _terminal_safe(
                        json.dumps(
                            report,
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )
                    )
                )
            else:
                _print_answer(result, args.format)
                print(
                    "Agent trace: "
                    f"{agent_run.trace['model_calls_completed']} decision calls, "
                    f"{agent_run.trace['tool_calls_recorded']} read-only tool calls."
                )
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
            if args.format == "json" and (initial_gpu_search_trace or args.device == "cuda"):
                report = result.model_dump(mode="json")
                if initial_gpu_search_trace is not None:
                    report["gpu_retrieval_trace"] = initial_gpu_search_trace
                if (
                    args.backend == "llama"
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
                _print_answer(result, args.format)
        return 0  # Evidence insufficiency is a visible result, not a transport error.
    except (ModelError, RetrievalError, StoreError, OSError, sqlite3.Error, ValueError) as exc:
        # ValidationError/OS errors can contain raw user input or paths; don't dump them.
        detail = (
            "invalid configuration or local resource"
            if not isinstance(exc, (ModelError, RetrievalError))
            else str(exc)
        )
        print(f"page-cite-rag: {args.command} failed: {detail}", file=sys.stderr)
        return 1
    finally:
        if retrieval_session is not None:
            retrieval_session.__exit__(None, None, None)
        if client is not None:
            client.close()


def _run_serve(args: argparse.Namespace) -> int:
    """One bound document and one worker; never expose a network-facing listener."""
    from page_cite_rag.retrieval import RetrievalError

    try:
        api_token = os.environ.get("PAGE_CITE_RAG_API_TOKEN")
        if api_token is None or not api_token.strip():
            raise ValueError("PAGE_CITE_RAG_API_TOKEN must contain a local API secret")
        store = SQLiteDocumentStore(args.db, read_only=True)
        if args.method is None:
            record = store.get_document(args.document_id)
            args.method = (
                "hybrid"
                if record is not None
                and record.parser_name == "docling"
                and _has_native_docling_source(store, args.document_id)
                else "bm25"
            )
        if not 1 <= args.port <= 65535:
            raise ValueError("invalid loopback port")
        if args.backend != "llama" or args.token_budget != "exact":
            raise ValueError("local API requires llama with exact token budgeting")
        config = _generation_config(args)
        from page_cite_rag.api import create_app
        from page_cite_rag.service import DocumentService, ServiceConfig

        if args.embedding_backend == "native-gpu" and (
            args.method == "bm25" or args.device != "cuda" or not args.gpu_uuid
        ):
            raise ValueError("native GPU embedding needs dense/hybrid and CUDA generation")
        encoder = (
            _load_encoder(args)
            if args.method != "bm25" and args.embedding_backend == "cpu"
            else None
        )
        staged_search = None
        if args.embedding_backend == "native-gpu":
            from page_cite_rag.embedding import DEFAULT_MODEL_DIR
            from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH
            from page_cite_rag.gpu_retrieval import StagedGpuSearch

            staged_search = StagedGpuSearch(
                SQLiteDocumentStore(args.db, read_only=True),
                args.document_id,
                args.method,
                args.embedding_gguf or DEFAULT_GGUF_PATH,
                args.gpu_uuid,
                tokenizer_directory=args.model_dir or DEFAULT_MODEL_DIR,
            )
        runtime = DocumentService(
            ServiceConfig(
                database_path=args.db,
                document_id=args.document_id,
                method=args.method,
                top_k=args.top_k,
                queue_capacity=args.queue_capacity,
                idle_seconds=args.idle_seconds,
                request_timeout_seconds=args.request_timeout_seconds,
            ),
            config,
            lambda: _open_generation_client(args, config),
            encoder=encoder,
            staged_search=staged_search,
        )
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
    except (ImportError, ValueError, OSError, StoreError, RetrievalError) as exc:
        print(f"page-cite-rag: local service failed: {type(exc).__name__}", file=sys.stderr)
        return 1


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
        "--parser",
        choices=("pdfplumber", "docling"),
        default="pdfplumber",
        help="PDF parser; Docling requires prepared local models and its optional environment.",
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
            help="Explicit native GPU temperature guard, at most 88 C; default 80 C.",
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
    ask_parser.add_argument("--top-k", type=_positive_integer, default=10)
    ask_parser.add_argument(
        "--rewrite-query",
        action="store_true",
        help="Opt in to Chinese-to-English lexical query rewriting; dense keeps the original.",
    )
    ask_parser.add_argument(
        "--agent",
        action="store_true",
        help=(
            "Use a bounded local LangChain reading agent; requires --backend llama "
            "--token-budget exact and the optional agent extra."
        ),
    )
    ask_parser.add_argument(
        "--delegate-readers",
        action="store_true",
        help=(
            "Allow the reading agent to delegate up to two sequential, read-only tasks; "
            "requires --agent and shares its token/tool budgets."
        ),
    )
    _embedding_options(ask_parser)
    ask_parser.add_argument(
        "--native-gpu",
        action="store_true",
        help=(
            "Use the pinned native GPU embedding model for one dense/hybrid retrieval, "
            "then close it before generation; bounded Agent follow-up searches may co-reside "
            "with generation under the combined 12,000 MiB limit."
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
    ask_parser.set_defaults(output_tokens=1536)
    explain_parser.set_defaults(output_tokens=2048)

    serve_parser = subparsers.add_parser(
        "serve", help="Run one bound document's local-only API with a bounded model worker."
    )
    serve_parser.add_argument("document_id", help="The only document accessible to this API.")
    serve_parser.add_argument("--db", type=Path, default=_DEFAULT_DATABASE)
    serve_parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default=None)
    serve_parser.add_argument("--top-k", type=_positive_integer, default=10)
    serve_parser.add_argument("--port", type=_positive_integer, default=8765)
    serve_parser.add_argument("--queue-capacity", type=int, default=2)
    serve_parser.add_argument("--idle-seconds", type=float, default=300.0)
    serve_parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    serve_parser.add_argument("--embedding-backend", choices=("cpu", "native-gpu"), default="cpu")
    serve_parser.add_argument("--embedding-gguf", type=Path)
    _embedding_options(serve_parser)
    _generation_options(serve_parser)
    serve_parser.set_defaults(output_tokens=2048)
    serve_parser.set_defaults(backend="llama", token_budget="exact")

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
    if args.command == "serve":
        return _run_serve(args)
    if args.command is None:
        parser.print_help()
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
