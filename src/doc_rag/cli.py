"""Small CPU-only command-line entry point."""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from doc_rag.ingest import DocumentIngestError, prepare_source
from doc_rag.store import SQLiteDocumentStore, StoreError

_DISTRIBUTION_NAME = "doc-rag"
_DEFAULT_DATABASE = Path("database.sqlite3")


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
        prepared = prepare_source(
            args.source,
            language=args.language,
            usage_scope=args.usage_scope,
        )
        store = SQLiteDocumentStore(args.db)
        result = store.ingest(prepared.document, prepared.iter_units())
    except (DocumentIngestError, StoreError, OSError, sqlite3.Error, ValidationError) as exc:
        print(f"doc-rag: ingest failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 1 if result.document.status == "failed" else 0


def _run_show(args: argparse.Namespace) -> int:
    try:
        store = SQLiteDocumentStore(args.db)
        document = store.get_document(args.document_id)
        if document is None:
            print("doc-rag: document was not found", file=sys.stderr)
            return 1

        if args.unit is None:
            report = {
                "document": document.model_dump(mode="json"),
                "units": [
                    unit.model_dump(mode="json")
                    for unit in store.get_unit_summaries(args.document_id)
                ],
            }
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0

        unit = store.get_unit(args.document_id, args.unit)
        if unit is None:
            print("doc-rag: source unit was not found", file=sys.stderr)
            return 1
        if unit.status == "error":
            print(
                f"doc-rag: source unit could not be extracted ({unit.error_type})",
                file=sys.stderr,
            )
            return 1
        if unit.status == "no_text":
            print("doc-rag: this unit has no extractable text", file=sys.stderr)
            return 0

        sys.stdout.write(unit.text)
        return 0
    except (StoreError, OSError, sqlite3.Error, ValidationError) as exc:
        print(f"doc-rag: read failed: {exc}", file=sys.stderr)
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
    from doc_rag.retrieval import BM25Retriever, RetrievalError, build_bm25_index

    try:
        if not args.db.expanduser().is_file():
            raise RetrievalError("database was not found; ingest a document first")
        store = SQLiteDocumentStore(args.db)
        if args.command == "index":
            result = build_bm25_index(store, args.document_id)
        else:
            retriever = BM25Retriever(store, args.document_id)
            result = retriever.search(args.query, top_k=args.top_k)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
        return 0
    except (RetrievalError, StoreError, OSError, sqlite3.Error, ValidationError) as exc:
        print(f"doc-rag: {args.command} failed: {exc}", file=sys.stderr)
        return 1


def _run_eval(args: argparse.Namespace) -> int:
    from doc_rag.qasper import QasperError, download_qasper

    try:
        if args.eval_action == "download-qasper":
            report = download_qasper(args.data_dir)
        else:
            from doc_rag.evaluation import EvaluationError, run_qasper

            try:
                report = run_qasper(
                    args.data_dir,
                    args.output_dir,
                    dev_questions=args.dev_questions,
                    validation_questions=args.validation_questions,
                    seed=args.seed,
                    resume=args.resume,
                    retry_errors=args.retry_errors,
                )
            except EvaluationError as exc:
                print(f"doc-rag: evaluation failed: {exc}", file=sys.stderr)
                return 1
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        errors = sum(row["errors"] for row in report.get("results", {}).values())
        return 1 if errors else 0
    except (QasperError, StoreError, OSError, sqlite3.Error, ValidationError) as exc:
        print(f"doc-rag: evaluation failed: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-rag",
        description="Local-first document analysis and retrieval experiments.",
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

    index_parser = subparsers.add_parser(
        "index", help="Build and activate a local BM25 index for one stored document."
    )
    search_parser = subparsers.add_parser(
        "search", help="Search one document's existing BM25 index and return source blocks."
    )
    for retrieval_parser in (index_parser, search_parser):
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
    qasper_parser.add_argument(
        "--resume", action="store_true", help="Reuse verified completed question records."
    )
    qasper_parser.add_argument(
        "--retry-errors", action="store_true", help="With --resume, retry recorded failures."
    )

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
    if args.command in ("index", "search"):
        return _run_retrieval(args)
    if args.command == "eval":
        return _run_eval(args)
    if args.command is None:
        parser.print_help()
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
