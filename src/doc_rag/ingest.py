"""Text and text-based PDF parsing for the first document-ingestion slice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

from doc_rag.models import DocumentMetadata, ParsedUnit


class DocumentIngestError(Exception):
    """A user-facing input or parsing error without document contents or paths."""


@dataclass(frozen=True)
class PreparedSource:
    document: DocumentMetadata
    path: Path

    def iter_units(self) -> Iterator[ParsedUnit]:
        if self.document.parser_name == "pdfplumber":
            yield from _iter_pdf_units(self.path, self.document)
        else:
            yield from _iter_text_units(self.path, self.document)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DocumentIngestError(f"cannot read input file ({type(exc).__name__})") from exc
    return digest.hexdigest()


def _document_id(source_sha256: str, parser_name: str, parser_version: str) -> str:
    identity = f"{source_sha256}\0{parser_name}\0{parser_version}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def prepare_source(
    source_path: str | Path,
    *,
    language: str | None = None,
    usage_scope: str = "not-recorded",
) -> PreparedSource:
    """Validate an input and capture stable identity before opening the database."""
    path = Path(source_path).expanduser()
    if not path.is_file():
        raise DocumentIngestError("input must be an existing regular file")

    suffix = path.suffix.lower()
    page_count: int | None = None
    source_sha256 = _sha256_file(path)
    if suffix == ".txt":
        media_type = "text/plain"
        parser_name = "plain_text"
        parser_version = "utf8-paragraph-v1"
    elif suffix in {".md", ".markdown"}:
        media_type = "text/markdown"
        parser_name = "markdown"
        parser_version = "utf8-paragraph-v1"
    elif suffix == ".pdf":
        media_type = "application/pdf"
        parser_name = "pdfplumber"
        try:
            import pdfplumber

            parser_version = (
                f"pdfplumber-{version('pdfplumber')};extract-v1;"
                "x-tolerance-3;y-tolerance-3;layout-false"
            )
            with pdfplumber.open(path) as pdf:
                page_count = len(pdf.pages)
            if _sha256_file(path) != source_sha256:
                raise DocumentIngestError(
                    "input changed while it was being inspected; retry the import"
                )
        except DocumentIngestError:
            raise
        except MemoryError:
            raise
        except Exception as exc:
            raise DocumentIngestError(f"cannot open PDF ({type(exc).__name__})") from exc
    else:
        raise DocumentIngestError("supported input types are .txt, .md, .markdown, and .pdf")

    document_id = _document_id(source_sha256, parser_name, parser_version)
    document = DocumentMetadata(
        document_id=document_id,
        source_sha256=source_sha256,
        source_name=path.name,
        media_type=media_type,
        parser_name=parser_name,
        parser_version=parser_version,
        language=language,
        usage_scope=usage_scope,
        page_count=page_count,
    )
    return PreparedSource(document=document, path=path)


def _iter_text_units(path: Path, document: DocumentMetadata) -> Iterator[ParsedUnit]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            text = source.read()
    except UnicodeDecodeError as exc:
        raise DocumentIngestError("text input must be valid UTF-8") from exc
    except OSError as exc:
        raise DocumentIngestError(f"cannot read text input ({type(exc).__name__})") from exc

    if _sha256_file(path) != document.source_sha256:
        raise DocumentIngestError("input changed while it was being read; retry the import")

    if text.strip():
        yield ParsedUnit(unit_index=1, page_number=None, text=text, status="ok")
    else:
        yield ParsedUnit(unit_index=1, page_number=None, text="", status="no_text")


def _iter_pdf_units(path: Path, document: DocumentMetadata) -> Iterator[ParsedUnit]:
    import pdfplumber

    try:
        with pdfplumber.open(path) as pdf:
            for unit_index, page in enumerate(pdf.pages, start=1):
                try:
                    text = (
                        page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=False,
                        )
                        or ""
                    )
                except MemoryError:
                    raise
                except Exception as exc:
                    yield ParsedUnit(
                        unit_index=unit_index,
                        page_number=unit_index,
                        text="",
                        status="error",
                        error_type=type(exc).__name__,
                    )
                    continue

                if text.strip():
                    yield ParsedUnit(
                        unit_index=unit_index,
                        page_number=unit_index,
                        text=text,
                        status="ok",
                    )
                else:
                    yield ParsedUnit(
                        unit_index=unit_index,
                        page_number=unit_index,
                        text="",
                        status="no_text",
                    )
    except MemoryError:
        raise
    except DocumentIngestError:
        raise
    except Exception as exc:
        raise DocumentIngestError(f"cannot extract PDF pages ({type(exc).__name__})") from exc

    if _sha256_file(path) != document.source_sha256:
        raise DocumentIngestError("input changed while it was being read; retry the import")
