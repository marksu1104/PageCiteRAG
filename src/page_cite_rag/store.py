"""SQLite persistence for source metadata, extracted units, and text blocks."""

from __future__ import annotations

import json
import sqlite3
from bisect import bisect_right
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from page_cite_rag.models import (
    Block,
    DocumentMetadata,
    DocumentRecord,
    DocumentStatus,
    IngestResult,
    ParsedUnit,
    UnitSummary,
)
from page_cite_rag.source_outline import OUTLINE_KIND, outline_from_markdown
from page_cite_rag.text import iter_blocks

_SCHEMA_VERSION = 2
_MAX_SQLITE_INTEGER = (1 << 63) - 1
SOURCE_LOCATIONS_KIND = "source-locations-v1"


class StoreError(RuntimeError):
    """The database is not a compatible PageCiteRAG store."""


class SQLiteDocumentStore:
    """A small local store with atomic, idempotent document ingestion."""

    def __init__(self, database_path: str | Path, *, read_only: bool = False) -> None:
        if type(read_only) is not bool:
            raise ValueError("read_only must be a boolean")
        self.database_path = Path(database_path).expanduser()
        self._read_only = read_only
        if read_only:
            if not self.database_path.is_file():
                raise StoreError("read-only database was not found")
            self._validate_read_only_schema()
        else:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = (
            sqlite3.connect(
                self.database_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30
            )
            if self._read_only
            else sqlite3.connect(self.database_path, timeout=30)
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _validate_read_only_schema(self) -> None:
        with self._read_connection() as connection:
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        if schema_version != _SCHEMA_VERSION or not {
            "documents",
            "document_units",
            "blocks",
            "retrieval_indexes",
        }.issubset(tables):
            raise StoreError(
                "read-only answering requires an existing schema v2 store; "
                "prepare or migrate it explicitly using ingestion"
            )

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }

            if version == 0 and tables:
                raise StoreError(
                    "database already contains tables but is not a PageCiteRAG store; "
                    "choose a new database file"
                )
            if version not in (0, 1, _SCHEMA_VERSION):
                raise StoreError(f"database schema version {version} is not supported")

            if version == 0:
                self._create_schema(connection)
            else:
                required = {"documents", "document_units", "blocks"}
                if version == _SCHEMA_VERSION:
                    required.add("retrieval_indexes")
                if not required.issubset(tables):
                    raise StoreError("PageCiteRAG database schema is incomplete")

            if version in (0, 1):
                connection.execute(
                    "CREATE TABLE retrieval_indexes ("
                    "document_id TEXT NOT NULL, kind TEXT NOT NULL, manifest TEXT NOT NULL, "
                    "PRIMARY KEY (document_id, kind), "
                    "FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE)"
                )
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                source_sha256 TEXT NOT NULL,
                source_name TEXT NOT NULL,
                media_type TEXT NOT NULL,
                parser_name TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                language TEXT,
                usage_scope TEXT NOT NULL,
                page_count INTEGER,
                status TEXT NOT NULL CHECK (status IN ('ready', 'partial', 'no_text', 'failed')),
                unit_count INTEGER NOT NULL,
                block_count INTEGER NOT NULL,
                imported_at TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE document_units (
                document_id TEXT NOT NULL,
                unit_index INTEGER NOT NULL,
                page_number INTEGER,
                text TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('ok', 'no_text', 'error')),
                error_type TEXT,
                PRIMARY KEY (document_id, unit_index),
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
                    ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE blocks (
                document_id TEXT NOT NULL,
                block_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                unit_index INTEGER NOT NULL,
                page_number INTEGER,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (document_id, block_id),
                UNIQUE (document_id, ordinal),
                FOREIGN KEY (document_id, unit_index)
                    REFERENCES document_units(document_id, unit_index)
                    ON DELETE CASCADE
            );
            """,
            """
            CREATE INDEX blocks_by_document_order
                ON blocks(document_id, ordinal);
            """,
        )
        for statement in statements:
            connection.execute(statement)

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord.model_validate(dict(row))

    def ingest(
        self,
        document: DocumentMetadata,
        units: Iterable[ParsedUnit],
        *,
        block_factory: Callable[[DocumentMetadata, ParsedUnit, int], Iterable[Block]] = iter_blocks,
        source_manifests: dict[str, str] | None = None,
    ) -> IngestResult:
        """Store one parsed source atomically; repeated identical imports are no-ops."""
        if getattr(self, "_read_only", False):
            raise StoreError("this document store is read-only")
        connection = self._connect()
        iterator = iter(units)
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document.document_id,),
            ).fetchone()
            if existing is not None:
                metadata_fields = (
                    "source_sha256",
                    "source_name",
                    "media_type",
                    "parser_name",
                    "parser_version",
                    "language",
                    "usage_scope",
                    "page_count",
                )
                if any(existing[field] != getattr(document, field) for field in metadata_fields):
                    raise StoreError(
                        "this source and parser version already exist with different metadata; "
                        "metadata updates are not supported"
                    )
                connection.rollback()
                return self._make_result(connection, self._document_from_row(existing), True)

            imported_at = datetime.now(timezone.utc)
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_sha256, source_name, media_type,
                    parser_name, parser_version, language, usage_scope,
                    page_count, status, unit_count, block_count, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'failed', 0, 0, ?)
                """,
                (
                    document.document_id,
                    document.source_sha256,
                    document.source_name,
                    document.media_type,
                    document.parser_name,
                    document.parser_version,
                    document.language,
                    document.usage_scope,
                    document.page_count,
                    imported_at.isoformat(),
                ),
            )

            unit_count = 0
            block_count = 0
            failed_units = 0
            has_empty_units = False
            next_ordinal = 1
            markdown_unit: ParsedUnit | None = None
            markdown_blocks: list[Block] = []

            for expected_index, unit in enumerate(iterator, start=1):
                if unit.unit_index != expected_index:
                    raise StoreError("parser returned a non-sequential unit index")
                if document.parser_name == "pdfplumber":
                    if unit.page_number != unit.unit_index:
                        raise StoreError("PDF parser returned a mismatched page number")
                elif document.parser_name != "docling" and unit.page_number is not None:
                    raise StoreError("text parsers must not fabricate PDF page numbers")

                connection.execute(
                    """
                    INSERT INTO document_units (
                        document_id, unit_index, page_number, text, status, error_type
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.document_id,
                        unit.unit_index,
                        unit.page_number,
                        unit.text,
                        unit.status,
                        unit.error_type,
                    ),
                )
                unit_count += 1
                if unit.status == "no_text":
                    has_empty_units = True
                elif unit.status == "error":
                    failed_units += 1
                if document.media_type == "text/markdown":
                    markdown_unit = unit

                line_starts = [0]
                for line in unit.text.splitlines(keepends=True):
                    line_starts.append(line_starts[-1] + len(line))
                for block in block_factory(document, unit, next_ordinal):
                    if (
                        block.document_id != document.document_id
                        or block.unit_index != unit.unit_index
                        or block.page_number != unit.page_number
                        or block.ordinal != next_ordinal
                        or unit.text[block.char_start : block.char_end] != block.text
                        or block.char_end > len(unit.text)
                        or block.line_start != bisect_right(line_starts, block.char_start)
                        or block.line_end != bisect_right(line_starts, block.char_end - 1)
                        or unit.status != "ok"
                    ):
                        raise StoreError("parser returned a block outside its source unit")
                    self._insert_block(connection, block)
                    if document.media_type == "text/markdown":
                        markdown_blocks.append(block)
                    block_count += 1
                    next_ordinal = block.ordinal + 1

            if document.parser_name == "pdfplumber" and unit_count != document.page_count:
                raise StoreError("PDF parser page count did not match the source metadata")
            if document.media_type != "application/pdf" and unit_count != 1:
                raise StoreError("text parser must return exactly one source unit")

            status: DocumentStatus
            if block_count == 0:
                status = "failed" if failed_units else "no_text"
            elif failed_units or has_empty_units:
                status = "partial"
            else:
                status = "ready"

            connection.execute(
                """
                UPDATE documents
                SET status = ?, unit_count = ?, block_count = ?
                WHERE document_id = ?
                """,
                (status, unit_count, block_count, document.document_id),
            )
            if markdown_unit is not None:
                outline = outline_from_markdown(document, markdown_unit, markdown_blocks)
                if outline is not None:
                    connection.execute(
                        "INSERT INTO retrieval_indexes (document_id, kind, manifest) "
                        "VALUES (?, ?, ?)",
                        (document.document_id, OUTLINE_KIND, outline.model_dump_json()),
                    )
            for kind, manifest in (source_manifests or {}).items():
                if not kind or not isinstance(manifest, str) or not manifest:
                    raise StoreError("source manifest is invalid")
                connection.execute(
                    "INSERT INTO retrieval_indexes (document_id, kind, manifest) VALUES (?, ?, ?)",
                    (document.document_id, kind, manifest),
                )
            connection.commit()
            record = DocumentRecord(
                **document.model_dump(),
                status=status,
                unit_count=unit_count,
                block_count=block_count,
                imported_at=imported_at,
            )
            return self._make_result(connection, record, False)
        except BaseException:
            connection.rollback()
            raise
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
            connection.close()

    @staticmethod
    def _insert_block(connection: sqlite3.Connection, block: Block) -> None:
        connection.execute(
            """
            INSERT INTO blocks (
                document_id, block_id, ordinal, unit_index, page_number,
                line_start, line_end, char_start, char_end, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                block.document_id,
                block.block_id,
                block.ordinal,
                block.unit_index,
                block.page_number,
                block.line_start,
                block.line_end,
                block.char_start,
                block.char_end,
                block.text,
            ),
        )

    @staticmethod
    def _make_result(
        connection: sqlite3.Connection,
        document: DocumentRecord,
        duplicate: bool,
    ) -> IngestResult:
        rows = connection.execute(
            """
            SELECT unit_index, status FROM document_units
            WHERE document_id = ? ORDER BY unit_index
            """,
            (document.document_id,),
        ).fetchall()
        return IngestResult(
            document=document,
            duplicate=duplicate,
            empty_unit_indexes=tuple(
                row["unit_index"] for row in rows if row["status"] == "no_text"
            ),
            error_unit_indexes=tuple(row["unit_index"] for row in rows if row["status"] == "error"),
        )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._document_from_row(row) if row is not None else None

    def get_unit(self, document_id: str, unit_index: int) -> ParsedUnit | None:
        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT unit_index, page_number, text, status, error_type
                FROM document_units WHERE document_id = ? AND unit_index = ?
                """,
                (document_id, unit_index),
            ).fetchone()
        return ParsedUnit.model_validate(dict(row)) if row is not None else None

    def get_unit_summaries(self, document_id: str) -> list[UnitSummary]:
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT unit_index, page_number, status, error_type
                FROM document_units WHERE document_id = ? ORDER BY unit_index
                """,
                (document_id,),
            ).fetchall()
        return [UnitSummary.model_validate(dict(row)) for row in rows]

    def get_unit_summaries_page(
        self, document_id: str, *, after_unit_index: int = 0, limit: int = 33
    ) -> list[UnitSummary]:
        """Page the source-unit inventory; no extracted text enters the listing."""
        if type(after_unit_index) is not int or not 0 <= after_unit_index <= _MAX_SQLITE_INTEGER:
            raise ValueError("after_unit_index must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 33:
            raise ValueError("limit must be between one and 33")
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT unit_index, page_number, status, error_type FROM document_units "
                "WHERE document_id = ? AND unit_index > ? ORDER BY unit_index LIMIT ?",
                (document_id, after_unit_index, limit),
            ).fetchall()
        return [UnitSummary.model_validate(dict(row)) for row in rows]

    def get_blocks(self, document_id: str) -> list[Block]:
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM blocks WHERE document_id = ? ORDER BY ordinal",
                (document_id,),
            ).fetchall()
        return [Block.model_validate(dict(row)) for row in rows]

    def get_unit_blocks(
        self, document_id: str, unit_index: int, *, after_ordinal: int = 0, limit: int = 5
    ) -> list[Block]:
        """Page original blocks in one source unit without loading the whole document."""
        if type(unit_index) is not int or not 1 <= unit_index <= _MAX_SQLITE_INTEGER:
            raise ValueError("unit_index must be a positive integer")
        if type(after_ordinal) is not int or not 0 <= after_ordinal <= _MAX_SQLITE_INTEGER:
            raise ValueError("after_ordinal must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("limit must be between one and five")
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM blocks WHERE document_id = ? AND unit_index = ? "
                "AND ordinal > ? ORDER BY ordinal LIMIT ?",
                (document_id, unit_index, after_ordinal, limit),
            ).fetchall()
        return [Block.model_validate(dict(row)) for row in rows]

    def get_ordinal_blocks(
        self,
        document_id: str,
        start_ordinal: int,
        end_ordinal: int,
        *,
        after_ordinal: int = 0,
        limit: int = 5,
    ) -> list[Block]:
        """Page a known section's contiguous source-block range."""
        if (
            type(start_ordinal) is not int
            or type(end_ordinal) is not int
            or not 1 <= start_ordinal <= end_ordinal <= _MAX_SQLITE_INTEGER
        ):
            raise ValueError("invalid source ordinal range")
        if type(after_ordinal) is not int or not 0 <= after_ordinal <= _MAX_SQLITE_INTEGER:
            raise ValueError("after_ordinal must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("limit must be between one and five")
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM blocks WHERE document_id = ? AND ordinal BETWEEN ? AND ? "
                "AND ordinal > ? ORDER BY ordinal LIMIT ?",
                (document_id, start_ordinal, end_ordinal, after_ordinal, limit),
            ).fetchall()
        return [Block.model_validate(dict(row)) for row in rows]

    def get_block(self, document_id: str, block_id: str) -> Block | None:
        """Read one authoritative block within the caller's explicit document scope."""
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM blocks WHERE document_id = ? AND block_id = ?",
                (document_id, block_id),
            ).fetchone()
        return Block.model_validate(dict(row)) if row is not None else None

    def get_neighbor_blocks(
        self, document_id: str, ordinal: int, *, radius: int = 1
    ) -> list[Block]:
        """Read a bounded ordinal neighborhood, never crossing the document scope."""
        if type(ordinal) is not int or ordinal < 1:
            raise ValueError("ordinal must be a positive integer")
        if type(radius) is not int or not 0 <= radius <= 2:
            raise ValueError("neighbor radius must be an integer between zero and two")
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM blocks WHERE document_id = ? AND ordinal BETWEEN ? AND ? "
                "ORDER BY ordinal",
                (document_id, max(1, ordinal - radius), ordinal + radius),
            ).fetchall()
        return [Block.model_validate(dict(row)) for row in rows]

    def count_documents(self) -> int:
        with self._read_connection() as connection:
            return connection.execute("SELECT count(*) FROM documents").fetchone()[0]

    def get_retrieval_manifest(self, document_id: str, kind: str) -> str | None:
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT manifest FROM retrieval_indexes WHERE document_id = ? AND kind = ?",
                (document_id, kind),
            ).fetchone()
        return row["manifest"] if row is not None else None

    def get_source_location(self, document_id: str, block_id: str) -> dict | None:
        """Read optional immutable parser provenance saved for one source block."""
        raw = self.get_retrieval_manifest(document_id, SOURCE_LOCATIONS_KIND)
        if raw is None:
            return None
        try:
            manifest = json.loads(raw)
            document = self.get_document(document_id)
            locations = manifest["locations"]
            if (
                manifest.get("version") != 1
                or manifest.get("document_id") != document_id
                or document is None
                or manifest.get("source_sha256") != document.source_sha256
                or not isinstance(locations, dict)
            ):
                raise ValueError("source location identity mismatch")
            location = locations.get(block_id)
            if location is None:
                return None
            if not isinstance(location, dict):
                raise ValueError("invalid source location")
            return json.loads(json.dumps(location, allow_nan=False))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            raise StoreError("saved source location metadata is invalid") from None

    def save_retrieval_manifest(self, document_id: str, kind: str, manifest: str) -> None:
        """Activate an already-written index; failure preserves the previous manifest."""
        if getattr(self, "_read_only", False):
            raise StoreError("this document store is read-only")
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO retrieval_indexes (document_id, kind, manifest) VALUES (?, ?, ?) "
                    "ON CONFLICT(document_id, kind) DO UPDATE SET manifest = excluded.manifest",
                    (document_id, kind, manifest),
                )
        finally:
            connection.close()

    def save_retrieval_manifest_once(self, document_id: str, kind: str, manifest: str) -> None:
        """Record immutable source-derived metadata; refuse conflicting reimports."""
        if getattr(self, "_read_only", False):
            raise StoreError("this document store is read-only")
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO retrieval_indexes (document_id, kind, manifest) VALUES (?, ?, ?) "
                    "ON CONFLICT(document_id, kind) DO NOTHING",
                    (document_id, kind, manifest),
                )
                row = connection.execute(
                    "SELECT manifest FROM retrieval_indexes WHERE document_id = ? AND kind = ?",
                    (document_id, kind),
                ).fetchone()
                if row is None or row["manifest"] != manifest:
                    raise StoreError("source outline conflicts with existing document metadata")
        finally:
            connection.close()
