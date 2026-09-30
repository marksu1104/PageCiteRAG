"""SQLite persistence for source metadata, extracted units, and text blocks."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from doc_rag.models import (
    Block,
    DocumentMetadata,
    DocumentRecord,
    DocumentStatus,
    IngestResult,
    ParsedUnit,
    UnitSummary,
)
from doc_rag.text import iter_blocks

_SCHEMA_VERSION = 1


class StoreError(RuntimeError):
    """The database is not a compatible doc-rag store."""


class SQLiteDocumentStore:
    """A small local store with atomic, idempotent document ingestion."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

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
                    "database already contains tables but is not a doc-rag store; "
                    "choose a new database file"
                )
            if version not in (0, _SCHEMA_VERSION):
                raise StoreError(f"database schema version {version} is not supported")

            if version == 0:
                self._create_schema(connection)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            else:
                required = {"documents", "document_units", "blocks"}
                if not required.issubset(tables):
                    raise StoreError("doc-rag database schema is incomplete")

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
    ) -> IngestResult:
        """Store one parsed source atomically; repeated identical imports are no-ops."""
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

            for expected_index, unit in enumerate(iterator, start=1):
                if unit.unit_index != expected_index:
                    raise StoreError("parser returned a non-sequential unit index")
                if document.media_type == "application/pdf":
                    if unit.page_number != unit.unit_index:
                        raise StoreError("PDF parser returned a mismatched page number")
                elif unit.page_number is not None:
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

                for block in iter_blocks(document, unit, next_ordinal):
                    self._insert_block(connection, block)
                    block_count += 1
                    next_ordinal = block.ordinal + 1

            if document.media_type == "application/pdf" and unit_count != document.page_count:
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

    def get_blocks(self, document_id: str) -> list[Block]:
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM blocks WHERE document_id = ? ORDER BY ordinal",
                (document_id,),
            ).fetchall()
        return [Block.model_validate(dict(row)) for row in rows]

    def count_documents(self) -> int:
        with self._read_connection() as connection:
            return connection.execute("SELECT count(*) FROM documents").fetchone()[0]
