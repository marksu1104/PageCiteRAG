from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess

import pytest
from pydantic import ValidationError

from doc_rag.cli import main
from doc_rag.ingest import DocumentIngestError, prepare_source
from doc_rag.models import DocumentMetadata, ParsedUnit
from doc_rag.store import SQLiteDocumentStore, StoreError


def _write_two_page_pdf(path) -> None:
    text_stream = b"BT /F1 12 Tf 72 720 Td (English PDF page one.) Tj ET\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 6 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << >> /Contents 7 0 R >>"
        ),
        (
            f"<< /Length {len(text_stream)} >>\nstream\n".encode("ascii")
            + text_stream
            + b"endstream"
        ),
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, payload in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(payload)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(b"xref\n0 8\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (f"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n").encode("ascii")
    )
    path.write_bytes(pdf)


@pytest.mark.parametrize(
    ("suffix", "expected_media_type", "expected_parser"),
    [
        (".txt", "text/plain", "plain_text"),
        (".md", "text/markdown", "markdown"),
        (".markdown", "text/markdown", "markdown"),
    ],
)
def test_text_ingestion_preserves_source_and_block_offsets(
    tmp_path, suffix, expected_media_type, expected_parser
) -> None:
    source_text = "Heading\r\n\r\nEnglish and 中文.\r\nSecond detail.\r\n\r\nFinal paragraph."
    source = tmp_path / f"sample{suffix}"
    source.write_bytes(source_text.encode("utf-8"))
    prepared = prepare_source(source, language="zh-Hant+en", usage_scope="synthetic-test")
    database = tmp_path / "documents.sqlite3"
    store = SQLiteDocumentStore(database)

    result = store.ingest(prepared.document, prepared.iter_units())
    reopened_store = SQLiteDocumentStore(database)
    stored_document = reopened_store.get_document(prepared.document.document_id)
    unit = reopened_store.get_unit(prepared.document.document_id, 1)
    blocks = reopened_store.get_blocks(prepared.document.document_id)

    assert result.document.status == "ready"
    assert result.duplicate is False
    assert stored_document is not None
    assert stored_document.media_type == expected_media_type
    assert stored_document.parser_name == expected_parser
    assert stored_document.language == "zh-Hant+en"
    assert stored_document.usage_scope == "synthetic-test"
    assert unit is not None and unit.text == source_text
    assert [block.line_start for block in blocks] == [1, 3, 6]
    assert [block.line_end for block in blocks] == [1, 4, 6]
    assert all(unit.text[block.char_start : block.char_end] == block.text for block in blocks)

    duplicate = reopened_store.ingest(prepared.document, prepared.iter_units())
    assert duplicate.duplicate is True
    assert reopened_store.count_documents() == 1
    assert len(reopened_store.get_blocks(prepared.document.document_id)) == 3


def test_empty_text_is_stored_as_visible_no_text_status(tmp_path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text(" \r\n\t\n", encoding="utf-8", newline="")
    prepared = prepare_source(source)
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")

    result = store.ingest(prepared.document, prepared.iter_units())

    assert result.document.status == "no_text"
    assert result.empty_unit_indexes == (1,)
    assert store.get_unit(prepared.document.document_id, 1).text == ""
    assert store.get_blocks(prepared.document.document_id) == []


def test_duplicate_source_with_changed_metadata_fails_explicitly(tmp_path) -> None:
    source = tmp_path / "same.txt"
    source.write_text("Same source.", encoding="utf-8")
    first = prepare_source(source, usage_scope="personal")
    changed_metadata = prepare_source(source, usage_scope="public-license")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")

    store.ingest(first.document, first.iter_units())
    with pytest.raises(StoreError, match="different metadata"):
        store.ingest(changed_metadata.document, changed_metadata.iter_units())

    stored = store.get_document(first.document.document_id)
    assert stored is not None and stored.usage_scope == "personal"
    assert store.count_documents() == 1


def test_invalid_utf8_rolls_back_document_insert(tmp_path) -> None:
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff\xfe")
    prepared = prepare_source(source)
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")

    with pytest.raises(DocumentIngestError, match="valid UTF-8"):
        store.ingest(prepared.document, prepared.iter_units())

    assert store.count_documents() == 0


def test_store_rolls_back_all_units_when_parser_fails_mid_import(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("First paragraph.\n", encoding="utf-8")
    prepared = prepare_source(source)
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")

    def broken_units():
        yield ParsedUnit(
            unit_index=1,
            page_number=None,
            text="First paragraph.\n",
            status="ok",
        )
        raise RuntimeError("synthetic parser failure")

    with pytest.raises(RuntimeError, match="synthetic parser failure"):
        store.ingest(prepared.document, broken_units())

    assert store.count_documents() == 0


def test_pdf_ingest_tracks_each_page_and_cli_reads_page_back(tmp_path, capsys) -> None:
    source = tmp_path / "synthetic.pdf"
    _write_two_page_pdf(source)
    database = tmp_path / "documents.sqlite3"

    assert main(["ingest", str(source), "--db", str(database), "--language", "en"]) == 0
    ingest_output = capsys.readouterr().out
    ingest_report = json.loads(ingest_output)
    document_id = ingest_report["document"]["document_id"]
    store = SQLiteDocumentStore(database)
    document = store.get_document(document_id)
    units = store.get_unit_summaries(document_id)

    assert str(source) not in ingest_output
    assert document is not None
    assert document.page_count == 2
    assert document.unit_count == 2
    assert document.block_count == 1
    assert document.status == "partial"
    assert [(unit.page_number, unit.status) for unit in units] == [
        (1, "ok"),
        (2, "no_text"),
    ]

    assert main(["show", document_id, "--unit", "1", "--db", str(database)]) == 0
    assert "English PDF page one." in capsys.readouterr().out

    assert main(["show", document_id, "--unit", "2", "--db", str(database)]) == 0
    assert "no extractable text" in capsys.readouterr().err


def test_installed_cli_ingests_and_reads_text_from_outside_repository(
    tmp_path,
) -> None:
    source = tmp_path / "bilingual.md"
    source_text = "# Notes\n\nEnglish detail.\n\n中文細節。\n"
    source.write_text(source_text, encoding="utf-8", newline="")
    database = tmp_path / "local" / "documents.sqlite3"
    executable = shutil.which("doc-rag")
    assert executable is not None

    ingest = subprocess.run(
        [
            executable,
            "ingest",
            str(source),
            "--db",
            str(database),
            "--usage-scope",
            "synthetic-test",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ingest.returncode == 0, ingest.stderr
    document_id = json.loads(ingest.stdout)["document"]["document_id"]

    readback = subprocess.run(
        [
            executable,
            "show",
            document_id,
            "--unit",
            "1",
            "--db",
            str(database),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert readback.returncode == 0, readback.stderr
    assert readback.stdout == source_text


def test_corrupt_pdf_is_rejected_before_database_creation(tmp_path) -> None:
    source = tmp_path / "corrupt.pdf"
    source.write_bytes(b"not a PDF")
    database = tmp_path / "documents.sqlite3"

    with pytest.raises(DocumentIngestError, match="cannot open PDF"):
        prepared = prepare_source(source)
        SQLiteDocumentStore(database).ingest(prepared.document, prepared.iter_units())

    assert not database.exists()


def test_unknown_existing_sqlite_database_is_not_modified(tmp_path) -> None:
    database = tmp_path / "existing.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE keep_me (value TEXT NOT NULL)")
        connection.execute("INSERT INTO keep_me VALUES ('preserve')")

    with pytest.raises(StoreError, match="choose a new database file"):
        SQLiteDocumentStore(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM keep_me").fetchone() == ("preserve",)


def test_metadata_rejects_parser_mismatch(tmp_path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("text", encoding="utf-8")
    prepared = prepare_source(source)

    with pytest.raises(ValidationError, match="parser name does not match"):
        DocumentMetadata.model_validate(
            {
                **prepared.document.model_dump(),
                "parser_name": "markdown",
            }
        )
