from __future__ import annotations

import hashlib
import shutil
import sqlite3

import pytest

from page_cite_rag.ingest import prepare_source
from page_cite_rag.store import SQLiteDocumentStore, StoreError


def _store(tmp_path):
    path = tmp_path / "source.md"
    path.write_text("First detail.\n\nSecond detail.\n\nThird detail.", encoding="utf-8")
    source = prepare_source(path, usage_scope="self-authored-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(source.document, source.iter_units())
    return store, source


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_only_store_reads_scoped_source_without_changing_database(tmp_path):
    writable, source = _store(tmp_path)
    before = _sha(writable.database_path)
    stamp = writable.database_path.stat().st_mtime_ns
    store = SQLiteDocumentStore(writable.database_path, read_only=True)
    blocks = store.get_blocks(source.document.document_id)
    assert len(blocks) == 3
    assert store.get_document(source.document.document_id) is not None
    assert store.get_unit(source.document.document_id, 1).text.startswith("First detail.")
    assert store.get_block(source.document.document_id, blocks[1].block_id) == blocks[1]
    assert store.get_block("0" * 64, blocks[1].block_id) is None
    assert store.get_neighbor_blocks(source.document.document_id, 2) == blocks
    assert store.get_neighbor_blocks(source.document.document_id, 2, radius=0) == blocks[1:2]
    assert store.get_neighbor_blocks("0" * 64, 2, radius=2) == []
    assert _sha(writable.database_path) == before
    assert writable.database_path.stat().st_mtime_ns == stamp


def test_read_only_connection_and_public_mutators_reject_writes(tmp_path):
    writable, source = _store(tmp_path)
    before = _sha(writable.database_path)
    store = SQLiteDocumentStore(writable.database_path, read_only=True)
    with pytest.raises(StoreError, match="read-only"):
        store.ingest(source.document, source.iter_units())
    with pytest.raises(StoreError, match="read-only"):
        store.save_retrieval_manifest(source.document.document_id, "bm25", "{}")
    connection = store._connect()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM blocks")
    finally:
        connection.close()
    assert _sha(writable.database_path) == before


def test_read_only_open_does_not_create_missing_file_or_parent(tmp_path):
    path = tmp_path / "absent" / "missing.sqlite3"
    with pytest.raises(StoreError, match="not found"):
        SQLiteDocumentStore(path, read_only=True)
    assert not path.exists()
    assert not path.parent.exists()


def test_read_only_open_does_not_initialize_an_empty_existing_file(tmp_path):
    path = tmp_path / "empty.sqlite3"
    path.touch()
    with pytest.raises(StoreError, match="schema v2"):
        SQLiteDocumentStore(path, read_only=True)
    assert path.read_bytes() == b""


def test_read_only_open_rejects_v1_without_migration(tmp_path):
    writable, _ = _store(tmp_path)
    path = tmp_path / "old.sqlite3"
    shutil.copyfile(writable.database_path, path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE retrieval_indexes")
        connection.execute("PRAGMA user_version = 1")
    before = _sha(path)
    with pytest.raises(StoreError, match="schema v2"):
        SQLiteDocumentStore(path, read_only=True)
    assert _sha(path) == before
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)


def test_read_only_open_handles_uri_sensitive_path_characters(tmp_path):
    writable, source = _store(tmp_path)
    path = tmp_path / "data #?% 中文.sqlite3"
    shutil.copyfile(writable.database_path, path)
    before = _sha(path)
    store = SQLiteDocumentStore(path, read_only=True)
    assert store.get_document(source.document.document_id) is not None
    assert _sha(path) == before


@pytest.mark.parametrize("radius", [True, -1, 3, 1.5, "1", None])
def test_neighbor_radius_is_strictly_bounded(tmp_path, radius):
    store, source = _store(tmp_path)
    with pytest.raises(ValueError, match="radius"):
        store.get_neighbor_blocks(source.document.document_id, 1, radius=radius)


@pytest.mark.parametrize("ordinal", [True, 0, -1, 1.5, "1", None])
def test_neighbor_ordinal_is_a_positive_integer(tmp_path, ordinal):
    store, source = _store(tmp_path)
    with pytest.raises(ValueError, match="ordinal"):
        store.get_neighbor_blocks(source.document.document_id, ordinal)


@pytest.mark.parametrize("read_only", [1, "true", None])
def test_read_only_option_is_strict(tmp_path, read_only):
    with pytest.raises(ValueError, match="boolean"):
        SQLiteDocumentStore(tmp_path / "missing.sqlite3", read_only=read_only)
