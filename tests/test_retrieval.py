from __future__ import annotations

import hashlib
import json
import sqlite3

import bm25s
import pytest

from doc_rag.cli import main
from doc_rag.ingest import prepare_source
from doc_rag.models import DocumentMetadata, ParsedUnit
from doc_rag.retrieval import BM25Retriever, RetrievalError, build_bm25_index, index_root
from doc_rag.store import SQLiteDocumentStore
from doc_rag.tokenize import LexicalTokenizer


def _ingest(tmp_path, text="Alpha marker.\n\nBeta marker.\n\nAlpha marker.", name="sample.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    prepared = prepare_source(path, usage_scope="synthetic-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    return store, prepared.document.document_id


def test_persisted_search_preserves_ranking_scope_and_source_offsets(tmp_path) -> None:
    store, document_id = _ingest(tmp_path)
    _, other_id = _ingest(tmp_path, "Alpha marker is unrelated.", name="other.txt")
    manifest = build_bm25_index(store, document_id)
    build_bm25_index(store, other_id)
    reopened = SQLiteDocumentStore(store.database_path)
    retriever = BM25Retriever(reopened, document_id)

    result = retriever.search("ALPHA", top_k=100)

    assert result.index_id == manifest.index_id
    assert [hit.block.ordinal for hit in result.hits] == [1, 3]
    assert [hit.rank for hit in result.hits] == [1, 2]
    for hit in result.hits:
        assert hit.score > 0
        assert hit.block.document_id == document_id
        unit = reopened.get_unit(document_id, hit.block.unit_index)
        assert unit.text[hit.block.char_start : hit.block.char_end] == hit.block.text


@pytest.mark.parametrize("query", ["", "!!!", "absentvocabulary", "完全不存在的詞彙"])
def test_zero_match_does_not_return_arbitrary_passages(tmp_path, query) -> None:
    store, document_id = _ingest(tmp_path)
    build_bm25_index(store, document_id)
    assert BM25Retriever(store, document_id).search(query).hits == ()


def test_tokenizer_keeps_details_and_searches_mixed_chinese_and_english(tmp_path) -> None:
    tokenizer = LexicalTokenizer()
    tokens = tokenizer.tokenize("不得增加 -3.5% ＢＭ２５ not unless 2026")
    assert {"不得", "-3.5", "bm25", "not", "unless", "2026"}.issubset(tokens)
    adjacent = tokenizer.tokenize("中文ＢＭ２５檢索 English中文 −3.5")
    assert {"bm25", "english", "中文", "-3.5"}.issubset(adjacent)
    text = "中文檢索保留原文與數字。BM25 不會翻譯。\n\nEnglish retrieval preserves numbers."
    store, document_id = _ingest(tmp_path, text)
    build_bm25_index(store, document_id)
    retriever = BM25Retriever(store, document_id)
    assert retriever.search("中文檢索 BM25").hits[0].block.ordinal == 1
    assert retriever.search("ENGLISH retrieval").hits[0].block.ordinal == 2
    assert store.get_unit(document_id, 1).text == text


def test_many_queries_reuse_loaded_index_without_building_or_loading_again(tmp_path, monkeypatch):
    store, document_id = _ingest(tmp_path)
    manifest = build_bm25_index(store, document_id)
    retriever = BM25Retriever(store, document_id)
    timestamps = {
        p.name: p.stat().st_mtime_ns for p in (index_root(store) / manifest.index_id).iterdir()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("query attempted to rebuild or reload resources")

    monkeypatch.setattr(bm25s.BM25, "index", forbidden)
    monkeypatch.setattr(bm25s.BM25, "load", forbidden)
    monkeypatch.setattr(store, "get_blocks", forbidden)
    results = list(retriever.search_many(["alpha", "beta", "missing"] * 20))

    assert len(results) == 60
    assert all(result.index_id == manifest.index_id for result in results)
    assert all(result.hits == () for result in results[2::3])
    assert timestamps == {
        p.name: p.stat().st_mtime_ns for p in (index_root(store) / manifest.index_id).iterdir()
    }


def test_missing_index_is_explicit_and_does_not_build(tmp_path) -> None:
    store, document_id = _ingest(tmp_path)
    with pytest.raises(RetrievalError, match="no BM25 index"):
        BM25Retriever(store, document_id)
    assert not index_root(store).exists()


@pytest.mark.parametrize("text", ["", "!!!\n\n---"])
def test_unsearchable_document_cannot_activate_an_index(tmp_path, text) -> None:
    store, document_id = _ingest(tmp_path, text)
    with pytest.raises(RetrievalError, match="no extractable blocks|no searchable words"):
        build_bm25_index(store, document_id)
    assert store.get_retrieval_manifest(document_id, "bm25") is None


def test_failed_rebuild_keeps_previous_active_index(tmp_path, monkeypatch) -> None:
    store, document_id = _ingest(tmp_path)
    original = build_bm25_index(store, document_id)

    def failed_save(*args, **kwargs):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(bm25s.BM25, "save", failed_save)
    with pytest.raises(OSError, match="synthetic"):
        build_bm25_index(store, document_id)
    assert BM25Retriever(store, document_id).manifest.index_id == original.index_id
    assert not list(index_root(store).glob(".build-*"))


def test_failed_activation_keeps_previous_index_queryable(tmp_path, monkeypatch) -> None:
    store, document_id = _ingest(tmp_path)
    original = build_bm25_index(store, document_id)

    def failed_activation(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic transaction failure")

    monkeypatch.setattr(store, "save_retrieval_manifest", failed_activation)
    with pytest.raises(sqlite3.OperationalError, match="synthetic"):
        build_bm25_index(store, document_id)
    retriever = BM25Retriever(store, document_id)
    assert retriever.manifest.index_id == original.index_id
    assert retriever.search("alpha").hits


@pytest.mark.parametrize("change", ["checksum", "version", "source", "scope", "manifest"])
def test_incompatible_snapshots_fail_instead_of_silently_rebuilding(tmp_path, change) -> None:
    store, document_id = _ingest(tmp_path)
    manifest = build_bm25_index(store, document_id)
    data = manifest.model_dump(mode="json")
    if change == "checksum":
        (index_root(store) / manifest.index_id / "data.csc.index.npy").write_bytes(b"damaged")
    elif change == "version":
        data["tokenizer_version"] = "old-tokenizer"
    elif change == "source":
        data["source_fingerprint"] = "0" * 64
    elif change == "scope":
        data["document_id"] = "0" * 64
    else:
        data["index_id"] = "../../outside"
    store.save_retrieval_manifest(document_id, "bm25", json.dumps(data))
    with pytest.raises(RetrievalError, match="rebuild"):
        BM25Retriever(store, document_id)
    assert len(list(index_root(store).glob("[0-9a-f]*"))) == 1


def test_array_shape_validation_even_when_checksum_matches(tmp_path) -> None:
    import numpy as np

    store, document_id = _ingest(tmp_path)
    manifest = build_bm25_index(store, document_id)
    path = index_root(store) / manifest.index_id / "indices.csc.index.npy"
    np.save(path, np.array([999], dtype=np.int32), allow_pickle=False)
    data = manifest.model_dump(mode="json")
    data["file_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    store.save_retrieval_manifest(document_id, "bm25", json.dumps(data))
    with pytest.raises(RetrievalError, match="incompatible"):
        BM25Retriever(store, document_id)


def test_pickle_backed_array_is_rejected(tmp_path) -> None:
    import numpy as np

    store, document_id = _ingest(tmp_path)
    manifest = build_bm25_index(store, document_id)
    path = index_root(store) / manifest.index_id / "data.csc.index.npy"
    np.save(path, np.array(["unsupported-object"], dtype=object), allow_pickle=True)
    data = manifest.model_dump(mode="json")
    data["file_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    store.save_retrieval_manifest(document_id, "bm25", json.dumps(data))
    with pytest.raises(RetrievalError, match="unreadable"):
        BM25Retriever(store, document_id)


def test_search_retains_pdf_page_and_line_provenance(tmp_path) -> None:
    document = DocumentMetadata(
        document_id="1" * 64,
        source_sha256="2" * 64,
        source_name="synthetic.pdf",
        media_type="application/pdf",
        parser_name="pdfplumber",
        parser_version="synthetic-store-fixture",
        page_count=2,
    )
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(
        document,
        [
            ParsedUnit(unit_index=1, page_number=1, text="Overview.", status="ok"),
            ParsedUnit(
                unit_index=2, page_number=2, text="Result.\nNot fewer than 12 samples.", status="ok"
            ),
        ],
    )
    build_bm25_index(store, document.document_id)
    block = BM25Retriever(store, document.document_id).search("samples").hits[0].block
    assert block.page_number == block.unit_index == 2
    assert (block.line_start, block.line_end) == (1, 2)
    assert block.text == "Result.\nNot fewer than 12 samples."


def test_v1_database_migration_preserves_source_and_enables_indexing(tmp_path) -> None:
    store, document_id = _ingest(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TABLE retrieval_indexes")
        connection.execute("PRAGMA user_version = 1")
    migrated = SQLiteDocumentStore(store.database_path)
    assert migrated.get_unit(document_id, 1).text.startswith("Alpha")
    build_bm25_index(migrated, document_id)
    assert BM25Retriever(migrated, document_id).search("alpha").hits
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_cli_index_search_and_missing_database_are_explicit(tmp_path, capsys) -> None:
    store, document_id = _ingest(tmp_path)
    args = [document_id, "--db", str(store.database_path)]
    assert main(["index", *args]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert main(["search", *args, "alpha", "--top-k", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["index_id"] == manifest["index_id"]
    assert len(result["hits"]) == 1
    assert result["hits"][0]["block"]["text"] == store.get_blocks(document_id)[0].text
    with pytest.raises(SystemExit) as exc:
        main(["search", *args, "alpha", "--top-k", "0"])
    assert exc.value.code == 2
    capsys.readouterr()
    missing = tmp_path / "not-created" / "missing.sqlite3"
    assert main(["search", document_id, "alpha", "--db", str(missing)]) == 1
    assert "database was not found" in capsys.readouterr().err
    assert not missing.parent.exists()


def test_top_k_contract_and_empty_block_mapping(tmp_path) -> None:
    store, document_id = _ingest(tmp_path, "!!!\n\nAlpha marker.")
    build_bm25_index(store, document_id)
    retriever = BM25Retriever(store, document_id)
    assert retriever.search("alpha").hits[0].block.ordinal == 2
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="top_k"):
            retriever.search("alpha", top_k=invalid)
