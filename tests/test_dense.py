from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from doc_rag import dense, embedding
from doc_rag.dense import DenseRetriever, build_dense_index
from doc_rag.embedding import normalized_vectors
from doc_rag.ingest import prepare_source
from doc_rag.retrieval import RetrievalError, index_root
from doc_rag.store import SQLiteDocumentStore


class FakeEncoder:
    dimension = 2

    def __init__(self):
        self.identity = {"model": "synthetic", "revision": "v1", "dimension": 2}
        self.document_calls = self.query_calls = 0

    def split_document(self, text):
        return [(0, len(text))]

    def encode_documents(self, texts):
        self.document_calls += 1
        return np.array([[1, 0] if "Alpha" in text else [0, 1] for text in texts])

    def encode_queries(self, texts):
        self.query_calls += 1
        return np.array([[1, 0] if text != "negative" else [-1, 0] for text in texts])


def ingest(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Alpha 12 samples.\n\nBeta not unless 3.5%.\n\nAlpha controls.")
    prepared = prepare_source(path, usage_scope="synthetic")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    return store, prepared.document.document_id


def test_dense_round_trip_ranking_source_and_resource_reuse(tmp_path, monkeypatch):
    store, document_id = ingest(tmp_path)
    encoder = FakeEncoder()
    manifest = build_dense_index(store, document_id, encoder)
    retriever = DenseRetriever(store, document_id, encoder)
    assert isinstance(retriever._vectors, np.memmap)
    monkeypatch.setattr(store, "get_blocks", lambda *args: pytest.fail("query reloaded blocks"))
    monkeypatch.setattr(dense.np, "load", lambda *args, **kw: pytest.fail("query reloaded index"))
    for query in ["中文問題", "question", "negative"]:
        result = retriever.search(query, top_k=100)
        assert result.index_id == manifest.index_id
        assert result.method == "dense"
        assert len(result.hits) == 3
        assert all(hit.block.document_id == document_id for hit in result.hits)
        for hit in result.hits:
            unit = store.get_unit(document_id, hit.block.unit_index)
            assert unit.text[hit.block.char_start : hit.block.char_end] == hit.block.text
    assert [hit.block.ordinal for hit in retriever.search("query").hits] == [1, 3, 2]
    assert [hit.score for hit in retriever.search("negative").hits] == [0, -1, -1]
    assert retriever.search("   ").hits == ()
    assert encoder.document_calls == 1
    assert encoder.query_calls == 5


def test_dense_subchunks_aggregate_to_unique_original_blocks(tmp_path):
    store, document_id = ingest(tmp_path)
    encoder = FakeEncoder()
    encoder.split_document = lambda text: [(0, len(text) // 2), (len(text) // 2, len(text))]
    encoder.encode_documents = lambda texts: np.array([[1, i % 2] for i in range(len(texts))])
    manifest = build_dense_index(store, document_id, encoder)
    assert len(manifest.segments) == 6
    result = DenseRetriever(store, document_id, encoder).search("q")
    assert len(result.hits) == 3
    assert len({h.block.block_id for h in result.hits}) == 3
    assert all(h.score == 1 for h in result.hits)


@pytest.mark.parametrize(
    "damage",
    ["checksum", "shape", "zero", "nan", "pickle", "source", "encoder", "scope", "segment", "path"],
)
def test_dense_invalid_snapshots_fail_without_rebuild(tmp_path, damage):
    store, document_id = ingest(tmp_path)
    encoder = FakeEncoder()
    manifest = build_dense_index(store, document_id, encoder)
    raw = manifest.model_dump(mode="json")
    path = index_root(store) / manifest.index_id / "vectors.npy"
    if damage == "checksum":
        path.write_bytes(b"damaged")
    elif damage in ("shape", "zero", "nan", "pickle"):
        value = {
            "shape": np.ones((1, 2)),
            "zero": np.zeros((3, 2), dtype=np.float32),
            "nan": np.full((3, 2), np.nan, dtype=np.float32),
            "pickle": np.array(["bad"], dtype=object),
        }[damage]
        np.save(path, value, allow_pickle=damage == "pickle")
        raw["vectors_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif damage == "source":
        raw["source_fingerprint"] = "0" * 64
    elif damage == "encoder":
        encoder.identity["revision"] = "v2"
    elif damage == "scope":
        raw["document_id"] = "0" * 64
    elif damage == "segment":
        raw["segments"][0]["char_start"] = 1
    else:
        raw["index_id"] = "../../outside"
    store.save_retrieval_manifest(document_id, "dense", json.dumps(raw))
    with pytest.raises(RetrievalError):
        DenseRetriever(store, document_id, encoder)


def test_failed_dense_build_keeps_active_snapshot(tmp_path, monkeypatch):
    store, document_id = ingest(tmp_path)
    encoder = FakeEncoder()
    original = build_dense_index(store, document_id, encoder)
    encoder.encode_documents = lambda texts: np.zeros((len(texts), 2))
    with pytest.raises(RetrievalError, match="zero"):
        build_dense_index(store, document_id, encoder)
    assert DenseRetriever(store, document_id, encoder).manifest.index_id == original.index_id
    encoder.encode_documents = lambda texts: np.ones((len(texts), 2))
    monkeypatch.setattr(
        store, "save_retrieval_manifest", lambda *args: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(OSError):
        build_dense_index(store, document_id, encoder)
    assert DenseRetriever(store, document_id, encoder).manifest.index_id == original.index_id


def test_invalid_encoder_segments_and_outputs_are_not_activated(tmp_path):
    store, document_id = ingest(tmp_path)
    encoder = FakeEncoder()
    encoder.split_document = lambda text: [(1, len(text))]
    with pytest.raises(RetrievalError, match="mapping"):
        build_dense_index(store, document_id, encoder)
    assert store.get_retrieval_manifest(document_id, "dense") is None
    with pytest.raises(RetrievalError):
        normalized_vectors(np.ones((3, 5)), 3, 2)


def test_pinned_model_preparation_checks_files_and_does_not_redownload(tmp_path, monkeypatch):
    import io

    content = b"synthetic model (not real weights)"
    monkeypatch.setattr(
        embedding,
        "MODEL_FILES",
        {"model.safetensors": (len(content), hashlib.sha256(content).hexdigest())},
    )
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(embedding.urllib.request, "urlopen", download)
    directory = tmp_path / "model"
    report = embedding.prepare_embedding_model(directory)
    assert len(calls) == 1
    assert embedding.prepare_embedding_model(directory) == report
    assert len(calls) == 1
    (directory / "unexpected.py").write_text("raise AssertionError")
    with pytest.raises(RetrievalError, match="file list"):
        embedding.verify_model(directory)


def test_bad_model_download_never_activates_directory(tmp_path, monkeypatch):
    import io

    monkeypatch.setattr(embedding, "MODEL_FILES", {"config.json": (2, "0" * 64)})
    monkeypatch.setattr(embedding.urllib.request, "urlopen", lambda *args, **kw: io.BytesIO(b"bad"))
    with pytest.raises(RetrievalError, match="size"):
        embedding.prepare_embedding_model(tmp_path / "model")
    assert not (tmp_path / "model").exists()
