"""Native adapter contracts without a model, network or GPU."""

from __future__ import annotations

import json

import numpy as np
import pytest

from page_cite_rag import embedding_native
from page_cite_rag.embedding_native import NativeEmbeddingEncoder
from page_cite_rag.retrieval import RetrievalError

tokenizers = pytest.importorskip("tokenizers")


class FakeTransport:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.embedded = []

    def token_ids(self, text):
        return self.tokenizer.encode(text).ids

    def embed(self, texts):
        self.embedded.extend(texts)
        return np.ones((len(texts), 1024), dtype=np.float32)


@pytest.fixture
def encoder_parts(tmp_path, monkeypatch):
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": "Instruct: search\nQuery:", "document": ""}})
    )
    monkeypatch.setattr(
        embedding_native, "verify_model", lambda directory: {"tokenizer.json": "a" * 64}
    )
    return tmp_path, FakeTransport(tokenizer)


def test_native_adapter_windows_cover_source_and_reject_overlong_query(encoder_parts):
    directory, transport = encoder_parts
    encoder = NativeEmbeddingEncoder(transport, directory, max_tokens=32, overlap=4, batch_size=2)
    long_text = "word " * 100
    windows = encoder.split_document(long_text)
    assert len(windows) > 1
    assert windows[0][0] == 0 and windows[-1][1] == len(long_text)
    assert all(a[1] >= b[0] for a, b in zip(windows, windows[1:]))
    assert all(len(transport.token_ids(long_text[a:b])) <= 32 for a, b in windows)
    with pytest.raises(RetrievalError, match="token budget"):
        encoder.encode_queries([long_text])
    assert not transport.embedded


def test_native_adapter_uses_only_query_prefix_and_validates_vectors(encoder_parts):
    directory, transport = encoder_parts
    encoder = NativeEmbeddingEncoder(transport, directory, max_tokens=32, overlap=4, batch_size=2)
    assert encoder.encode_documents(["word", "word", "word"]).shape == (3, 1024)
    assert transport.embedded == ["word", "word", "word"]
    assert encoder.encode_queries(["word"]).shape == (1, 1024)
    assert transport.embedded[-1] == "Instruct: search\nQuery:word"
    assert encoder.encode_documents([]).shape == (0, 1024)
    assert encoder.identity["gguf_sha256"] == embedding_native.GGUF_SHA256

    transport.embed = lambda texts: np.zeros((len(texts), 1024), dtype=np.float32)
    with pytest.raises(RetrievalError, match="zero"):
        encoder.encode_documents(["word"])
    transport.token_ids = lambda text: [True]
    with pytest.raises(RetrievalError, match="invalid token IDs"):
        encoder.encode_documents(["word"])


def test_query_context_can_exceed_document_chunk_limit_without_truncation(encoder_parts):
    directory, transport = encoder_parts
    (directory / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    transport._context_size = lambda: 1024
    encoder = NativeEmbeddingEncoder(
        transport,
        directory,
        max_tokens=32,
        query_max_tokens=1024,
        overlap=4,
        batch_size=1,
    )

    query = "word " * 40
    assert encoder.identity["max_tokens"] == 32
    assert encoder.identity["query_max_tokens"] == 1024
    assert encoder.encode_queries([query]).shape == (1, 1024)
    assert transport.embedded[-1] == "Instruct: search\nQuery:" + query
    with pytest.raises(RetrievalError, match="token budget"):
        encoder.encode_documents([query])

    transport._context_size = lambda: 512
    with pytest.raises(RetrievalError, match="server context"):
        NativeEmbeddingEncoder(
            transport,
            directory,
            max_tokens=32,
            query_max_tokens=1024,
            overlap=4,
        )


def test_explicit_large_context_checks_model_and_server_capacity(encoder_parts):
    directory, transport = encoder_parts
    (directory / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    transport._context_size = lambda: 8192
    encoder = NativeEmbeddingEncoder(transport, directory, max_tokens=8192, batch_size=1)
    assert encoder.identity["max_tokens"] == 8192
    assert encoder.encode_documents(["word " * 1100]).shape == (1, 1024)
    transport._context_size = lambda: 1024
    with pytest.raises(RetrievalError, match="server context"):
        NativeEmbeddingEncoder(transport, directory, max_tokens=8192)
    (directory / "config.json").write_text(json.dumps({"max_position_embeddings": 2048}))
    with pytest.raises(RetrievalError, match="model capacity"):
        NativeEmbeddingEncoder(transport, directory, max_tokens=8192)


def test_embedding_server_large_context_is_explicit_and_bounded(tmp_path, monkeypatch):
    from page_cite_rag.embedding_server import NativeEmbeddingServer

    monkeypatch.setattr(NativeEmbeddingServer, "_start", lambda self: None)
    with NativeEmbeddingServer(tmp_path / "fake.gguf", "fake", context_tokens=8192) as server:
        assert server._context_size() == 8192
        assert server._request_timeout() == 120
    with pytest.raises(ValueError, match="embedding context"):
        NativeEmbeddingServer(tmp_path / "fake.gguf", "fake", context_tokens=8193)


def test_embedding_server_timeout_is_explicitly_bounded(tmp_path, monkeypatch):
    from page_cite_rag.embedding_server import NativeEmbeddingServer

    monkeypatch.setattr(NativeEmbeddingServer, "_start", lambda self: None)
    with NativeEmbeddingServer(
        tmp_path / "fake.gguf", "fake", request_timeout_seconds=60
    ) as server:
        assert server._request_timeout() == 60
    with pytest.raises(ValueError, match="request timeout"):
        NativeEmbeddingServer(tmp_path / "fake.gguf", "fake", request_timeout_seconds=601)


@pytest.mark.parametrize(
    ("path", "expected_waits"),
    [
        ("/v1/embeddings", 1),
        ("/v1/rerank", 1),
        ("/tokenize", 0),
    ],
)
def test_gpu_server_gates_compute_requests_but_not_tokenization(path, expected_waits):
    from page_cite_rag.embedding_server import NativeEmbeddingServer

    class Response:
        status_code = 200
        content = b"{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class Client:
        def post(self, *_args, **_kwargs):
            return Response()

    server = object.__new__(NativeEmbeddingServer)
    server._check = lambda: None
    server._http = Client()
    waits = []
    server.wait_for_gpu_headroom = lambda: waits.append(path) or 3.0

    assert server._post(path, {}) == {"ok": True}
    assert len(waits) == expected_waits
