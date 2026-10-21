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
