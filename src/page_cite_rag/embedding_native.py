"""Experimental GGUF embedding adapter; an owned local server supplies the transport.

The transport is deliberately injected.  F1a callers must own and monitor the
llama.cpp process; this module neither starts a server nor downloads a model.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import numpy as np

from page_cite_rag.embedding import DEFAULT_MODEL_DIR, MODEL_ID, normalized_vectors, verify_model
from page_cite_rag.llama import LLAMA_COMMIT
from page_cite_rag.retrieval import RetrievalError

GGUF_REVISION = "370f27d7550e0def9b39c1f16d3fbaa13aa67728"
GGUF_SHA256 = "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340"


class NativeEmbeddingTransport(Protocol):
    """One already-authenticated, owned llama.cpp embedding server."""

    def token_ids(self, text: str) -> list[int]: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class NativeEmbeddingEncoder:
    """Exact native token admission with local fast-tokenizer offset suggestions.

    A window boundary from the fast tokenizer is never trusted as a budget
    check: the active native server re-tokenizes the exact emitted substring.
    """

    dimension = 1024

    def __init__(
        self,
        transport: NativeEmbeddingTransport,
        tokenizer_directory: Path = DEFAULT_MODEL_DIR,
        *,
        max_tokens: int = 512,
        overlap: int = 64,
        batch_size: int = 4,
    ) -> None:
        if (
            type(max_tokens) is not int
            or not 32 <= max_tokens <= 1024
            or type(overlap) is not int
            or not 0 <= overlap < max_tokens - 16
            or type(batch_size) is not int
            or batch_size < 1
        ):
            raise ValueError("invalid embedding token, overlap or batch budget")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RetrievalError("native embedding requires the tokenizers extra") from exc
        directory = tokenizer_directory.expanduser()
        hashes = verify_model(directory)
        import json

        prompt_info = json.loads((directory / "config_sentence_transformers.json").read_text())
        prompt = prompt_info["prompts"]["query"]
        if not prompt or prompt_info["prompts"].get("document") != "":
            raise RetrievalError("local embedding query/document prompt changed")
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._transport = transport
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._overlap = overlap
        self._batch_size = batch_size
        self.identity = {
            "model_id": MODEL_ID,
            "gguf_revision": GGUF_REVISION,
            "gguf_sha256": GGUF_SHA256,
            "backend": "llama-cpp-cuda",
            "llama_commit": LLAMA_COMMIT,
            "pooling": "last",
            "dimension": self.dimension,
            "query_prompt": prompt,
            "document_prompt": "",
            "max_tokens": max_tokens,
            "overlap": overlap,
            "batch_size": batch_size,
            "chunking": "fast-offset-suggestion/native-token-admission/v1",
            "tokenizer_sha256": hashes["tokenizer.json"],
            "tokenizers_version": version("tokenizers"),
        }

    def _token_ids(self, text: str) -> list[int]:
        tokens = self._transport.token_ids(text)
        if not isinstance(tokens, list) or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise RetrievalError("native tokenizer returned invalid token IDs")
        return tokens

    def split_document(self, text: str) -> list[tuple[int, int]]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding document must be nonempty text")
        if len(self._token_ids(text)) <= self._max_tokens:
            return [(0, len(text))]
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        if not offsets:
            raise RetrievalError("fast tokenizer could not suggest document windows")
        window = self._max_tokens - 16
        spans = []
        start = 0
        while start < len(offsets):
            end = min(start + window, len(offsets))
            char_start = 0 if start == 0 else offsets[start][0]
            if spans:
                char_start = min(char_start, spans[-1][1])
            char_end = len(text) if end == len(offsets) else offsets[end - 1][1]
            while len(self._token_ids(text[char_start:char_end])) > self._max_tokens:
                end -= 1
                if end <= start + self._overlap:
                    raise RetrievalError("cannot split passage within its native token budget")
                char_end = offsets[end - 1][1]
            if char_end <= char_start:
                raise RetrievalError("native embedding window has no source text")
            spans.append((char_start, char_end))
            if end == len(offsets):
                break
            start = end - self._overlap
        if spans[0][0] != 0 or spans[-1][1] != len(text):
            raise RetrievalError("native embedding windows do not cover source text")
        return spans

    def _encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding texts must be nonempty strings")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        inputs = [(self._prompt if query else "") + text for text in texts]
        for text in inputs:
            if len(self._token_ids(text)) > self._max_tokens:
                raise RetrievalError(
                    "embedding input exceeds the native token budget; not truncated"
                )
        batches = []
        for start in range(0, len(inputs), self._batch_size):
            group = inputs[start : start + self._batch_size]
            vectors = self._transport.embed(group)
            batches.append(normalized_vectors(vectors, len(group), self.dimension))
        return np.concatenate(batches, axis=0)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, query=False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, query=True)
