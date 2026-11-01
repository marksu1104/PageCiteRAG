"""Owned local Qwen3 reranking through the pinned llama.cpp server."""

from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path

from page_cite_rag.embedding_server import NativeEmbeddingServer
from page_cite_rag.retrieval import RetrievalError

MODEL_REVISION = "ff8a9aaee9ddbcdf483ec1b17bc62e864bcb8b04"
MODEL_FILENAME = "qwen3-reranker-0.6b-q8_0.gguf"
MODEL_BYTES = 639_153_184
MODEL_SHA256 = "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48"
DEFAULT_MODEL_PATH = Path("models/qwen3-reranker-0.6b-gguf") / MODEL_REVISION / MODEL_FILENAME
MAX_PAIR_TOKENS = 4096
MAX_RERANK_DOCUMENTS_PER_REQUEST = 8
_RERANK_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
    "<Query>: "
)
_RERANK_DOCUMENT = "\n<Document>: "
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class NativeRerankerServer(NativeEmbeddingServer):
    """Use the existing owned-process, localhost and GPU guards for reranking."""

    def __init__(self, *args, **kwargs):
        self.last_pair_tokens: tuple[int, ...] = ()
        self.last_tokenization_ms: float | None = None
        self.last_forward_ms: float | None = None
        self.last_total_ms: float | None = None
        self.last_batch_sizes: tuple[int, ...] = ()
        self.last_attempted_batch_count = 0
        self.last_completed_batch_count = 0
        super().__init__(*args, **kwargs)

    def _verify_model(self) -> None:
        path = self.model_path
        if not path.is_file() or path.is_symlink() or path.stat().st_size != MODEL_BYTES:
            raise RetrievalError("prepared native reranker GGUF is missing or has wrong size")
        with path.open("rb") as model_file:
            if hashlib.file_digest(model_file, "sha256").hexdigest() != MODEL_SHA256:
                raise RetrievalError("prepared native reranker GGUF checksum changed")

    def _context_size(self) -> int:
        return 4096

    def _batch_size(self) -> int:
        # A reranker pair is one prompt; its token length, not document count,
        # determines the minimum physical batch size accepted by llama.cpp.
        return MAX_PAIR_TOKENS

    def _mode_options(self) -> list[str]:
        return ["--embedding", "--pooling", "rank", "--reranking"]

    def _request_timeout(self) -> int:
        return 60

    def _verify_offload(self, log: str) -> None:
        matches = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
        if not matches or any(int(loaded) != int(total) for loaded, total in matches):
            raise RetrievalError("native reranker full GPU offload was not confirmed")

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not isinstance(query, str) or not query.strip() or not documents:
            raise RetrievalError("reranking needs a nonempty query and candidate documents")
        if any(not isinstance(document, str) or not document.strip() for document in documents):
            raise RetrievalError("reranking candidates must be nonempty strings")
        started = time.perf_counter()
        self.last_pair_tokens = ()
        self.last_tokenization_ms = None
        self.last_forward_ms = None
        self.last_total_ms = None
        self.last_batch_sizes = tuple(
            min(MAX_RERANK_DOCUMENTS_PER_REQUEST, len(documents) - start)
            for start in range(0, len(documents), MAX_RERANK_DOCUMENTS_PER_REQUEST)
        )
        self.last_attempted_batch_count = 0
        self.last_completed_batch_count = 0
        tokenization_started = time.perf_counter()
        token_counts = []
        for document in documents:
            tokenized = self._post(
                "/tokenize",
                {
                    "content": f"{_RERANK_PREFIX}{query}{_RERANK_DOCUMENT}{document}"
                    f"{_RERANK_SUFFIX}",
                    "add_special": False,
                    "parse_special": True,
                },
            )
            tokens = tokenized.get("tokens")
            if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                raise RetrievalError("native reranker tokenizer returned invalid token IDs")
            token_counts.append(len(tokens))
            self.last_pair_tokens = tuple(token_counts)
            self.last_tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
        if any(count > MAX_PAIR_TOKENS for count in token_counts):
            raise RetrievalError("a reranker query-document pair exceeds 4096 tokens")
        forward_started = time.perf_counter()
        ranked = []
        for offset in range(0, len(documents), MAX_RERANK_DOCUMENTS_PER_REQUEST):
            batch = documents[offset : offset + MAX_RERANK_DOCUMENTS_PER_REQUEST]
            self.last_attempted_batch_count += 1
            response = self._post(
                "/v1/rerank",
                {"model": "qwen3-reranker-0.6b", "query": query, "documents": batch},
            )
            results = response.get("results")
            if not isinstance(results, list) or len(results) != len(batch):
                raise RetrievalError("native reranker returned an invalid result count")
            seen = set()
            for item in results:
                if not isinstance(item, dict):
                    raise RetrievalError("native reranker returned a malformed result")
                index, score = item.get("index"), item.get("relevance_score")
                if (
                    type(index) is not int
                    or not 0 <= index < len(batch)
                    or index in seen
                    or type(score) not in (int, float)
                    or not math.isfinite(score)
                ):
                    raise RetrievalError("native reranker returned invalid indexes or scores")
                seen.add(index)
                ranked.append((offset + index, float(score)))
            if len(seen) != len(batch):
                raise RetrievalError("native reranker omitted a candidate")
            self.last_completed_batch_count += 1
        ordered = sorted(ranked, key=lambda item: (-item[1], item[0]))
        self.last_forward_ms = (time.perf_counter() - forward_started) * 1000
        self.last_total_ms = (time.perf_counter() - started) * 1000
        return ordered

    def diagnostics(self) -> dict:
        return {
            **super().diagnostics(),
            "pair_tokens": list(self.last_pair_tokens),
            "tokenization_ms": self.last_tokenization_ms,
            "forward_ms": self.last_forward_ms,
            "total_ms": self.last_total_ms,
            "batch_sizes": list(self.last_batch_sizes),
            "attempted_batch_count": self.last_attempted_batch_count,
            "completed_batch_count": self.last_completed_batch_count,
        }
