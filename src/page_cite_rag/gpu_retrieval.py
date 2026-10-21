"""Explicit sequential GPU query embedding over an existing versioned index."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path

import numpy as np

from page_cite_rag.application import RetrievalMethod, document_retriever
from page_cite_rag.embedding import DEFAULT_MODEL_DIR
from page_cite_rag.embedding_native import NativeEmbeddingEncoder
from page_cite_rag.embedding_server import NativeEmbeddingServer, verify_native_model
from page_cite_rag.retrieval import SearchResult
from page_cite_rag.store import SQLiteDocumentStore


class _TransportSlot:
    def __init__(self) -> None:
        self.active: NativeEmbeddingServer | None = None

    def token_ids(self, text: str) -> list[int]:
        if self.active is None:
            raise RuntimeError("GPU embedding session is not active")
        return self.active.token_ids(text)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.active is None:
            raise RuntimeError("GPU embedding session is not active")
        return self.active.embed(texts)


class StagedGpuSearch:
    """Keep the index loaded; start only the encoder server for a bounded query."""

    def __init__(
        self,
        store: SQLiteDocumentStore,
        document_id: str,
        method: RetrievalMethod,
        model_path: Path,
        gpu_uuid: str,
        *,
        tokenizer_directory: Path = DEFAULT_MODEL_DIR,
        gpu_temperature_limit_c: int = 80,
    ) -> None:
        if method not in ("dense", "hybrid"):
            raise ValueError("GPU staged search requires dense or hybrid retrieval")
        verify_native_model(model_path.expanduser())
        self._slot = _TransportSlot()
        self.encoder = NativeEmbeddingEncoder(self._slot, tokenizer_directory)
        self.retriever = document_retriever(
            store, document_id, method, encoder=self.encoder, versioned_dense=True
        )
        self.document_id = document_id
        self.method = method
        self.model_path = model_path
        self.gpu_uuid = gpu_uuid
        self.gpu_temperature_limit_c = gpu_temperature_limit_c
        self.last_trace: dict | None = None
        self._cancel_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._current_cancel: threading.Event | None = None
        self._session_context: ContextVar[object | None] = ContextVar(
            f"page_cite_rag_gpu_search_session_{id(self)}", default=None
        )
        self._session_token: object | None = None
        self._session_started = 0.0
        self._session_query_count = 0
        self._session_query_ms = 0.0
        self._session_start_attempted = False
        self._server_context = None

    def cancel(self) -> None:
        """Signal only this search's owned embedding process to stop."""
        with self._cancel_lock:
            if self._current_cancel is not None:
                self._current_cancel.set()

    def __call__(self, question: str, top_k: int) -> SearchResult:
        return self._query(question, top_k)

    def search_scoped(
        self,
        query: str,
        *,
        allowed_block_ids: set[str],
        candidate_k: int,
        top_k: int = 5,
    ) -> SearchResult:
        """Search within an authorized source range using the bound retrieval backend."""
        if (
            not allowed_block_ids
            or any(not isinstance(block_id, str) for block_id in allowed_block_ids)
            or type(candidate_k) is not int
            or candidate_k < 1
            or type(top_k) is not int
            or top_k < 1
        ):
            raise ValueError("invalid scoped GPU retrieval request")
        return self._query(
            query,
            top_k,
            allowed_block_ids=allowed_block_ids,
            candidate_k=candidate_k,
        )

    def _query(
        self,
        query: str,
        top_k: int,
        *,
        allowed_block_ids: set[str] | None = None,
        candidate_k: int | None = None,
    ) -> SearchResult:
        context_token = self._session_context.get()
        with self._cancel_lock:
            owner_token = self._session_token
            if owner_token is not None and context_token is not owner_token:
                raise RuntimeError("GPU embedding session belongs to another request")
            if owner_token is None and self._slot.active is not None:
                raise RuntimeError("GPU embedding server is active outside its request owner")
        if owner_token is not None:
            with self._query_lock:
                self._start_session_server()
                return self._search_active(
                    query,
                    top_k,
                    allowed_block_ids=allowed_block_ids,
                    candidate_k=candidate_k,
                )
        with self.session():
            return self._query(
                query,
                top_k,
                allowed_block_ids=allowed_block_ids,
                candidate_k=candidate_k,
            )

    def _search_active(
        self,
        query: str,
        top_k: int,
        *,
        allowed_block_ids: set[str] | None = None,
        candidate_k: int | None = None,
    ) -> SearchResult:
        server = self._slot.active
        if server is None:
            raise RuntimeError("GPU embedding session is not active")
        started = time.perf_counter()
        try:
            if allowed_block_ids is None:
                result = self.retriever.search(query, top_k=top_k)
            else:
                if candidate_k is None:
                    raise RuntimeError("scoped GPU retrieval is missing its candidate budget")
                result = self._search_scoped(
                    query,
                    allowed_block_ids=allowed_block_ids,
                    candidate_k=candidate_k,
                    top_k=top_k,
                )
            return result
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._session_query_count += 1
            self._session_query_ms += elapsed_ms
            self.last_trace = {
                **server.diagnostics(),
                "server_started": True,
                "server_start_attempted": self._session_start_attempted,
                "startup_ms": server.startup_ms,
                "query_ms": elapsed_ms,
                "query_count": self._session_query_count,
                "cumulative_query_ms": self._session_query_ms,
                "session_elapsed_ms": (time.perf_counter() - self._session_started) * 1000,
            }

    def _search_scoped(
        self,
        query: str,
        *,
        allowed_block_ids: set[str],
        candidate_k: int,
        top_k: int,
    ) -> SearchResult:
        backend = self.retriever
        native_store = getattr(backend, "store", None)
        chunk_block_ids = getattr(native_store, "chunk_block_ids", None)
        if chunk_block_ids is not None:
            allowed_chunks = tuple(
                index
                for index, block_ids in enumerate(chunk_block_ids)
                if block_ids and all(block_id in allowed_block_ids for block_id in block_ids)
            )
            if not allowed_chunks:
                return SearchResult(
                    document_id=self.document_id,
                    index_id=getattr(backend, "index_id", "empty-scoped-gpu-search"),
                    method="hybrid",
                    query=query,
                    hits=(),
                    elapsed_ms=0.0,
                )
            return backend.search(
                query,
                top_k=top_k,
                allowed_chunk_indices=allowed_chunks,
            )

        scoped_search = getattr(backend, "search_scoped", None)
        if callable(scoped_search):
            return scoped_search(
                query,
                allowed_block_ids=allowed_block_ids,
                candidate_k=candidate_k,
                top_k=top_k,
            )

        if self.method != "dense":
            raise ValueError("bound hybrid retriever does not implement scoped search")
        raw = backend.search(query, top_k=candidate_k)
        hits = tuple(hit for hit in raw.hits if hit.block.block_id in allowed_block_ids)[:top_k]
        return raw.model_copy(
            update={
                "query": query,
                "hits": tuple(
                    hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(hits, 1)
                ),
                "branches": (),
                "context_groups": (),
            }
        )

    @contextmanager
    def session(self) -> Iterator[StagedGpuSearch]:
        """Lazily reuse one owned embedding process for searches in one request."""
        with self._cancel_lock:
            if (
                self._current_cancel is not None
                or self._session_token is not None
                or self._slot.active is not None
            ):
                raise RuntimeError("concurrent GPU embedding sessions are forbidden")
            cancelled = threading.Event()
            self._current_cancel = cancelled
            session_token = object()
            self._session_token = session_token
        context_token: Token[object | None] = self._session_context.set(session_token)
        self._session_started = time.perf_counter()
        self._session_query_count = 0
        self._session_query_ms = 0.0
        self._server_context = None
        self.last_trace = None
        error_info = (None, None, None)
        try:
            self._session_owner = threading.get_ident()
            yield self
        except BaseException:
            error_info = sys.exc_info()
            raise
        finally:
            try:
                with self._query_lock:
                    server = self._slot.active
                    self._slot.active = None
                    cleanup_started = time.perf_counter()
                    if self._server_context is not None:
                        self._server_context.__exit__(*error_info)
                    cleanup_ms = (time.perf_counter() - cleanup_started) * 1000
                    if server is not None:
                        self.last_trace = {
                            **server.diagnostics(),
                            "server_started": True,
                            "server_start_attempted": self._session_start_attempted,
                            "startup_ms": server.startup_ms,
                            "query_count": self._session_query_count,
                            "cumulative_query_ms": self._session_query_ms,
                            "session_elapsed_ms": (time.perf_counter() - self._session_started)
                            * 1000,
                            "cleanup_ms": cleanup_ms,
                        }
                    elif self.last_trace is None:
                        self.last_trace = {
                            "server_started": False,
                            "server_start_attempted": self._session_start_attempted,
                            "query_count": self._session_query_count,
                            "session_elapsed_ms": (time.perf_counter() - self._session_started)
                            * 1000,
                            "cleanup_ms": cleanup_ms,
                        }
                    else:
                        self.last_trace.update(
                            {
                                "server_started": False,
                                "server_start_attempted": self._session_start_attempted,
                                "query_count": self._session_query_count,
                                "session_elapsed_ms": (time.perf_counter() - self._session_started)
                                * 1000,
                                "cleanup_ms": cleanup_ms,
                            }
                        )
                    self._server_context = None
            finally:
                self._session_context.reset(context_token)
                with self._cancel_lock:
                    self._session_token = None
                    self._current_cancel = None

    def _start_session_server(self) -> NativeEmbeddingServer:
        if self._session_token is None or self._session_context.get() is not self._session_token:
            raise RuntimeError("GPU embedding session belongs to another request")
        if self._slot.active is not None:
            return self._slot.active
        self._session_start_attempted = True
        try:
            context = NativeEmbeddingServer(
                self.model_path,
                self.gpu_uuid,
                cancel_event=self._current_cancel,
                max_gpu_temperature_c=self.gpu_temperature_limit_c,
            )
            server = context.__enter__()
        except Exception as exc:
            self.last_trace = {
                "server_started": False,
                "server_start_attempted": True,
                "guard_error": getattr(exc, "category", type(exc).__name__),
                "last_gpu_sample": getattr(exc, "sample", None),
                "query_count": self._session_query_count,
            }
            raise
        self._server_context = context
        self._slot.active = server
        return server

    def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        """Expose the staged GPU query path through the common retriever contract."""
        return self(query, top_k)
