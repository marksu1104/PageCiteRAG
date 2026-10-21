from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import pytest

import page_cite_rag.gpu_retrieval as gpu_retrieval


class FakeSearch:
    def __init__(self):
        self.queries = []
        self.scoped_queries = []

    def search(self, query, *, top_k):
        self.queries.append((query, top_k))
        return SimpleNamespace(elapsed_ms=2.0)

    def search_scoped(self, query, *, allowed_block_ids, candidate_k, top_k):
        self.scoped_queries.append((query, allowed_block_ids, candidate_k, top_k))
        return SimpleNamespace(elapsed_ms=3.0)


class FakeEmbeddingServer:
    instances = []

    def __init__(self, *_args, **kwargs):
        self.startup_ms = 12.0
        self.closed = False
        self.diagnostics_value = {"gpu_temperature_limit_c": kwargs["max_gpu_temperature_c"]}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.closed = True

    def diagnostics(self):
        return self.diagnostics_value


def build_search(monkeypatch):
    FakeEmbeddingServer.instances.clear()
    search = FakeSearch()
    monkeypatch.setattr(gpu_retrieval, "verify_native_model", lambda _path: None)
    monkeypatch.setattr(gpu_retrieval, "NativeEmbeddingEncoder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gpu_retrieval, "NativeEmbeddingServer", FakeEmbeddingServer)
    monkeypatch.setattr(gpu_retrieval, "document_retriever", lambda *_args, **_kwargs: search)
    staged = gpu_retrieval.StagedGpuSearch(
        object(), "doc", "hybrid", Path("embedding.gguf"), "GPU-test"
    )
    return staged, search


def test_explicit_request_session_reuses_one_embedding_server(monkeypatch):
    staged, search = build_search(monkeypatch)

    with staged.session():
        staged("first question", 10)
        staged.search("follow-up question", top_k=10)
        assert staged.last_trace["query_count"] == 2

    assert len(FakeEmbeddingServer.instances) == 1
    assert FakeEmbeddingServer.instances[0].closed
    assert search.queries == [("first question", 10), ("follow-up question", 10)]
    assert staged._slot.active is None
    assert staged.last_trace["query_count"] == 2
    assert staged.last_trace["gpu_temperature_limit_c"] == 80


def test_request_session_allows_context_propagated_tool_worker(monkeypatch):
    staged, search = build_search(monkeypatch)

    with staged.session():
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(context.run, staged, "reader question", 10).result()

    assert result.elapsed_ms == 2.0
    assert len(FakeEmbeddingServer.instances) == 1
    assert FakeEmbeddingServer.instances[0].closed
    assert search.queries == [("reader question", 10)]
    assert staged.last_trace["query_count"] == 1
    assert staged._slot.active is None
    assert staged._session_token is None


def test_scoped_search_uses_backend_scope_api_inside_context_propagated_worker(monkeypatch):
    staged, search = build_search(monkeypatch)

    with staged.session():
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                context.run,
                staged.search_scoped,
                "reader question",
                allowed_block_ids={"allowed-block"},
                candidate_k=468,
                top_k=10,
            ).result()

    assert result.elapsed_ms == 3.0
    assert search.scoped_queries == [("reader question", {"allowed-block"}, 468, 10)]
    assert search.queries == []
    assert staged.last_trace["query_count"] == 1
    assert FakeEmbeddingServer.instances[0].closed


def test_scoped_docling_search_maps_source_ids_to_complete_native_chunks(monkeypatch):
    staged, _search = build_search(monkeypatch)
    calls = []

    class FakeDoclingRetriever:
        store = SimpleNamespace(chunk_block_ids=[("a", "b"), ("c",), ("d", "e")])
        index_id = "docling-index"

        def search(self, query, *, top_k, allowed_chunk_indices=None):
            calls.append((query, top_k, allowed_chunk_indices))
            return SimpleNamespace(elapsed_ms=4.0)

    staged.retriever = FakeDoclingRetriever()
    with staged.session():
        result = staged.search_scoped(
            "reader question",
            allowed_block_ids={"a", "c", "d"},
            candidate_k=468,
            top_k=10,
        )

    assert result.elapsed_ms == 4.0
    assert calls == [("reader question", 10, (1,))]
    assert staged.last_trace["query_count"] == 1
    assert FakeEmbeddingServer.instances[0].closed


def test_request_session_rejects_unbound_worker_context(monkeypatch):
    staged, _search = build_search(monkeypatch)

    with staged.session():
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(staged, "foreign request", 10)
            with pytest.raises(RuntimeError, match="belongs to another request"):
                future.result()

    assert FakeEmbeddingServer.instances == []
    assert staged.last_trace["server_started"] is False
    assert staged._session_token is None


def test_parallel_context_propagated_tools_share_one_serialized_search_session(monkeypatch):
    staged, search = build_search(monkeypatch)
    active = 0
    max_active = 0
    activity_lock = threading.Lock()

    def observe_serial_search(query, *, top_k):
        nonlocal active, max_active
        with activity_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with activity_lock:
            active -= 1
        search.queries.append((query, top_k))
        return SimpleNamespace(elapsed_ms=2.0)

    monkeypatch.setattr(search, "search", observe_serial_search)
    with staged.session():
        contexts = [copy_context(), copy_context()]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(context.run, staged, f"question {index}", 10)
                for index, context in enumerate(contexts)
            ]
            [future.result() for future in futures]

    assert max_active == 1
    assert len(FakeEmbeddingServer.instances) == 1
    assert FakeEmbeddingServer.instances[0].closed
    assert len(search.queries) == 2
    assert staged.last_trace["query_count"] == 2


def test_request_without_a_retrieval_call_does_not_start_embedding_server(monkeypatch):
    staged, _search = build_search(monkeypatch)

    with staged.session():
        pass

    assert FakeEmbeddingServer.instances == []
    assert staged.last_trace["server_started"] is False
    assert staged.last_trace["server_start_attempted"] is False
    assert staged.last_trace["query_count"] == 0


def test_failed_guarded_start_retains_safe_sample_and_cleanup_trace(monkeypatch):
    staged, _search = build_search(monkeypatch)

    class GuardError(RuntimeError):
        category = "gpu_temperature_limit"
        sample = {"memory_mib": 5000, "temperature_c": 89, "pids": [1234], "phase": "guard"}

    def fail_start(*_args, **_kwargs):
        raise GuardError("guard stopped before embedding server startup")

    monkeypatch.setattr(gpu_retrieval, "NativeEmbeddingServer", fail_start)
    with pytest.raises(GuardError):
        with staged.session():
            staged("question", 10)

    assert staged.last_trace["server_started"] is False
    assert staged.last_trace["server_start_attempted"] is True
    assert staged.last_trace["guard_error"] == "gpu_temperature_limit"
    assert staged.last_trace["last_gpu_sample"]["temperature_c"] == 89
    assert staged.last_trace["cleanup_ms"] >= 0


def test_request_session_cleans_up_when_a_query_fails(monkeypatch):
    staged, search = build_search(monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic retrieval failure")

    monkeypatch.setattr(search, "search", fail)
    with pytest.raises(RuntimeError, match="synthetic retrieval failure"):
        with staged.session():
            staged("question", 10)

    assert len(FakeEmbeddingServer.instances) == 1
    assert FakeEmbeddingServer.instances[0].closed
    assert staged._slot.active is None
    assert staged._current_cancel is None
