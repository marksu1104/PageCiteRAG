"""F2 local service controls use self-authored text and no real GPU process."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
import pytest

from page_cite_rag.api import create_app
from page_cite_rag.embedding_server import NativeEmbeddingServer
from page_cite_rag.generation import (
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    check_budget,
)
from page_cite_rag.ingest import prepare_source
from page_cite_rag.query import EnglishQuery
from page_cite_rag.retrieval import BM25Retriever, RetrievalError, build_bm25_index
from page_cite_rag.service import (
    DocumentService,
    ServiceBusyError,
    ServiceConfig,
    ServiceDeadlineError,
    ServiceStoppedError,
)
from page_cite_rag.store import SQLiteDocumentStore
from test_answer_cli import FakeClient


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "synthetic.txt"
    source.write_text(
        "The result was not significant unless n > 64. Children were excluded.\n\n"
        "The trial enrolled 120 adults.",
        encoding="utf-8",
    )
    parsed = prepare_source(source, usage_scope="self-authored-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(parsed.document, parsed.iter_units())
    build_bm25_index(store, parsed.document.document_id)
    blocks = store.get_blocks(parsed.document.document_id)
    return store.database_path, parsed.document.document_id, blocks


def make_service(prepared, *, idle_seconds=300, queue_capacity=2, factory=None):
    database, document_id, _ = prepared
    config = GenerationConfig()
    clients = []

    def model_factory():
        client = factory(config) if factory else FakeClient(config)
        clients.append(client)
        return client

    service = DocumentService(
        ServiceConfig(
            database_path=database,
            document_id=document_id,
            idle_seconds=idle_seconds,
            queue_capacity=queue_capacity,
        ),
        config,
        model_factory,
    )
    return service, clients


def test_service_reuses_one_model_then_releases_owned_model_on_idle(prepared):
    service, clients = make_service(prepared, idle_seconds=0.08)
    with service:
        first = service.submit("ask", "result significant")[1].result(timeout=5)
        second = service.submit("ask", "children excluded")[1].result(timeout=5)
        assert first.result.status == second.result.status == "answered"
        assert first.trace.load_count == second.trace.load_count == 1
        assert first.trace.startup_ms >= 0
        assert second.trace.startup_ms == 0
        assert len(clients) == 1
        deadline = time.monotonic() + 3
        while not clients[0].close_count and time.monotonic() < deadline:
            time.sleep(0.01)
        assert clients[0].close_count == 1
        third = service.submit("ask", "result significant")[1].result(timeout=5)
        assert third.trace.load_count == 2
        assert len(clients) == 2
    assert clients[1].close_count == 1
    with pytest.raises(ServiceStoppedError):
        service.submit("ask", "result significant")


def test_queue_full_cancel_waiter_and_active_owned_model_only(prepared):
    entered = threading.Event()
    release = threading.Event()

    class BlockingClient(FakeClient):
        def generate(self, messages, schema_class):
            entered.set()
            assert release.wait(timeout=5)
            return super().generate(messages, schema_class)

    service, clients = make_service(
        prepared, queue_capacity=1, factory=lambda config: BlockingClient(config)
    )
    with service:
        first_id, first = service.submit("ask", "result significant")
        assert entered.wait(timeout=3)
        second_id, second = service.submit("ask", "children excluded")
        with pytest.raises(ServiceBusyError):
            service.submit("ask", "result significant")
        assert service.cancel(second_id)
        with pytest.raises(ServiceStoppedError):
            second.result(timeout=1)
        assert not service.cancel(second_id)
        release.set()
        assert first.result(timeout=5).trace.request_id == first_id
    assert clients[0].close_count == 1


def test_deadline_closes_owned_model_and_reports_timeout(prepared):
    entered = threading.Event()
    released = threading.Event()

    class SlowClient(FakeClient):
        def generate(self, messages, schema_class):
            entered.set()
            assert released.wait(timeout=3)
            raise AssertionError("closed request must not produce an answer")

        def close(self):
            super().close()
            released.set()

    service, clients = make_service(prepared, factory=lambda config: SlowClient(config))
    service.config = service.config.model_copy(update={"request_timeout_seconds": 1.05})
    with service:
        future = service.submit("ask", "result significant")[1]
        assert entered.wait(timeout=3)
        with pytest.raises(ServiceDeadlineError):
            future.result(timeout=3)
        assert clients[0].close_count >= 1


def test_api_bound_document_source_openapi_and_remote_denial(prepared):
    service, clients = make_service(prepared)
    _, document_id, blocks = prepared
    app = create_app(service, api_token="test-local-token", allow_test_client=True)

    async def check_local():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx.AsyncClient(
                transport=transport, base_url="http://127.0.0.1"
            ) as anonymous:
                assert (await anonymous.get("/health")).status_code == 401
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1",
                headers={"Authorization": "Bearer test-local-token"},
            ) as client:
                assert (
                    await client.get("/health", headers={"Authorization": "Bearer wrong-token"})
                ).status_code == 401
                assert (await client.get("/health")).json()["model_loaded"] is False
                document = await client.get("/v1/document")
                assert document.json()["document"]["document_id"] == document_id
                first_page = await client.get("/v1/blocks", params={"limit": 1})
                assert first_page.status_code == 200
                assert first_page.json()["blocks"][0]["block_id"] == blocks[0].block_id
                assert first_page.json()["next_after"] == blocks[0].ordinal
                second_page = await client.get(
                    "/v1/blocks", params={"limit": 1, "after": blocks[0].ordinal}
                )
                assert second_page.json()["blocks"][0]["block_id"] == blocks[1].block_id
                assert second_page.json()["next_after"] is None
                assert (await client.get("/v1/blocks?limit=51")).status_code == 422
                assert (await client.get("/v1/blocks?after=-1")).status_code == 422
                assert clients == []
                source = await client.get(f"/v1/source/{blocks[0].block_id}")
                assert source.json()["text"] == blocks[0].text
                assert (await client.get("/v1/source/" + "0" * 64)).status_code == 404
                assert "/v1/ask" in (await client.get("/openapi.json")).json()["paths"]
                first = await client.post("/v1/ask", json={"question": "result significant"})
                second = await client.post(
                    "/v1/explain",
                    json={"question": "Explain this source.", "block_id": blocks[0].block_id},
                )
                assert first.status_code == second.status_code == 200
                assert first.json()["trace"]["load_count"] == 1
                assert second.json()["trace"]["load_count"] == 1
                assert first.json()["result"]["answer_claims"][0]["citations"]
                invalid = await client.post(
                    "/v1/ask", json={"question": "a", "source_path": "/secret"}
                )
                assert invalid.status_code == 422
                invalid_explain = await client.post(
                    "/v1/explain",
                    json={
                        "question": "Explain.",
                        "block_id": blocks[0].block_id,
                        "rewrite_query": True,
                    },
                )
                assert invalid_explain.status_code == 422

    asyncio.run(check_local())
    assert clients[0].close_count == 1
    remote_service, _ = make_service(prepared)
    remote_app = create_app(remote_service, api_token="test-local-token")

    async def check_remote():
        async with remote_app.router.lifespan_context(remote_app):
            transport = httpx.ASGITransport(app=remote_app, client=("198.51.100.7", 12345))
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://remote",
                headers={"Authorization": "Bearer test-local-token"},
            ) as remote:
                assert (await remote.get("/health")).status_code == 403

    asyncio.run(check_remote())


def test_serve_requires_environment_token_before_runtime_start(monkeypatch, capsys):
    from page_cite_rag.cli import main

    monkeypatch.delenv("PAGE_CITE_RAG_API_TOKEN", raising=False)
    assert main(["serve", "a" * 64]) == 1
    captured = capsys.readouterr()
    assert "local service failed" in captured.err
    assert "PAGE_CITE_RAG_API_TOKEN" not in captured.err
    with pytest.raises(ValueError, match="nonempty local API token"):
        create_app(None, api_token=" ")


def test_source_tamper_is_rejected_before_model_start(prepared):
    service, clients = make_service(prepared)
    _, _, blocks = prepared
    with service:
        with pytest.raises(ValueError):
            service.submit("explain", "Explain", block_id="../../secret")
        with pytest.raises(Exception):
            service.submit("explain", "Explain", block_id="0" * 64)
        assert service.source(blocks[0].block_id).text == blocks[0].text
        assert clients == []


def test_formal_api_answer_is_not_replaced_by_tool_output(prepared):
    service, _ = make_service(prepared)
    with service:
        result = service.submit("ask", "result significant")[1].result(timeout=5)
        assert result.result.answer == "The result was not significant unless n > 64."
        source = result.result.claims[0].citations[0]
        assert source.quote in service.source(source.block_id).text
        assert json.loads(result.model_dump_json())["trace"]["retrieval_mode"] == "resident"


def test_resident_english_rewrite_option_uses_no_extra_model_call(prepared):
    service, clients = make_service(prepared)
    with service:
        answer = service.submit("ask", "result significant", rewrite_query=True)[1].result(
            timeout=5
        )
        assert answer.result.status == "answered"
        assert answer.result.retrieval.query_plan["status"] == "not_needed"
        assert answer.result.retrieval.query_plan["lexical_query"] == "result significant"
        assert len(clients) == 1 and len(clients[0].calls) == 1


def test_api_opt_in_chinese_rewrite_keeps_original_question_and_records_cost(prepared):
    class RewriteClient(FakeClient):
        def generate(self, messages, schema_class):
            if schema_class is EnglishQuery:
                self.calls.append((messages, schema_class))
                return GenerationResult(
                    schema_class(query="Explain result"),
                    GenerationStats(
                        prompt_eval_count=50,
                        eval_count=10,
                        total_duration=100,
                        load_duration=None,
                        prompt_eval_duration=30,
                        eval_duration=70,
                        done_reason="stop",
                        elapsed_ms=1.0,
                    ),
                    check_budget(messages, schema_class.model_json_schema(), self.config),
                )
            return super().generate(messages, schema_class)

    service, clients = make_service(prepared, factory=RewriteClient)
    app = create_app(service, api_token="test-local-token", allow_test_client=True)

    async def ask():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1",
                headers={"Authorization": "Bearer test-local-token"},
            ) as client:
                response = await client.post(
                    "/v1/ask", json={"question": "請說明 result", "rewrite_query": True}
                )
                assert response.status_code == 200
                return response.json()

    response = asyncio.run(ask())
    retrieval = response["result"]["retrieval"]
    assert retrieval["query"] == "請說明 result"
    assert retrieval["query_plan"]["lexical_query"] == "Explain result"
    assert retrieval["query_plan"]["status"] == "rewritten"
    assert retrieval["query_plan"]["stats"]["eval_count"] == 10
    assert len(clients) == 1 and len(clients[0].calls) == 2
    assert clients[0].close_count == 1


def test_staged_rewrite_is_rejected_before_any_model_or_search(prepared):
    database, document_id, _ = prepared
    clients = []
    searches = []

    def model_factory():
        clients.append(FakeClient(GenerationConfig()))
        return clients[-1]

    def staged_search(question, top_k):
        searches.append((question, top_k))
        raise AssertionError("unsupported route must not start GPU search")

    service = DocumentService(
        ServiceConfig(database_path=database, document_id=document_id, method="hybrid"),
        GenerationConfig(),
        model_factory,
        staged_search=staged_search,
    )
    with service:
        with pytest.raises(ValueError, match="resident BM25 or hybrid"):
            service.submit("ask", "請說明 result", rewrite_query=True)
        with pytest.raises(ValueError, match="resident BM25 or hybrid"):
            service.submit("explain", "result", rewrite_query=True)
    assert clients == searches == []


def test_staged_retrieval_releases_prior_model_before_next_gpu_search(prepared):
    database, document_id, _ = prepared
    config = GenerationConfig()
    clients = []
    search_checks = []
    store = SQLiteDocumentStore(database, read_only=True)
    lexical = BM25Retriever(store, document_id)

    def stage(question, top_k):
        search_checks.append(not clients or clients[-1].close_count == 1)
        return lexical.search(question, top_k=top_k).model_copy(update={"method": "dense"})

    def factory():
        client = FakeClient(config)
        clients.append(client)
        return client

    service = DocumentService(
        ServiceConfig(database_path=database, document_id=document_id, method="dense"),
        config,
        factory,
        staged_search=stage,
    )
    with service:
        first = service.submit("ask", "result significant")[1].result(timeout=5)
        second = service.submit("ask", "result significant")[1].result(timeout=5)
        assert first.result.status == second.result.status == "answered"
        assert first.trace.load_count == 1
        assert second.trace.load_count == 2
        assert first.trace.retrieval_mode == second.trace.retrieval_mode == "staged"
        assert search_checks == [True, True]
    assert all(client.close_count == 1 for client in clients)


def test_staged_retrieval_receives_deadline_cancel_before_model_load(prepared):
    database, document_id, _ = prepared
    started = threading.Event()
    cancelled = threading.Event()
    config = GenerationConfig()
    clients = []

    class BlockingStage:
        def __call__(self, question, top_k):
            started.set()
            assert cancelled.wait(timeout=3)
            raise RuntimeError("owned embedding stage stopped")

        def cancel(self):
            cancelled.set()

    def factory():
        client = FakeClient(config)
        clients.append(client)
        return client

    service = DocumentService(
        ServiceConfig(
            database_path=database,
            document_id=document_id,
            method="dense",
            request_timeout_seconds=1.05,
        ),
        config,
        factory,
        staged_search=BlockingStage(),
    )
    with service:
        future = service.submit("ask", "result significant")[1]
        assert started.wait(timeout=2)
        with pytest.raises(ServiceDeadlineError):
            future.result(timeout=3)
    assert cancelled.is_set()
    assert clients == []


def test_owned_native_server_checks_external_cancel_signal():
    server = object.__new__(NativeEmbeddingServer)
    server._cancel_event = threading.Event()
    server._check_cancelled()
    server._cancel_event.set()
    with pytest.raises(RetrievalError, match="cancelled"):
        server._check_cancelled()
