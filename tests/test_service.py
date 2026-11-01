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
from page_cite_rag.retrieval import BM25Retriever, RetrievalError, build_bm25_index
from page_cite_rag.service import (
    DocumentService,
    ServiceBusyError,
    ServiceConfig,
    ServiceDeadlineError,
    ServiceStoppedError,
)
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.workflows import WorkflowConfig
from test_answer_cli import FakeClient


def test_service_and_base_workflow_keep_the_same_explanation_defaults(tmp_path):
    config = ServiceConfig(
        database_path=tmp_path / "documents.sqlite3",
        document_id="0" * 64,
    )

    assert (config.top_k, config.neighbor_blocks) == (100, 1)
    assert (WorkflowConfig().top_k, WorkflowConfig().neighbor_blocks) == (6, 1)


def test_public_api_defaults_to_traditional_chinese():
    from page_cite_rag.api import QuestionRequest

    assert QuestionRequest(question="為什麼結果不顯著？").language == "zh-Hant"


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "synthetic.txt"
    source.write_text(
        "The result was not significant unless n > 64. Children were excluded.\n\n"
        "The trial enrolled 120 adults.",
        encoding="utf-8",
    )
    parsed = prepare_source(source, usage_scope="self-authored-test", language="en")
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


def test_service_reading_uses_shared_navigation_and_returns_its_trace(prepared, monkeypatch):
    from page_cite_rag import application
    from page_cite_rag.dense import build_dense_index
    from page_cite_rag.reading_plan import ReadingNeed, ReadingPlan
    from test_dense import FakeEncoder

    database, document_id, _ = prepared
    store = SQLiteDocumentStore(database)
    encoder = FakeEncoder()
    build_dense_index(store, document_id, encoder)
    calls = []

    def shared_reading(workflow, question, **kwargs):
        calls.append((workflow.document_id, question, kwargs))
        return application.AgentAnswer(
            workflow.ask(question, language=kwargs["language"]),
            {"version": "source-navigation/v1", "agent": {"tool_calls_recorded": 1}},
        )

    monkeypatch.setattr(application, "ask_with_agent", shared_reading)
    monkeypatch.setattr(
        application,
        "plan_document_question",
        lambda *_args, **_kwargs: ReadingPlan(
            requirements=(ReadingNeed(need="result", queries=("result",)),)
        ),
    )
    config = GenerationConfig()
    with DocumentService(
        ServiceConfig(
            database_path=database,
            document_id=document_id,
            method="hybrid",
            reading_agent=True,
        ),
        config,
        lambda: FakeClient(config),
        encoder=encoder,
    ) as service:
        answer = service.submit("ask", "significant")[1].result(timeout=5)
        assert answer.trace.reading_trace["version"] == "source-navigation/v1"
        assert calls[0][2]["delegated"] is False
        assert calls[0][2].get("max_model_calls") is None
        with pytest.raises(ValueError, match="already uses planned query routes"):
            service.submit("ask", "significant", rewrite_query=True)


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


def test_serve_accepts_agent_with_native_reranker(prepared, monkeypatch):
    import uvicorn

    from page_cite_rag import api, cli, gpu_retrieval

    database, document_id, _ = prepared
    captured = {}

    class FakeStage:
        def __init__(self, *_args, reranker_model_path, **_kwargs):
            self.reranker_model_path = reranker_model_path
            captured["reranker"] = reranker_model_path

        def search(self, question, *, top_k=5):
            raise AssertionError("startup must not execute retrieval")

    def run(service, **_kwargs):
        captured["agent_enabled"] = service.config.reading_agent

    monkeypatch.setenv("PAGE_CITE_RAG_API_TOKEN", "test-local-token")
    monkeypatch.setattr(gpu_retrieval, "StagedGpuSearch", FakeStage)
    monkeypatch.setattr(api, "create_app", lambda service, **_kwargs: service)
    monkeypatch.setattr(uvicorn, "run", run)
    result = cli.main(
        [
            "serve",
            document_id,
            "--db",
            str(database),
            "--method",
            "hybrid",
            "--embedding-backend",
            "native-gpu",
            "--device",
            "cuda",
            "--gpu-uuid",
            "GPU-12345678-1234-1234-1234-123456789abc",
            "--agent",
            "--rerank",
        ]
    )
    assert result == 0
    assert captured["agent_enabled"] is True
    assert captured["reranker"] is not None


@pytest.mark.parametrize("temperature_limit", [None, 88])
def test_serve_native_gpu_temperature_policy_reaches_both_backends(
    prepared, monkeypatch, temperature_limit
):
    import uvicorn

    from page_cite_rag import api, cli, gpu_retrieval, llama

    database, document_id, blocks = prepared
    lexical = BM25Retriever(SQLiteDocumentStore(database, read_only=True), document_id)
    expected_limit = temperature_limit or 88
    embedding_limits = []
    clients = []

    class FakeStage:
        def __init__(self, *_args, gpu_temperature_limit_c, **_kwargs):
            embedding_limits.append(gpu_temperature_limit_c)
            self.last_trace = None
            self.reranker_model_path = None

        def search(self, question, *, top_k=5):
            self.last_trace = {
                "gpu_temperature_limit_c": embedding_limits[-1],
                "last_gpu_sample": {"temperature_c": 65},
                "query_count": 1,
            }
            return lexical.search(question, top_k=top_k).model_copy(update={"method": "dense"})

    def open_client(config, *, budget_policy):
        assert budget_policy == "exact"
        clients.append(FakeClient(config))
        return clients[-1]

    def run(service, **_kwargs):
        with service:
            answer = service.submit("ask", "result significant")[1].result(timeout=5)
            trace = json.loads(answer.model_dump_json())["trace"]["gpu_retrieval_trace"]
            assert trace["gpu_temperature_limit_c"] == expected_limit
            assert trace["last_gpu_sample"] == {"temperature_c": 65}
            assert trace["query_count"] == 1
            explanation = service.submit(
                "explain", "Explain this source.", block_id=blocks[0].block_id
            )[1].result(timeout=5)
            assert explanation.trace.gpu_retrieval_trace is None

    monkeypatch.setenv("PAGE_CITE_RAG_API_TOKEN", "test-local-token")
    monkeypatch.setattr(gpu_retrieval, "StagedGpuSearch", FakeStage)
    monkeypatch.setattr(llama, "LlamaModelClient", open_client)
    monkeypatch.setattr(api, "create_app", lambda service, **_kwargs: service)
    monkeypatch.setattr(uvicorn, "run", run)
    arguments = [
        "serve",
        document_id,
        "--db",
        str(database),
        "--method",
        "dense",
        "--embedding-backend",
        "native-gpu",
        "--device",
        "cuda",
        "--gpu-uuid",
        "GPU-12345678-1234-1234-1234-123456789abc",
    ]
    if temperature_limit is not None:
        arguments.extend(["--gpu-temperature-limit", str(temperature_limit)])
    assert cli.main(arguments) == 0
    assert embedding_limits == [expected_limit]
    assert len(clients) == 1
    assert clients[0].config.max_gpu_temperature_c == expected_limit
    assert clients[0].close_count == 1


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
        result = service.submit("ask", "result significant", language="en")[1].result(timeout=5)
        assert result.result.answer == "The result was not significant unless n > 64."
        source = result.result.claims[0].citations[0]
        assert source.quote in service.source(source.block_id).text
        trace = json.loads(result.model_dump_json())["trace"]
        assert trace["retrieval_mode"] == "resident"
        assert trace["gpu_retrieval_trace"] is None


def test_service_passes_and_clears_one_total_request_deadline(prepared):
    class DeadlineClient(FakeClient):
        def __init__(self, config):
            super().__init__(config)
            self.deadlines = []
            self.cleared = 0

        def set_request_deadline(self, deadline):
            self.deadlines.append(deadline)

        def clear_request_deadline(self):
            self.cleared += 1

    service, clients = make_service(prepared, factory=DeadlineClient)
    with service:
        _request_id, future = service.submit("ask", "result significant", language="en")
        answer = future.result(timeout=5)

    assert answer.result.status == "answered"
    assert len(clients) == 1
    assert len(clients[0].deadlines) == 1
    assert clients[0].deadlines[0] > time.monotonic()
    assert clients[0].cleared == 1


def test_resident_english_rewrite_option_uses_one_bounded_query_call(prepared):
    service, clients = make_service(prepared)
    with service:
        answer = service.submit("ask", "result significant", rewrite_query=True)[1].result(
            timeout=5
        )
        assert answer.result.status == "answered"
        assert answer.result.retrieval.query_plan["status"] == "rewritten"
        assert answer.result.retrieval.query_plan["lexical_query"] == (
            "result significant document, evidence"
        )
        assert len(clients) == 1 and len(clients[0].calls) == 2


def test_api_opt_in_chinese_rewrite_keeps_original_question_and_records_cost(prepared):
    class RewriteClient(FakeClient):
        def generate(self, messages, schema_class):
            if schema_class is None:
                self.calls.append((messages, schema_class))
                return GenerationResult(
                    "explain, result",
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
                    check_budget(messages, None, self.config),
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
    assert retrieval["query_plan"]["lexical_query"] == "請說明 result explain, result"
    assert retrieval["query_plan"]["status"] == "rewritten"
    assert retrieval["query_plan"]["stats"]["eval_count"] == 10
    assert len(clients) == 1 and len(clients[0].calls) == 2
    assert clients[0].close_count == 1


@pytest.mark.parametrize("docling_source", [False, True])
def test_staged_rewrite_runs_once_before_native_search(prepared, docling_source):
    database, document_id, _ = prepared
    clients = []
    searches = []

    def model_factory():
        clients.append(FakeClient(GenerationConfig()))
        return clients[-1]

    class StagedSearch:
        reranker_model_path = None
        last_trace = None

        def __init__(self):
            self.lexical = BM25Retriever(SQLiteDocumentStore(database, read_only=True), document_id)

        def search_with_lexical_query(self, question, lexical_query, *, top_k):
            searches.append((question, lexical_query, top_k))
            return self.lexical.search(lexical_query, top_k=top_k).model_copy(
                update={"query": question, "method": "hybrid"}
            )

        def search(self, question, *, top_k=5):
            searches.append((question, question, top_k))
            return self.lexical.search(question, top_k=top_k).model_copy(
                update={"method": "hybrid"}
            )

    service = DocumentService(
        ServiceConfig(database_path=database, document_id=document_id, method="hybrid"),
        GenerationConfig(),
        model_factory,
        staged_search=StagedSearch(),
    )
    # A Docling document with staged GPU retrieval has no resident CPU retriever.
    service._docling_source = docling_source
    with service:
        answer = service.submit("ask", "請說明 result", rewrite_query=True)[1].result(timeout=5)
        assert answer.result.status == "answered"
        with pytest.raises(ValueError, match="BM25 or hybrid ask"):
            service.submit("explain", "result", rewrite_query=True)
    assert len(clients) == 1
    assert len(searches) == 1
    assert searches[0][0] == "請說明 result"
    assert searches[0][1].startswith("請說明 result ")


def test_staged_retrieval_coexists_with_the_resident_owned_generator(prepared):
    database, document_id, _ = prepared
    config = GenerationConfig()
    clients = []
    search_checks = []
    store = SQLiteDocumentStore(database, read_only=True)
    lexical = BM25Retriever(store, document_id)

    class Stage:
        reranker_model_path = None

        def __init__(self):
            self.last_trace = {
                "gpu_temperature_limit_c": 88,
                "last_gpu_sample": {"temperature_c": 65},
            }

        def search(self, question, *, top_k=5):
            search_checks.append(not clients or clients[-1].close_count == 0)
            self.last_trace["last_gpu_sample"]["temperature_c"] += 1
            return lexical.search(question, top_k=top_k).model_copy(update={"method": "dense"})

    stage = Stage()

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
        assert second.trace.load_count == 1
        assert first.trace.retrieval_mode == second.trace.retrieval_mode == "staged"
        assert first.trace.gpu_retrieval_trace["last_gpu_sample"]["temperature_c"] == 66
        assert second.trace.gpu_retrieval_trace["last_gpu_sample"]["temperature_c"] == 67
        assert search_checks == [True, True]
    assert all(client.close_count == 1 for client in clients)


def test_staged_retrieval_receives_deadline_cancel_with_generator_loaded(prepared):
    database, document_id, _ = prepared
    started = threading.Event()
    cancelled = threading.Event()
    config = GenerationConfig()
    clients = []

    class BlockingStage:
        reranker_model_path = None
        last_trace = None

        def search(self, question, *, top_k=5):
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
    assert len(clients) == 1 and clients[0].close_count == 1


def test_owned_native_server_checks_external_cancel_signal():
    server = object.__new__(NativeEmbeddingServer)
    server._cancel_event = threading.Event()
    server._check_cancelled()
    server._cancel_event.set()
    with pytest.raises(RetrievalError, match="cancelled"):
        server._check_cancelled()
