from __future__ import annotations

from types import SimpleNamespace

import pytest

from page_cite_rag.agent import _search_preview
from page_cite_rag.tokenize import LexicalTokenizer


def test_preview_uses_relevant_original_sentence_and_marks_cross_language_fallback():
    tokenizer = LexicalTokenizer()
    source = "Background. " + ("Unrelated text. " * 30) + "Accuracy is 92 percent."
    preview, kind = _search_preview(source, "accuracy 92", tokenizer)
    assert preview == "Accuracy is 92 percent."
    assert preview in source and kind == "lexical_match"
    preview, kind = _search_preview("A document in English.", "中文問題", tokenizer)
    assert preview == "A document in English." and kind == "prefix_fallback"


def test_quote_verification_tolerates_whitespace_and_script_but_not_invention():
    from page_cite_rag.span_citations import quote_is_verbatim

    source = "教育领域的人工智能\n必须遵守人权标准。"
    assert quote_is_verbatim("教育領域的人工智能 必須遵守人權標準", source)
    assert not quote_is_verbatim("人工智能可以取代教師", source)
    assert not quote_is_verbatim("教育", source)  # too short to verify anything


@pytest.mark.parametrize("route,reason", [("auto", "rerank_unavailable"), ("fixed", "requested")])
def test_routes_use_fixed_answer_without_planning(monkeypatch, route, reason):
    from page_cite_rag import application

    workflow = SimpleNamespace(whole_document_fits=lambda *a, **kw: False, _retriever=None)
    expected = object()
    calls = []

    def ask(workflow_arg, question, *, language):
        calls.append((workflow_arg, question, language))
        return expected

    monkeypatch.setattr(application, "ask_document", ask)
    answer = application.answer_question(workflow, "What happened?", language="auto", route=route)
    assert calls == [(workflow, "What happened?", "en")]
    assert answer.result is expected
    assert answer.trace == {
        "version": "routed-answer/v5",
        "requested_mode": "detail",
        "mode": "detail",
        "requested_route": route,
        "route": "fixed",
        "reason": reason,
    }


def test_retired_route_is_rejected():
    from page_cite_rag.application import ROUTES, answer_question

    assert ROUTES == ("auto", "fixed", "agent")
    with pytest.raises(ValueError, match="route must be auto, fixed or agent"):
        answer_question(object(), "question", route="whole-document")


def test_retired_writer_is_rejected(tmp_path):
    from page_cite_rag.application import document_workflow
    from tests.reading_fixtures import _setup

    workflow, _, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="^unsupported writer$"):
        document_workflow(
            workflow._store,
            workflow.document_id,
            workflow._retriever,
            workflow._model,
            workflow._budget_config,
            writer="quote_two_step",
        )


def test_service_rejects_retired_route(tmp_path):
    from pydantic import ValidationError

    from page_cite_rag.service import ServiceConfig

    for route in ("auto", "detail", "global", "deep"):
        assert (
            ServiceConfig(database_path=tmp_path / "db", document_id="0" * 64, mode=route).mode
            == route
        )
    for retired in ("fixed", "agent", "whole-document"):
        with pytest.raises(ValidationError):
            ServiceConfig(database_path=tmp_path / "db", document_id="0" * 64, mode=retired)
    with pytest.raises(ValidationError):
        ServiceConfig(database_path=tmp_path / "db", document_id="0" * 64, route="auto")


@pytest.mark.parametrize("writer", ["structured", "quote_two_step"])
def test_evidence_agent_rejects_other_writers(writer):
    from page_cite_rag.evidence_agent import run_evidence_agent
    from page_cite_rag.full_answer import FullAnswerWorkflow

    workflow = object.__new__(FullAnswerWorkflow)
    workflow.writer = writer
    with pytest.raises(
        ValueError, match="^evidence agent requires a quote-first FullAnswerWorkflow$"
    ):
        run_evidence_agent(workflow, None, "question", language="en")


@pytest.mark.parametrize("fits", [True, False])
def test_detail_selects_whole_document_or_reranked(monkeypatch, fits):
    from page_cite_rag import application
    from page_cite_rag.gpu_retrieval import StagedGpuSearch
    from page_cite_rag.workflows import WorkflowConfig

    retriever = object.__new__(StagedGpuSearch)
    retriever.reranker_model_path = "fake.gguf"
    blocks = [object(), object()]
    expected = object()
    calls = []

    def answer(question, sources, *, language):
        calls.append((question, sources, language))
        return expected

    workflow = SimpleNamespace(
        whole_document_fits=lambda *args, **kwargs: fits,
        answer_from_context=answer,
        _store=SimpleNamespace(get_blocks=lambda doc: blocks),
        document_id="doc",
        _retriever=retriever,
        config=WorkflowConfig(top_k=100),
    )

    def ask(selected, question, *, language):
        assert selected.config.top_k == 20
        assert selected._retriever is retriever
        calls.append((question, [], language))
        return expected

    monkeypatch.setattr(application, "ask_document", ask)
    result = application.answer_question(workflow, "question", language="en", mode="detail")
    assert result.result is expected
    assert result.trace == {
        "version": "routed-answer/v5",
        "requested_mode": "detail",
        "mode": "detail",
        "requested_route": "auto",
        "route": "whole_document_context" if fits else "reranked",
        "reason": "fits_context" if fits else "whole_document_exceeds_context",
    }
    assert calls == [("question", blocks if fits else [], "en")]
    assert workflow.config.top_k == 100


@pytest.mark.parametrize("bad", ["retriever", "method", "reranker", "client"])
def test_agent_requires_native_hybrid_without_rerank_and_owned_client(bad):
    from page_cite_rag.application import answer_question
    from page_cite_rag.gpu_retrieval import StagedGpuSearch

    retriever = object.__new__(StagedGpuSearch)
    retriever.method = "dense" if bad == "method" else "hybrid"
    retriever.reranker_model_path = "fake" if bad == "reranker" else None
    workflow = SimpleNamespace(
        _retriever=None if bad == "retriever" else retriever, _model=object()
    )
    requirement = {
        "retriever": "--native-gpu",
        "method": "--method hybrid",
        "reranker": "without reranking",
        "client": "--backend pinned-ollama",
    }
    with pytest.raises(ValueError, match=requirement[bad]):
        answer_question(workflow, "question", route="agent")


def test_agent_holds_session_and_reports_accounting(monkeypatch):
    from contextlib import contextmanager

    from page_cite_rag import application, evidence_agent
    from page_cite_rag.gpu_retrieval import StagedGpuSearch
    from page_cite_rag.ollama_backend import OllamaGgufClient

    events = []

    @contextmanager
    def session(self):
        events.append("start")
        yield self
        events.append("close")

    monkeypatch.setattr(StagedGpuSearch, "session", session)
    retriever = object.__new__(StagedGpuSearch)
    retriever.method = "hybrid"
    retriever.reranker_model_path = None
    client = object.__new__(OllamaGgufClient)
    workflow = SimpleNamespace(_retriever=retriever, _model=client)
    expected = object()

    def run(passed, passed_client, question, *, language):
        assert passed is workflow and passed_client is client
        assert question == "question" and language == "en"
        assert events == ["start"]
        return SimpleNamespace(
            answer=expected,
            trace={
                "version": "fake-agent/v1",
                "model_calls": [{}, {}],
                "tool_call_repairs": 1,
                "handoff_reason": "finished",
                "search_steps": ["q"],
                "token_usage": {"total": 10},
            },
        )

    monkeypatch.setattr(evidence_agent, "run_evidence_agent", run)
    result = application.answer_question(workflow, "question", language="en", mode="deep")
    assert events == ["start", "close"]
    assert result.result is expected
    assert result.trace == {
        "version": "routed-answer/v5",
        "requested_mode": "deep",
        "mode": "deep",
        "requested_route": "agent",
        "route": "agent",
        "reason": "requested",
        "agent_version": "fake-agent/v1",
        "model_calls": 2,
        "tool_call_repairs": 1,
        "handoff_reason": "finished",
        "search_steps": ["q"],
        "token_usage": {"total": 10},
    }
