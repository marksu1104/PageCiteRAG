"""Whole-document questions use the evidence agent with the coverage strategy; fakes only."""

import hashlib
from types import SimpleNamespace

import pytest

from page_cite_rag import (
    application,
    cli,
    evidence_agent,
    global_answer,
    gpu_retrieval,
    ollama_backend,
)
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL

# The multi-hop prompt every deep-mode result was measured with (08652ca) and the coverage
# prompt validated on LooGLE summaries (2026-11-16). Changing either needs a new evaluation.
CHAIN_SHA256 = "1d566f231aa2c8af7c661ebf5d10d785ca0d55aacda97968d0a9318ae31715ad"
COVER_SHA256 = "a38b9d5b54ee9653254efd91d989859907150768999fc2ac8b530f224e663e34"


def test_prompts_are_the_measured_ones():
    assert hashlib.sha256(evidence_agent.SYSTEM_PROMPT.encode()).hexdigest() == CHAIN_SHA256
    assert hashlib.sha256(evidence_agent.COVER_PROMPT.encode()).hexdigest() == COVER_SHA256
    assert evidence_agent.STRATEGIES == {
        "chain": evidence_agent.SYSTEM_PROMPT,
        "cover": evidence_agent.COVER_PROMPT,
    }


@pytest.mark.parametrize("strategy", ["chain", "cover"])
def test_agent_gets_the_strategy_prompt(monkeypatch, strategy):
    import langchain.agents

    seen = {}

    def create_agent(llm, tools, *, system_prompt, middleware):
        seen["prompt"] = system_prompt
        return object()

    monkeypatch.setattr(langchain.agents, "create_agent", create_agent)
    state = SimpleNamespace(model_call_limit=6, handoff_findings=False, tools=lambda: [])
    evidence_agent.evidence_agent(object(), state, strategy=strategy)
    expected = evidence_agent.STRATEGIES[strategy].replace("at most 8 steps", "at most 6 steps")
    assert seen["prompt"] == expected and "at most 6 steps" in expected


def test_unknown_strategy_rejected():
    state = SimpleNamespace(model_call_limit=8, handoff_findings=False, tools=lambda: [])
    with pytest.raises(ValueError, match="strategy must be chain or cover"):
        evidence_agent.evidence_agent(object(), state, strategy="plan")


class FakeSearch:
    def __init__(self, reranker=None, candidates=None):
        self.method = "hybrid"
        self.reranker_model_path = reranker
        self.candidates = candidates
        self.view = None

    def agent_view(self):
        return self if self.view is None else self.view


class FakeOllama:
    pass


def capable(monkeypatch, retriever, fits=False):
    monkeypatch.setattr(gpu_retrieval, "StagedGpuSearch", FakeSearch)
    monkeypatch.setattr(ollama_backend, "OllamaGgufClient", FakeOllama)
    workflow = SimpleNamespace(_retriever=retriever, _model=FakeOllama(), writer="quote_first")
    workflow.whole_document_fits = lambda question, *, language: fits
    return workflow


def test_global_question_uses_cover_agent_when_available(monkeypatch):
    workflow = capable(monkeypatch, FakeSearch())
    calls = []

    def agent_answer(passed, question, language, trace, *, strategy):
        calls.append((passed, strategy, trace["route"], trace["reason"]))
        return application.AgentAnswer("result", trace)

    monkeypatch.setattr(application, "_agent_answer", agent_answer)
    monkeypatch.setattr(global_answer, "answer_global", lambda *a, **kw: pytest.fail("map"))
    answer = application.answer_question(workflow, "Summarize it.", mode="global", language="en")
    assert answer.result == "result"
    assert calls == [(workflow, "cover", "agent_cover", "whole_document_exceeds_context")]


def test_global_auto_route_uses_the_agent_retrieval(monkeypatch):
    # Auto mode reranks; the agent must search as in deep mode (no reranker).
    reranked = FakeSearch(reranker="reranker.gguf", candidates=100)
    reranked.view = FakeSearch()
    workflow = capable(monkeypatch, reranked)
    monkeypatch.setattr(application, "choose_mode", lambda client, q: ("global", {"mode": "g"}))
    seen = []

    def agent_answer(passed, question, language, trace, *, strategy):
        seen.append((passed._retriever, strategy))
        return application.AgentAnswer("result", trace)

    monkeypatch.setattr(application, "_agent_answer", agent_answer)
    application.answer_question(workflow, "Main idea?", mode="auto", language="en")
    assert seen == [(reranked.view, "cover")] and workflow._retriever is reranked


@pytest.mark.parametrize("setup", ["fits", "cpu", "structured"])
def test_global_falls_back_to_map_reduce(monkeypatch, setup):
    workflow = capable(monkeypatch, FakeSearch(), fits=setup == "fits")
    if setup == "cpu":
        workflow._retriever = object()
    if setup == "structured":
        workflow.writer = "structured"
    monkeypatch.setattr(application, "_agent_answer", lambda *a, **kw: pytest.fail("agent"))
    monkeypatch.setattr(
        global_answer,
        "answer_global",
        lambda passed, q, *, language: global_answer.GlobalAnswer("mapped", {"version": "g"}),
    )
    answer = application.answer_question(workflow, "Summarize it.", mode="global", language="en")
    assert answer.result == "mapped" and answer.trace["route"] == "global"


def test_deep_mode_keeps_the_chain_strategy(monkeypatch):
    workflow = capable(monkeypatch, FakeSearch())
    seen = []

    def agent_answer(passed, question, language, trace, *, strategy):
        seen.append((strategy, trace["route"]))
        return application.AgentAnswer("result", trace)

    monkeypatch.setattr(application, "_agent_answer", agent_answer)
    application.answer_question(workflow, "Who founded it?", mode="deep", language="en")
    assert seen == [("chain", "agent")]


def test_deep_request_on_a_reranking_service_uses_the_agent_retrieval(monkeypatch):
    # serve/mcp rerank for auto mode; a per-request deep question must not fail on that and
    # must search without the reranker, as the agent was validated.
    reranked = FakeSearch(reranker="reranker.gguf", candidates=100)
    reranked.view = FakeSearch()
    workflow = capable(monkeypatch, reranked)
    seen = []

    def agent_answer(passed, question, language, trace, *, strategy):
        seen.append((passed._retriever, strategy, trace["route"]))
        return application.AgentAnswer("result", trace)

    monkeypatch.setattr(application, "_agent_answer", agent_answer)
    application.answer_question(workflow, "Who founded it?", mode="deep", language="en")
    assert seen == [(reranked.view, "chain", "agent")] and workflow._retriever is reranked


def test_agent_view_drops_reranking_only(monkeypatch):
    built = []

    def init(self, store, document_id, method, model_path, gpu_uuid, **kwargs):
        built.append((store, document_id, method, model_path, gpu_uuid, kwargs))

    def make(reranker, candidates):
        search = object.__new__(gpu_retrieval.StagedGpuSearch)
        search.__dict__.update(
            _store="store",
            document_id="doc",
            method="hybrid",
            model_path="m.gguf",
            gpu_uuid="GPU-1",
            tokenizer_directory="tok",
            gpu_temperature_limit_c=88,
            reranker_model_path=reranker,
            candidates=candidates,
        )
        return search

    plain = make(None, FORMAL_RETRIEVAL.candidates_per_branch)
    assert plain.agent_view() is plain and make(None, None).agent_view().candidates is None
    monkeypatch.setattr(gpu_retrieval.StagedGpuSearch, "__init__", init)
    make("reranker.gguf", 100).agent_view()
    assert built == [
        (
            "store",
            "doc",
            "hybrid",
            "m.gguf",
            "GPU-1",
            {"tokenizer_directory": "tok", "gpu_temperature_limit_c": 88},
        )
    ]


def test_route_line_names_the_coverage_agent(capsys):
    cli._print_route(
        {
            "mode": "global",
            "route": "agent_cover",
            "model_calls": 6,
            "search_steps": ["a", "b", "c", "d"],
        }
    )
    assert capsys.readouterr().out.strip() == (
        "Mode: global document answer (coverage agent, 6 model calls, 4 searches)."
    )
