"""Router and mode dispatch use fake clients only."""

from types import SimpleNamespace

import pytest

from page_cite_rag import application, global_answer
from page_cite_rag.generation import ModelError, ModelTimeoutError


@pytest.mark.parametrize(
    "reply,mode,error",
    [
        ("2", "global", None),
        ("Choice 1", "detail", None),
        ("(2) whole document", "global", None),
        ("no choice", "detail", "no_valid_choice"),
        ("2 then 1", "global", None),
        ("1" + "x" * 80, "detail", None),
    ],
)
def test_choose_mode_prompt_reply_and_accounting(reply, mode, error):
    calls = []

    def generate(messages, schema):
        calls.append((messages, schema))
        return SimpleNamespace(
            value=reply, stats=SimpleNamespace(model_dump=lambda **kwargs: {"eval_count": 3})
        )

    selected, record = application.choose_mode(SimpleNamespace(generate=generate), "Question?")
    assert selected == mode
    assert record == {"mode": mode, "reply": reply[:40], "error": error, "stats": {"eval_count": 3}}
    assert len(calls) == 1
    messages, schema = calls[0]
    assert schema is None and len(messages) == 1 and messages[0].role == "user"
    assert messages[0].content == (
        "Some choices are given below. It is provided in a numbered list (1 to 2), "
        "where each item in the list corresponds to a summary.\n---------------------\n"
        "(1) Useful for questions about specific details in the document: facts, numbers, "
        "definitions, methods, or explaining a particular concept or passage.\n\n"
        "(2) Useful for questions about the document as a whole: its main idea or summary, "
        "its contributions, strengths and limitations, or comparisons and lists that span "
        "many sections.\n---------------------\n"
        "Using only the choices above and not prior knowledge, return the choice that is "
        "most relevant to the question: 'Question?'\nReturn only the number of the choice."
    )


@pytest.mark.parametrize("failure", [ModelError("failed"), ModelTimeoutError("timeout")])
def test_choose_mode_model_error_falls_back_once(failure):
    calls = []

    def generate(messages, schema):
        calls.append(messages)
        raise failure

    mode, record = application.choose_mode(SimpleNamespace(generate=generate), "Question?")
    assert mode == "detail" and len(calls) == 1
    assert record == {"mode": "detail", "reply": "", "error": str(failure), "stats": None}


@pytest.mark.parametrize("mode", ["global", "auto"])
@pytest.mark.parametrize("selected", ["detail", "global"])
def test_modes_dispatch_global_or_detail(monkeypatch, mode, selected):
    result = object()
    calls = []
    router = {"mode": selected, "reply": "1" if selected == "detail" else "2"}
    workflow = SimpleNamespace(_model=object(), _retriever=None)

    def choose(client, question):
        assert client is workflow._model and question == "What happened?"
        calls.append("router")
        return selected, router

    def answer_global(passed, question, *, language):
        assert passed is workflow and question == "What happened?" and language == "en"
        calls.append("global")
        return global_answer.GlobalAnswer(result, {"version": "global-test", "calls": 7})

    def fits(question, *, language):
        calls.append("detail admission")
        return False

    workflow.whole_document_fits = fits
    monkeypatch.setattr(application, "choose_mode", choose)
    monkeypatch.setattr(global_answer, "answer_global", answer_global)
    monkeypatch.setattr(application, "ask_document", lambda *a, **kw: result)
    answer = application.answer_question(workflow, "What happened?", mode=mode, language="auto")
    actual = selected if mode == "auto" else "global"
    assert answer.result is result
    assert answer.trace["mode"] == actual and answer.trace["requested_mode"] == mode
    assert answer.trace["version"] == "routed-answer/v5"
    if mode == "auto":
        assert answer.trace["router"] == router
    else:
        assert "router" not in answer.trace
    assert calls == (["router"] if mode == "auto" else []) + (
        ["global"] if actual == "global" else ["detail admission"]
    )
    if actual == "global":
        assert answer.trace["global_trace"] == {"version": "global-test", "calls": 7}


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode must be auto, detail, global or deep"):
        application.answer_question(object(), "question", mode="fixed")
