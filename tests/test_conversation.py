from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from page_cite_rag.conversation import (
    CONDENSE_QUESTION_PROMPT,
    Turn,
    condense_question,
    format_history,
)
from page_cite_rag.generation import ChatMessage, GenerationStats, ModelError


def test_format_history_bounds_and_truncation():
    turns = [Turn(str(n), "a" * 1501 if n == 6 else f"answer {n}") for n in range(8)]
    expected = "\n".join(
        f"Human: {n}\nAssistant: " + ("a" * 1500 + "…" if n == 6 else f"answer {n}")
        for n in range(2, 8)
    )
    assert format_history(turns) == expected
    assert format_history([]) == ""
    assert format_history([Turn("Q", "x" * 1500)]) == "Human: Q\nAssistant: " + "x" * 1500


def test_no_history_does_not_call_model():
    client = Mock()
    assert condense_question(client, [], "Why?") == (
        "Why?",
        {"rewritten": False, "standalone": "Why?", "error": None, "stats": None},
    )
    client.generate.assert_not_called()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('\n\n  "Standalone?"  \nignored', "Standalone?"),
        ("'Standalone?'", "Standalone?"),
        ("", "Why?"),
        (" \n\t", "Why?"),
        ('""', "Why?"),
        ("x" * 1001, "Why?"),
        ("x" * 1000, "x" * 1000),
        ("Why?", "Why?"),
    ],
)
def test_condense_exact_prompt_and_output(value, expected):
    stats = GenerationStats(
        prompt_eval_count=10,
        eval_count=5,
        total_duration=1,
        load_duration=None,
        prompt_eval_duration=0,
        eval_duration=1,
        done_reason="stop",
        elapsed_ms=1.0,
    )
    client = Mock()
    client.generate.return_value = SimpleNamespace(value=value, stats=stats)
    prompt = (
        "Given the following conversation and a follow up question, rephrase the follow up "
        "question to be a standalone question, in its original language.\n\n"
        "Chat History:\nHuman: Q\nAssistant: A\nFollow Up Input: Why?\nStandalone question:"
    )
    standalone, record = condense_question(client, [Turn("Q", "A")], "Why?")
    assert (
        CONDENSE_QUESTION_PROMPT.format(chat_history="Human: Q\nAssistant: A", question="Why?")
        == prompt
    )
    client.generate.assert_called_once_with([ChatMessage(role="user", content=prompt)], None)
    assert standalone == expected
    assert record == {
        "rewritten": expected != "Why?",
        "standalone": expected,
        "error": None,
        "stats": stats.model_dump(mode="json"),
    }


@pytest.mark.parametrize("failure", [ModelError("failed"), OSError("failed"), ValueError("failed")])
def test_condense_failure_falls_back(failure):
    client = Mock()
    client.generate.side_effect = failure
    assert condense_question(client, [Turn("Q", "A")], "Why?") == (
        "Why?",
        {"rewritten": False, "standalone": "Why?", "error": type(failure).__name__, "stats": None},
    )
    assert client.generate.call_count == 1
