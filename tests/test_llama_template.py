"""Pinned text-template contract, separate from model answer quality."""

import json

import pytest

from page_cite_rag.generation import ChatMessage, ModelError
from page_cite_rag.llama import render_prompt


def test_schema_and_instructions_share_one_first_system_turn():
    schema = {"type": "object", "description": "保留否定"}
    messages = (
        ChatMessage(role="system", content="Preserve negations and conditions."),
        ChatMessage(role="user", content="Does it apply?"),
    )
    expected = (
        "<|im_start|>system\nReturn only JSON matching this schema:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n\nPreserve negations and conditions.<|im_end|>\n"
        "<|im_start|>user\nDoes it apply?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert render_prompt(messages, schema) == expected
    assert render_prompt(messages, schema).count("<|im_start|>system\n") == 1


def test_user_only_template_is_unchanged():
    messages = (ChatMessage(role="user", content="Question?"),)
    assert render_prompt(messages, {}) == (
        "<|im_start|>system\nReturn only JSON matching this schema:\n{}<|im_end|>\n"
        "<|im_start|>user\nQuestion?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@pytest.mark.parametrize("first_role", ["user", "system"])
def test_late_system_messages_are_rejected_not_silently_reordered(first_role):
    messages = (
        ChatMessage(role=first_role, content="First."),
        ChatMessage(role="user", content="Question?"),
        ChatMessage(role="system", content="Late instructions."),
    )
    with pytest.raises(ModelError, match="first"):
        render_prompt(messages, {})
