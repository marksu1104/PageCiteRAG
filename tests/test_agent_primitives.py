from __future__ import annotations

import pytest
from tests.reading_fixtures import _setup

from page_cite_rag.agent import _search_preview
from page_cite_rag.application import ask_with_agent
from page_cite_rag.tokenize import LexicalTokenizer


def test_shared_agent_entry_rejects_mismatched_search_before_model_use(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    for changed in ({"query": "different question"}, {"document_id": "f" * 64}):
        with pytest.raises(ValueError, match="bound document and question"):
            ask_with_agent(
                workflow,
                "What does the method say?",
                initial_search=result.model_copy(update=changed),
                initial_blocks=blocks,
            )


def test_preview_uses_relevant_original_sentence_and_marks_cross_language_fallback():
    tokenizer = LexicalTokenizer()
    source = "Background. " + ("Unrelated text. " * 30) + "Accuracy is 92 percent."
    preview, kind = _search_preview(source, "accuracy 92", tokenizer)
    assert preview == "Accuracy is 92 percent."
    assert preview in source and kind == "lexical_match"
    preview, kind = _search_preview("A document in English.", "中文問題", tokenizer)
    assert preview == "A document in English." and kind == "prefix_fallback"
