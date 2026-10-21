"""Small capability/lifecycle checks; synthetic counts are not semantic evidence."""

import json
from types import SimpleNamespace

import pytest

from page_cite_rag import cli
from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.generation import (
    ContextBudgetError,
    GenerationConfig,
    GenerationResult,
    ModelError,
    PromptBudget,
    budget_for_model,
    check_budget,
)
from page_cite_rag.llama import EXACT_BUDGET_VERSION, render_prompt
from test_llama import MESSAGES, Output, make_client
from test_workflows import FakeModel, stats
from test_workflows import setup as setup


def test_exact_measurement_counts_full_template_and_reuses_tokens(tmp_path, monkeypatch):
    client, calls, lifecycle = make_client(tmp_path, monkeypatch, budget_policy="exact")
    with client:
        from page_cite_rag.generation import ChatMessage

        messages = (ChatMessage(role="user", content="A long synthetic question. " * 1000),)
        schema = Output.model_json_schema()
        with pytest.raises(ContextBudgetError):
            check_budget(messages, schema, client.config)
        budget = budget_for_model(client, messages, schema, client.config)
        assert budget.method == EXACT_BUDGET_VERSION and budget.exact_input_tokens == 30
        assert budget.input_upper_estimate == 30 + client.config.template_reserve
        assert not any(path == "/completion" for path, _ in calls)
        token_payload = next(p for path, p in calls if path == "/tokenize")
        assert token_payload == {
            "content": render_prompt(messages, schema),
            "add_special": True,
            "parse_special": True,
        }
        result = client.generate(messages, Output)
        assert result.budget == budget and result.stats.prompt_eval_count == 30
        assert sum(path == "/tokenize" for path, _ in calls) == 1
        assert sum(path == "/completion" for path, _ in calls) == 1
        changed = {**schema, "description": "A different output contract."}
        client.prepare_budget(messages, changed)
        assert sum(path == "/tokenize" for path, _ in calls) == 2
    assert lifecycle["starts"] == lifecycle["stops"] == 1 and client._token_cache is None


def test_exact_overflow_is_recoverable_but_never_completes(tmp_path, monkeypatch):
    tokens = [7] * 16384
    client, calls, lifecycle = make_client(
        tmp_path, monkeypatch, tokens=tokens, budget_policy="exact"
    )
    with client:
        with pytest.raises(ContextBudgetError):
            client.prepare_budget(MESSAGES, Output.model_json_schema())
        assert not client._closed and not any(path == "/completion" for path, _ in calls)
        tokens[:] = [7] * 30
        assert client.prepare_budget(MESSAGES, Output.model_json_schema()).exact_input_tokens == 30
    assert lifecycle["stops"] == 1


def test_exact_cache_cannot_hide_changed_runtime(tmp_path, monkeypatch):
    client, calls, lifecycle = make_client(tmp_path, monkeypatch, budget_policy="exact")
    client.prepare_budget(MESSAGES, Output.model_json_schema())
    (client._bundle / "fake-model").write_bytes(b"Changed synthetic weights.")
    with pytest.raises(ModelError, match="changed"):
        client.prepare_budget(MESSAGES, Output.model_json_schema())
    assert lifecycle["stops"] == 1 and client._token_cache is None
    assert not any(path == "/completion" for path, _ in calls)


@pytest.mark.parametrize("tokens", [[True], [-1]])
def test_invalid_exact_tokenizer_response_never_falls_back(tmp_path, monkeypatch, tokens):
    client, calls, lifecycle = make_client(
        tmp_path, monkeypatch, tokens=tokens, budget_policy="exact"
    )
    with pytest.raises(ModelError, match="tokenizer"):
        client.prepare_budget(MESSAGES, Output.model_json_schema())
    assert lifecycle["stops"] == 1 and not any(path == "/completion" for path, _ in calls)


def test_changed_resident_context_closes_before_completion(tmp_path, monkeypatch):
    client, calls, lifecycle = make_client(tmp_path, monkeypatch, budget_policy="exact")
    request = client._request
    monkeypatch.setattr(
        client,
        "_request",
        lambda path, *a, **k: (
            {"default_generation_settings": {"n_ctx": 4096}}
            if path == "/props"
            else request(path, *a, **k)
        ),
    )
    with pytest.raises(ContextBudgetError, match="resident"):
        client.prepare_budget(MESSAGES, Output.model_json_schema())
    assert lifecycle["stops"] == 1 and not any(path == "/completion" for path, _ in calls)


def test_budget_capability_rejects_incompatible_config():
    config = GenerationConfig()
    invalid = PromptBudget(1024, config.num_predict, config.num_ctx * 2)
    client = SimpleNamespace(prepare_budget=lambda *a: invalid)
    with pytest.raises(ModelError, match="incompatible"):
        budget_for_model(client, MESSAGES, Output.model_json_schema(), config)
    assert budget_for_model(None, MESSAGES, Output.model_json_schema(), config) == check_budget(
        MESSAGES, Output.model_json_schema(), config
    )


def test_workflow_uses_same_budget_for_question_admission_and_generation(setup):
    store, document, blocks, retriever, _, _ = setup
    config = GenerationConfig(num_ctx=2048, num_predict=512)

    class Exact(FakeModel):
        def prepare_budget(self, messages, schema):
            return PromptBudget(100 + config.template_reserve, 512, 2048, EXACT_BUDGET_VERSION, 100)

        def generate(self, messages, response_model):
            self.calls.append((messages, response_model))
            evidence = json.loads(messages[1].content)["evidence"]
            draft = response_model.model_validate_json(
                json.dumps(
                    {
                        "status": "answered",
                        "answer_claims": [
                            {
                                "text": "Only if n > 64; children excluded.",
                                "citations": [{"source_id": evidence[0]["source_id"]}],
                            }
                        ],
                        "detail_claims": [],
                    }
                )
            )
            return GenerationResult(draft, stats(), self.prepare_budget(messages, {}))

    question = "x" * 3900
    conservative = FullAnswerWorkflow(store, document, retriever, FakeModel(config), config)
    with pytest.raises(ContextBudgetError):
        conservative.ask(question)
    model = Exact(config)
    workflow = FullAnswerWorkflow(store, document, retriever, model, config)
    result = workflow.ask(question)
    assert len(model.calls) == 1 and result.budget.exact_input_tokens == 100
    assert result.included_block_ids == tuple(block.block_id for block in blocks)
    assert blocks[0].text.startswith(result.claims[0].citations[0].quote)


def test_exact_cli_is_explicit_and_unsupported_backend_does_not_start(monkeypatch):
    parser = cli.build_parser()
    assert parser.parse_args(["ask", "a" * 64, "Question?"]).token_budget == "conservative"
    monkeypatch.setattr(cli, "_open_generation_client", lambda *a: pytest.fail("model started"))
    assert cli.main(["ask", "a" * 64, "Question?", "--token-budget", "exact"]) == 1
