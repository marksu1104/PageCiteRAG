from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from page_cite_rag import cli
from page_cite_rag.generation import (
    ContextBudgetError,
    GenerationResult,
    GenerationStats,
    ModelError,
    ModelTimeoutError,
    check_budget,
)
from page_cite_rag.ingest import prepare_source
from page_cite_rag.retrieval import build_bm25_index
from page_cite_rag.store import SQLiteDocumentStore


def test_default_answer_retrieval_profile_uses_full_admission_pool():
    ask = cli.build_parser().parse_args(["ask", "0" * 64, "question"])
    serve = cli.build_parser().parse_args(["serve", "0" * 64])

    assert ask.top_k == 100
    assert serve.top_k == 100


def _fake_navigation_agent_run(workflow, context, *, model_calls, tool_calls, ledger):
    from page_cite_rag.agent import AgentRun, ReadingReport

    final_input = max(context.exact_input_tokens or 100, 1)
    for _ in range(model_calls):
        ledger.admit(final_input, workflow._budget_config.num_ctx)
        ledger.complete(24)
    trace = {
        "model_calls_completed": model_calls,
        "tool_calls_recorded": tool_calls,
        "agent_input_tokens_exact": final_input * model_calls,
        "agent_output_tokens_observed": 24 * model_calls,
        "agent_output_tokens_unobserved_reserved": 0,
        "model_call_records": [
            {
                "status": "completed",
                "input_tokens_exact": final_input,
                "output_tokens_observed": 24,
                "output_token_limit": ledger.call_output_limit,
                "elapsed_ms": 10.0,
                "role": "navigation",
            }
            for _ in range(model_calls)
        ],
        "bootstrap_shown_block_ids": [block.block_id for block in context.blocks],
        "initial_answer_block_ids": [block.block_id for block in context.blocks],
        "tool_trace": [],
        "architecture": "single_agent_navigation_with_shared_writer",
    }
    return AgentRun(
        context,
        trace,
        structured_response=ReadingReport(
            requirements=[],
            unresolved=["the requested fact"],
        ),
        acquired_context=context,
    )


def _enable_exact_writer_budget(workflow):
    from dataclasses import replace

    from page_cite_rag.generation import PromptBudget

    model = workflow._model
    original_generate = model.generate

    def prepare_budget(_messages, _schema):
        return PromptBudget(
            input_upper_estimate=100 + model.config.template_reserve,
            output_reserved=model.config.num_predict,
            context_limit=model.config.num_ctx,
            method="fake-exact-token-test/v1",
            exact_input_tokens=100,
        )

    def generate(messages, schema):
        generated = original_generate(messages, schema)
        return replace(generated, budget=prepare_budget(messages, schema))

    model.prepare_budget = prepare_budget
    model.generate = generate


class FakeClient:
    def __init__(self, config):
        self.config = config
        self.identity = {"backend": "fake", "device": getattr(config, "device", "cpu")}
        self.calls = []
        self.close_count = 0
        self.failure = None
        self.invalid_citation = False
        self.insufficient = False

    def generate(self, messages, schema_class):
        self.calls.append((messages, schema_class))
        if self.failure:
            raise self.failure
        if schema_class is None:
            return GenerationResult(
                "document, evidence",
                GenerationStats(
                    prompt_eval_count=50,
                    eval_count=10,
                    total_duration=100,
                    load_duration=None,
                    prompt_eval_duration=40,
                    eval_duration=60,
                    done_reason="stop",
                    elapsed_ms=1.0,
                ),
                check_budget(messages, None, self.config),
            )
        payload = json.loads(messages[1].content)
        if self.insufficient:
            value = schema_class.model_validate_json(
                json.dumps({"status": "insufficient_evidence", "limitations": ["Not stated."]})
            )
        else:
            first_evidence = payload["evidence"][0]
            citation = {
                "source_id": "S999" if self.invalid_citation else first_evidence["source_id"]
            }
            value = schema_class.model_validate_json(
                json.dumps(
                    {
                        "status": "answered",
                        "answer_claims": [
                            {
                                "claim_id": "C1",
                                "support": "direct",
                                "text": (
                                    "结果并不显著，除非 n > 64；儿童被排除。"
                                    if "Traditional Chinese" in messages[0].content
                                    else "The result was not significant unless n > 64."
                                ),
                                "citations": [citation],
                            }
                        ],
                    }
                )
            )
        return GenerationResult(
            value,
            GenerationStats(
                prompt_eval_count=100,
                eval_count=20,
                total_duration=100,
                load_duration=None,
                prompt_eval_duration=40,
                eval_duration=60,
                done_reason="stop",
                elapsed_ms=1.0,
            ),
            check_budget(messages, schema_class.model_json_schema(), self.config),
        )

    def count_chat_input_tokens(self, _payload):
        return 100

    def close(self):
        self.close_count += 1


@pytest.fixture
def document(tmp_path):
    source = tmp_path / "self-authored.txt"
    source.write_text(
        "The result was not significant unless n > 64. Children were excluded.\n\n"
        "The trial included 120 adults.\n\n"
        "The confidence interval was 95%.",
        encoding="utf-8",
    )
    prepared = prepare_source(source, usage_scope="self-authored-cli-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    document_id = prepared.document.document_id
    return store, document_id, store.get_blocks(document_id)


@pytest.fixture
def clients(monkeypatch):
    opened = []

    def open_client(args, config):
        client = FakeClient(config)
        opened.append((args, config, client))
        return client

    monkeypatch.setattr(cli, "_open_generation_client", open_client)
    return opened


@pytest.fixture
def tokenizer_directory(tmp_path):
    directory = tmp_path / "test-tokenizer"
    directory.mkdir()
    return directory


def ask_arguments(document, *options):
    """The fixed path on your own (faked) Ollama; later options override these."""
    store, document_id, _ = document
    return [
        "ask",
        document_id,
        "significant",
        "--language",
        "en",
        "--db",
        str(store.database_path),
        "--route",
        "fixed",
        "--backend",
        "ollama",
        *options,
    ]


def test_ask_json_end_to_end_has_program_derived_exact_citations(document, clients, capsys):
    store, document_id, blocks = document
    build_bm25_index(store, document_id)
    before = store.database_path.read_bytes()
    assert cli.main(ask_arguments(document, "--format", "json")) == 0
    output = capsys.readouterr()
    assert output.err == ""
    result = json.loads(output.out)
    assert result["status"] == "answered"
    assert result["mode"] == "local"
    assert result["retrieval"]["method"] == "bm25"
    assert result["included_block_ids"] == [blocks[0].block_id]
    assert result["answer"] == "\n\n".join(claim["text"] for claim in result["answer_claims"])
    citation = result["answer_claims"][0]["citations"][0]
    unit = store.get_unit(document_id, citation["unit_index"])
    assert unit.text[citation["char_start"] : citation["char_end"]] == citation["quote"]
    assert citation["document_id"] == document_id
    assert citation["page_number"] is None
    assert citation["line_start"] == citation["line_end"] == 1
    assert result["generation_stats"]["load_duration"] is None
    assert len(clients) == 1 and len(clients[0][2].calls) == 1
    assert clients[0][2].close_count == 1
    assert before == store.database_path.read_bytes()


def test_cited_source_can_be_read_back_without_loading_model(document, clients, capsys):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    assert cli.main(ask_arguments(document, "--format", "json")) == 0
    answer = json.loads(capsys.readouterr().out)
    citation = answer["answer_claims"][0]["citations"][0]
    before = store.database_path.read_bytes()

    assert (
        cli.main(
            [
                "source",
                document_id,
                citation["block_id"],
                "--db",
                str(store.database_path),
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    source = json.loads(output.out)
    assert output.err == ""
    assert (
        source["text"][
            citation["char_start"] - source["char_start"] : citation["char_end"]
            - source["char_start"]
        ]
        == citation["quote"]
    )
    assert source["block_id"] == citation["block_id"]
    assert len(clients) == 1 and len(clients[0][2].calls) == 1
    assert before == store.database_path.read_bytes()


def test_source_refuses_wrong_scope_and_changed_original(document, clients, capsys):
    store, document_id, blocks = document
    args = ["source", "f" * 64, blocks[0].block_id, "--db", str(store.database_path)]
    assert cli.main(args) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "unavailable or invalid local source" in output.err

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE document_units SET text = ? WHERE document_id = ? AND unit_index = 1",
            ("Changed original text.", document_id),
        )
    args[1] = document_id
    assert cli.main(args) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Changed original text" not in output.err
    assert clients == []


def test_ask_text_output_displays_answer_quote_and_positions(document, clients, capsys):
    store, document_id, blocks = document
    build_bm25_index(store, document_id)
    assert cli.main(ask_arguments(document)) == 0
    output = capsys.readouterr()
    assert "- The result was not significant unless n > 64." in output.out
    assert "lines 1-1" in output.out
    assert f"block {blocks[0].block_id}" in output.out
    assert "chars [0," in output.out
    assert json.dumps(blocks[0].text, ensure_ascii=False) in output.out
    assert "Status: answered" in output.out
    assert "not whether a claim" in output.out
    assert output.err == ""
    assert clients[0][2].close_count == 1


def test_chinese_output_option_preserves_english_quotes(document, clients, capsys):
    store, document_id, blocks = document
    build_bm25_index(store, document_id)
    assert cli.main(ask_arguments(document, "--language", "zh-Hant", "--format", "json")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["language"] == "zh-Hant"
    assert "結果並不顯著" in result["answer"]
    assert "兒童被排除" in result["answer"]
    assert result["answer_claims"][0]["citations"][0]["quote"] == blocks[0].text
    assert clients[0][2].close_count == 1


@pytest.mark.parametrize("neighbors", [0, 1, 2])
def test_explain_no_index_required_and_neighbors_bounded(document, clients, capsys, neighbors):
    store, document_id, blocks = document
    arguments = [
        "explain",
        document_id,
        blocks[1].block_id,
        "--db",
        str(store.database_path),
        "--neighbors",
        str(neighbors),
        "--format",
        "json",
    ]
    assert cli.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "explain"
    assert result["retrieval"] is None
    assert result["included_block_ids"][0] == blocks[1].block_id
    assert len(result["included_block_ids"]) == (1 if neighbors == 0 else 3)
    assert result["answer_claims"][0]["citations"][0]["block_id"] == blocks[1].block_id
    assert store.get_retrieval_manifest(document_id, "bm25") is None
    assert clients[0][2].close_count == 1


def test_unknown_question_returns_zero_visible_insufficient_without_generation(
    document, clients, capsys
):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    arguments = ask_arguments(document, "--format", "json")
    arguments[2] = "zyxwvutsrqponmlk"
    assert cli.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "insufficient_evidence"
    assert result["stop_reason"] == "no_evidence"
    assert result["generation_stats"] is None
    assert clients[0][2].calls == []
    assert clients[0][2].close_count == 1


@pytest.mark.parametrize("failure", ["citation", "timeout", "budget", "model", "os"])
def test_answer_failures_return_nonzero_and_close_client_without_raw_data(
    document, monkeypatch, capsys, failure
):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    opened = []

    def open_client(args, config):
        client = FakeClient(config)
        if failure == "citation":
            client.invalid_citation = True
        else:
            client.failure = {
                "timeout": ModelTimeoutError("local generation timed out; not retried"),
                "budget": ContextBudgetError("input exceeds the reserved context budget"),
                "model": ModelError("local generation failed"),
                "os": OSError("PRIVATE_CANARY /private/path"),
            }[failure]
        opened.append(client)
        return client

    monkeypatch.setattr(cli, "_open_generation_client", open_client)
    assert cli.main(ask_arguments(document, "--format", "json")) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "ask failed" in output.err
    assert "PRIVATE_CANARY" not in output.err and "/private/path" not in output.err
    assert opened[0].close_count == 1
    assert len(opened[0].calls) == 1


def test_model_insufficiency_is_visible_success_and_closed(document, monkeypatch, capsys):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    opened = []

    def open_client(args, config):
        client = FakeClient(config)
        client.insufficient = True
        opened.append(client)
        return client

    monkeypatch.setattr(cli, "_open_generation_client", open_client)
    assert cli.main(ask_arguments(document, "--format", "json")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "insufficient_evidence"
    assert result["stop_reason"] == "model_insufficient_evidence"
    assert result["answer_claims"] == []
    assert result["detail_claims"] == []
    assert opened[0].close_count == 1


@pytest.mark.parametrize("missing", ["database", "document", "index", "block", "invalid_block"])
def test_missing_or_invalid_resources_never_start_model(
    document, clients, capsys, missing, tmp_path
):
    store, document_id, blocks = document
    if missing in ("block", "invalid_block"):
        arguments = [
            "explain",
            document_id,
            "f" * 64 if missing == "block" else "PRIVATE_CANARY",
            "--db",
            str(store.database_path),
        ]
    else:
        arguments = ask_arguments(document)
        if missing == "database":
            arguments[arguments.index("--db") + 1] = str(tmp_path / "PRIVATE_CANARY.sqlite3")
        elif missing == "document":
            arguments[1] = "f" * 64
    assert cli.main(arguments) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "PRIVATE_CANARY" not in output.err
    assert clients == []
    if missing == "database":
        assert not (tmp_path / "PRIVATE_CANARY.sqlite3").exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--device", "cuda"],
        ["--gpu-uuid", "GPU-12345678-1234-1234-1234-123456789abc"],
        ["--bundle-dir", "/PRIVATE_CANARY/model"],
        ["--llm-host", "http://PRIVATE_CANARY.invalid:1234"],
        ["--backend", "llama", "--device", "cuda"],
        ["--backend", "llama", "--gpu-uuid", "PRIVATE_CANARY"],
        ["--backend", "llama", "--llm-host", "http://127.0.0.1:9999"],
        ["--backend", "llama", "--output-tokens", "2049"],
        ["--backend", "llama", "--context-window", "16385"],
        ["--generation-timeout", "nan"],
        ["--context-window", "2048", "--output-tokens", "1024"],
    ],
)
def test_invalid_backend_options_are_safe_and_do_not_start_model(
    document, clients, capsys, options
):
    assert cli.main(ask_arguments(document, *options)) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "PRIVATE_CANARY" not in output.err
    assert clients == []


def test_explicit_gpu_route_passes_same_validated_uuid_and_budget(
    document, clients, capsys, tmp_path
):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    uuid = "GPU-12345678-1234-1234-1234-123456789abc"
    bundle = tmp_path / "prepared-bundle-not-started-by-fake"
    assert (
        cli.main(
            ask_arguments(
                document,
                "--backend",
                "llama",
                "--device",
                "cuda",
                "--gpu-uuid",
                uuid,
                "--bundle-dir",
                str(bundle),
                "--output-tokens",
                "800",
                "--format",
                "json",
            )
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    args, config, client = clients[0]
    assert args.backend == "llama"
    assert config.device == "cuda" and config.gpu_uuid == uuid
    assert config.bundle_dir == bundle
    assert config.num_ctx == 16384 and config.num_predict == 800
    assert result["budget"]["output_reserved"] == 800
    assert result["model_identity"]["device"] == "cuda"
    assert client.close_count == 1


def test_invalid_dense_rewrite_route_rejected_before_model_or_encoder(
    document, clients, monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "_load_encoder", lambda args: pytest.fail("invalid route must not load embeddings")
    )
    assert cli.main(ask_arguments(document, "--method", "dense", "--rewrite-query")) == 1
    assert clients == []
    assert "PRIVATE_CANARY" not in capsys.readouterr().err


@pytest.mark.parametrize("question", ["", "   ", "a" * 4001])
def test_invalid_question_rejected_before_model_start(document, clients, capsys, question):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    arguments = ask_arguments(document)
    arguments[2] = question
    assert cli.main(arguments) == 1
    assert clients == []
    assert "ask failed" in capsys.readouterr().err


def test_top_k_over_workflow_limit_rejected_before_model_start(document, clients, capsys):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    assert cli.main(ask_arguments(document, "--top-k", "101")) == 1
    assert clients == []
    assert "ask failed" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["ask", "explain"])
def test_answer_help_does_not_import_models_or_retrieval_libraries(command, tmp_path):
    source_root = Path(cli.__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'torch', 'transformers', 'sentence_transformers', 'httpx', 'ollama',
                   'bm25s', 'numpy', 'jieba', 'page_cite_rag.generation', 'page_cite_rag.llama',
                   'page_cite_rag.workflows', 'page_cite_rag.retrieval', 'page_cite_rag.embedding'}
        if fullname in blocked or fullname.split('.', 1)[0] in blocked:
            raise AssertionError('model dependency imported during help')
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, sys.argv[1])
from page_cite_rag.cli import main
main([sys.argv[2], '--help'])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_root), command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--backend" in result.stdout and "--language" in result.stdout
    if command == "ask":
        assert "--route" in result.stdout and "--agent" not in result.stdout
        assert "--native-gpu" in result.stdout


def test_public_cli_defaults_to_traditional_chinese():
    args = cli.build_parser().parse_args(["ask", "0" * 64, "為什麼結果不顯著？"])
    assert args.language == "zh-Hant"
    assert args.generation_timeout == 600
    assert args.request_timeout_seconds == 600


def _fake_routed_answer(events=None):
    """Stand-in for application.answer_question: the fixed writer plus its route trace."""
    from page_cite_rag import application

    def answer(workflow, question, *, language, source_language, route):
        if events is not None:
            events.append("fixed answer")
        result = application.ask_document(workflow, question, language=language)
        trace = {
            "version": "routed-answer/v3",
            "requested_route": route,
            "route": "fixed",
            "reason": "auto" if route == "auto" else "requested",
        }
        return application.AgentAnswer(result, trace)

    return answer


def test_auto_route_cli_answers_through_the_shared_entry_and_reports_it(
    document, clients, monkeypatch, capsys
):
    from page_cite_rag import application
    from page_cite_rag.dense import build_dense_index
    from tests.test_dense import FakeEncoder

    store, document_id, _ = document
    build_bm25_index(store, document_id)
    encoder = FakeEncoder()
    build_dense_index(store, document_id, encoder)
    monkeypatch.setattr(cli, "_load_encoder", lambda _args: encoder)
    seen = {}
    fake = _fake_routed_answer()

    def answer(workflow, question, **kwargs):
        seen.update(kwargs)
        return fake(workflow, question, **kwargs)

    monkeypatch.setattr(application, "answer_question", answer)
    assert cli.main(ask_arguments(document, "--route", "auto", "--format", "json")) == 0
    output = json.loads(capsys.readouterr().out)
    assert seen["route"] == "auto" and seen["language"] == "en"
    assert output["answer"]["retrieval"]["method"] == "bm25"
    assert output["routing"]["route"] == "fixed"
    assert output["routing"]["reason"] == "auto"

    assert cli.main(ask_arguments(document, "--route", "auto")) == 0
    text = capsys.readouterr().out
    assert "Route: fixed retrieval (auto)." in text
    assert all(client.close_count == 1 for _args, _config, client in clients)


def test_auto_accepts_bm25_and_uses_whole_document(document, clients, capsys):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    assert cli.main(ask_arguments(document, "--route", "auto", "--method", "bm25")) == 0
    assert "Route: whole document (fits the context)." in capsys.readouterr().out
    assert len(clients) == 1


def test_default_backend_is_the_validated_pinned_ollama_with_exact_budget():
    args = cli.build_parser().parse_args(["ask", "0" * 64, "question"])
    assert args.backend == "pinned-ollama" and args.route == "auto"
    assert args.output_tokens == 3072 and args.context_window == 16384
    config = cli._generation_config(args)
    assert args.token_budget == "exact"
    assert config.num_predict == 3072 and config.num_ctx == 16384
    own = cli.build_parser().parse_args(
        ["ask", "0" * 64, "question", "--backend", "ollama", "--llm-model", "mine:latest"]
    )
    assert cli._generation_config(own).model == "mine:latest" and own.token_budget == "conservative"


def test_native_gpu_ask_searches_before_opening_the_generation_model(
    document, clients, tokenizer_directory, monkeypatch, capsys
):
    from page_cite_rag.retrieval import BM25Retriever

    store, document_id, _ = document
    build_bm25_index(store, document_id)
    events = []
    expected_tokenizer_directory = tokenizer_directory

    class FakeStagedGpuSearch:
        def __init__(
            self,
            passed_store,
            passed_document_id,
            method,
            model_path,
            gpu_uuid,
            *,
            tokenizer_directory,
            gpu_temperature_limit_c,
            candidates=None,
            reranker_model_path=None,
        ):
            assert passed_store.database_path == store.database_path
            assert passed_document_id == document_id
            assert method == "hybrid"
            assert model_path.name == "Qwen3-Embedding-0.6B-f16.gguf"
            assert gpu_uuid == "GPU-12345678-1234-1234-1234-123456789abc"
            assert gpu_temperature_limit_c == 88
            assert candidates == 50
            assert tokenizer_directory == expected_tokenizer_directory
            assert tokenizer_directory.is_dir()
            assert clients == []
            events.append("embedding setup")
            self.retriever = BM25Retriever(store, document_id)

        def search(self, query, *, top_k):
            events.append("GPU query embedding and search")
            return self.retriever.search(query, top_k=top_k)

    from page_cite_rag import gpu_retrieval

    monkeypatch.setattr(gpu_retrieval, "StagedGpuSearch", FakeStagedGpuSearch)
    open_client = cli._open_generation_client

    def record_generation_start(args, config):
        events.append("generation model startup")
        return open_client(args, config)

    monkeypatch.setattr(cli, "_open_generation_client", record_generation_start)
    monkeypatch.setattr(
        cli, "_load_encoder", lambda _args: pytest.fail("native GPU route loaded the CPU encoder")
    )

    arguments = ask_arguments(
        document,
        "--method",
        "hybrid",
        "--native-gpu",
        "--backend",
        "llama",
        "--token-budget",
        "exact",
        "--device",
        "cuda",
        "--gpu-uuid",
        "GPU-12345678-1234-1234-1234-123456789abc",
        "--model-dir",
        str(tokenizer_directory),
        "--format",
        "json",
    )

    assert cli.main(arguments) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["status"] == "answered"
    assert events == [
        "embedding setup",
        "GPU query embedding and search",
        "generation model startup",
    ]
    assert len(clients) == 1 and clients[0][2].close_count == 1


@pytest.mark.parametrize("reranker_available", [True, False])
def test_native_gpu_auto_does_not_open_embedding_session(
    document, clients, tokenizer_directory, monkeypatch, capsys, tmp_path, reranker_available
):
    from page_cite_rag import application, gpu_retrieval, reranker_server
    from page_cite_rag.retrieval import BM25Retriever

    ranker_path = tmp_path / "fake-reranker.gguf"
    if reranker_available:
        ranker_path.touch()
    monkeypatch.setattr(reranker_server, "DEFAULT_MODEL_PATH", ranker_path)
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    monkeypatch.setattr(
        cli,
        "_load_encoder",
        lambda _args: pytest.fail("native GPU must not silently fall back to CPU"),
    )
    events = []
    expected_tokenizer_directory = tokenizer_directory

    class FakeStagedGpuSearch:
        def __init__(
            self,
            passed_store,
            passed_document_id,
            method,
            _model_path,
            _gpu_uuid,
            *,
            tokenizer_directory,
            gpu_temperature_limit_c,
            candidates=None,
            reranker_model_path=None,
        ):
            assert passed_store.database_path == store.database_path
            assert passed_document_id == document_id
            assert method == "hybrid"
            assert tokenizer_directory == expected_tokenizer_directory
            assert tokenizer_directory.is_dir()
            assert gpu_temperature_limit_c == 88
            assert candidates == (100 if reranker_available else 50)
            assert reranker_model_path == (ranker_path if reranker_available else None)
            assert clients == []
            events.append("GPU search setup")
            self.method = "hybrid"
            self.retriever = BM25Retriever(store, document_id)

        def search(self, query, *, top_k):
            events.append(f"GPU query: {query}")
            return self.retriever.search(query, top_k=top_k).model_copy(update={"method": "hybrid"})

        @contextmanager
        def session(self):
            events.append("GPU session start")
            try:
                yield self
            finally:
                events.append("GPU session cleanup")

    monkeypatch.setattr(gpu_retrieval, "StagedGpuSearch", FakeStagedGpuSearch)
    monkeypatch.setattr(application, "answer_question", _fake_routed_answer(events))
    open_client = cli._open_generation_client

    def record_generation_start(args, config):
        events.append("generation startup")
        return open_client(args, config)

    monkeypatch.setattr(cli, "_open_generation_client", record_generation_start)
    arguments = ask_arguments(
        document,
        "--method",
        "hybrid",
        "--native-gpu",
        "--backend",
        "llama",
        "--token-budget",
        "exact",
        "--device",
        "cuda",
        "--gpu-uuid",
        "GPU-12345678-1234-1234-1234-123456789abc",
        "--model-dir",
        str(tokenizer_directory),
        "--gpu-temperature-limit",
        "88",
        "--route",
        "auto",
        "--format",
        "json",
    )

    assert cli.main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["answer"]["status"] == "answered"
    assert output["routing"]["route"] == "fixed"
    assert output["routing"]["reason"] == "auto"
    assert events == [
        "GPU search setup",
        "generation startup",
        "fixed answer",
        "GPU query: significant",
    ]
    assert len(clients) == 1 and clients[0][2].close_count == 1


def test_cli_request_timeout_may_extend_to_an_hour_for_cpu_only_machines(document, clients):
    store, document_id, _ = document
    build_bm25_index(store, document_id)
    assert cli.main(ask_arguments(document, "--request-timeout-seconds", "3601")) == 1
    assert cli.main(ask_arguments(document, "--request-timeout-seconds", "3600")) == 0


@pytest.mark.parametrize("command", ["ask", "serve"])
@pytest.mark.parametrize("route", ["auto", "fixed", "agent"])
def test_route_choices_accept_only_active_routes(command, route):
    parser = cli.build_parser()
    arguments = [command, "0" * 64]
    if command == "ask":
        arguments.append("question")
    assert parser.parse_args([*arguments, "--route", route]).route == route
    for retired in ("whole-document", "unknown"):
        with pytest.raises(SystemExit) as error:
            parser.parse_args([*arguments, "--route", retired])
        assert error.value.code == 2


@pytest.mark.parametrize("fits", [True, False])
@pytest.mark.parametrize("writer", ["structured", "quote_first"])
def test_whole_document_fits_matches_writer_admission(document, fits, writer):
    from page_cite_rag.application import document_workflow
    from page_cite_rag.generation import GenerationConfig

    store, document_id, blocks = document
    config = GenerationConfig()
    client = FakeClient(config)
    workflow = document_workflow(store, document_id, None, client, config, writer=writer)
    from page_cite_rag import workflows

    budget = workflows.budget_for_model

    def bounded(model, messages, schema, config):
        content = messages[1].content
        if writer == "structured":
            count = len(json.loads(content).get("evidence", []))
        else:
            count = sum(block.text in content for block in blocks)
        if not fits and count > 1:
            raise ContextBudgetError("test budget")
        return budget(model, messages, schema, config)

    # Patch the budget boundary only; selection and writer messages remain real.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(workflows, "budget_for_model", bounded)
        assert workflow.whole_document_fits("question", language="en") is fits
        selected, omitted, _, _ = workflow._context("question", "en", "ask", blocks, None)
        assert (len(selected) == len(blocks) and not omitted) is fits
    assert client.calls == []


def test_whole_document_fits_skips_tokenizing_a_far_too_large_collection(document, monkeypatch):
    from page_cite_rag.application import document_workflow
    from page_cite_rag.generation import GenerationConfig

    store, document_id, blocks = document
    config = GenerationConfig()
    workflow = document_workflow(store, document_id, None, FakeClient(config), config)
    huge = [b.model_copy(update={"text": "x" * (32 * config.num_ctx)}) for b in blocks[:1]]
    monkeypatch.setattr(workflow._store, "get_blocks", lambda _doc: huge * 2)
    monkeypatch.setattr(workflow, "_context", lambda *a, **k: pytest.fail("must not tokenize"))
    assert workflow.whole_document_fits("question", language="en") is False


@pytest.mark.parametrize("route", ["auto", "agent"])
@pytest.mark.parametrize("option", ["--rewrite-query", "--rerank"])
def test_non_fixed_rejects_rewrite_and_rerank(document, clients, capsys, route, option):
    assert cli.main(ask_arguments(document, "--route", route, option)) == 1
    assert clients == []
    output = capsys.readouterr()
    assert output.out == ""
    assert "--rewrite-query and --rerank apply to --route fixed" in output.err


@pytest.mark.parametrize("route", ["fixed", "auto", "agent"])
@pytest.mark.parametrize("answer_format", ["general", "short"])
def test_cli_answer_format_selects_writer(
    document, clients, monkeypatch, capsys, route, answer_format, tokenizer_directory
):
    from page_cite_rag import application

    store, document_id, _ = document
    build_bm25_index(store, document_id)
    captured = []
    original = application.document_workflow

    def workflow(*args, **kwargs):
        captured.append((kwargs["writer"], kwargs["answer_format"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(application, "document_workflow", workflow)
    options = []
    if route == "agent":
        from page_cite_rag import gpu_retrieval

        monkeypatch.setattr(gpu_retrieval, "StagedGpuSearch", lambda *a, **kw: object())
        options = [
            "--native-gpu",
            "--method",
            "hybrid",
            "--backend",
            "pinned-ollama",
            "--device",
            "cuda",
            "--token-budget",
            "exact",
            "--model-dir",
            str(tokenizer_directory),
            "--gpu-uuid",
            "GPU-12345678-1234-1234-1234-123456789abc",
        ]

    # Stop at the concrete workflow so short format needs no structured fake response.
    def routed(*args, **kwargs):
        raise ModelError("checked writer")

    monkeypatch.setattr(application, "answer_question", routed)
    if route == "fixed":
        monkeypatch.setattr(application, "ask_document", routed)
    assert (
        cli.main(
            ask_arguments(document, "--route", route, "--answer-format", answer_format, *options)
        )
        == 1
    )
    assert captured == [
        (
            "quote_first" if route == "agent" or answer_format == "short" else "structured",
            "short_answer" if answer_format == "short" else "general",
        )
    ]
    capsys.readouterr()


def test_insufficient_passages_are_from_admitted_context_in_order(document, monkeypatch, capsys):
    store, document_id, blocks = document
    build_bm25_index(store, document_id)
    from page_cite_rag.generation import GenerationConfig

    client = FakeClient(GenerationConfig())
    client.insufficient = True
    monkeypatch.setattr(cli, "_open_generation_client", lambda *args: client)
    # A whole-document read is in document order, so it lists no "most relevant" passages.
    assert cli.main(ask_arguments(document, "--route", "auto", "--format", "json")) == 0
    assert "closest_passages" not in json.loads(capsys.readouterr().out)
    arguments = ask_arguments(document, "--route", "fixed", "--format", "json")
    assert cli.main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    passages = output["closest_passages"]
    admitted = output["included_block_ids"][:3]
    by_id = {b.block_id: b for b in blocks}
    assert passages and [p["block_id"] for p in passages] == admitted
    assert passages == [
        {
            "block_id": by_id[i].block_id,
            "page_number": by_id[i].page_number,
            "line_start": by_id[i].line_start,
            "line_end": by_id[i].line_end,
            "text": by_id[i].text[:300],
        }
        for i in admitted
    ]
    first = by_id[admitted[0]]
    assert cli.main(ask_arguments(document, "--route", "fixed")) == 0
    output = capsys.readouterr().out
    assert "Most relevant passages:" in output
    assert f"lines {first.line_start}-{first.line_end}" in output
    assert first.text[:300] in output


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--native-gpu", "--method", "dense"],
        ["--native-gpu", "--method", "hybrid", "--backend", "ollama"],
    ],
)
def test_cli_agent_rejects_missing_requirements_before_model(document, clients, capsys, options):
    assert cli.main(ask_arguments(document, "--route", "agent", *options)) == 1
    assert clients == []
    output = capsys.readouterr()
    assert output.out == ""
    assert "agent requires --native-gpu, --method hybrid and --backend pinned-ollama" in output.err


@pytest.mark.parametrize(
    "trace,expected",
    [
        ({"route": "whole_document_context"}, "Route: whole document (fits the context)."),
        ({"route": "reranked"}, "Route: reranked top 20 of 200."),
        (
            {"route": "fixed", "reason": "rerank_unavailable"},
            "Route: fixed retrieval (no reranker).",
        ),
        (
            {"route": "agent", "model_calls": 5, "search_steps": ["a", "b", "c", "d"]},
            "Route: evidence agent (5 model calls, 4 searches).",
        ),
    ],
)
def test_route_text_is_one_line(capsys, trace, expected):
    cli._print_route(trace)
    assert capsys.readouterr().out == expected + "\n"


@pytest.mark.parametrize("command", ["ask", "serve", "mcp"])
@pytest.mark.parametrize(
    "options,message",
    [
        (["--api-base", "https://provider.invalid/v1"], "--api-model is required"),
        (["--api-base", "http://remote.invalid/v1", "--api-model", "cloud"], "HTTPS URL"),
        (["--api-model", "cloud"], "--api-base is required"),
    ],
)
def test_api_cli_validation_shared(command, options, message):
    arguments = [command, "synthetic-document"]
    if command == "ask":
        arguments.append("question")
    args = cli.build_parser().parse_args([*arguments, "--backend", "api", *options])
    with pytest.raises(ModelError, match=message):
        cli._generation_config(args)


@pytest.mark.parametrize("command", ["ask", "serve", "mcp"])
def test_api_cli_warning_and_mock_generation(command, monkeypatch, capsys):
    import httpx

    from page_cite_rag.api_backend import ApiChatClient
    from page_cite_rag.generation import ChatMessage

    monkeypatch.setenv("CUSTOM_CLOUD_KEY", "cli-key-canary")
    monkeypatch.setattr(ApiChatClient, "_process_calls", 0)
    args = cli.build_parser().parse_args(
        [
            command,
            "synthetic-document",
            *(["question"] if command == "ask" else []),
            "--backend",
            "api",
            "--api-base",
            "https://provider.invalid/v1",
            "--api-model",
            "cloud",
            "--api-key-env",
            "CUSTOM_CLOUD_KEY",
            "--api-max-calls",
            "2",
        ]
    )
    config = cli._generation_config(args)
    assert args.token_budget == "conservative"
    real_init = ApiChatClient.__init__
    sent = []

    def handler(request):
        sent.append(request)
        assert request.headers["authorization"] == "Bearer cli-key-canary"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "synthetic answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    def initialize(self, *args, **kwargs):
        real_init(self, *args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(ApiChatClient, "__init__", initialize)
    with cli._open_generation_client(args, config) as client:
        assert client.max_calls == 2
        assert (
            client.generate([ChatMessage(role="user", content="synthetic")], None).value
            == "synthetic answer"
        )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err
        == "Document passages and questions will be sent to provider.invalid (--backend api).\n"
    )
    assert len(sent) == 1


def test_api_cli_provider_default_base_and_custom_still_requires_base(monkeypatch):
    args = cli.build_parser().parse_args(
        [
            "ask",
            "synthetic-document",
            "question",
            "--backend",
            "api",
            "--api-provider",
            "gemini",
            "--api-model",
            "free-model",
        ]
    )
    cli._generation_config(args)
    assert args.api_base is None
    custom = cli.build_parser().parse_args(
        [
            "ask",
            "synthetic-document",
            "question",
            "--backend",
            "api",
            "--api-model",
            "free-model",
        ]
    )
    with pytest.raises(ModelError, match="--api-base is required"):
        cli._generation_config(custom)


def test_api_cli_agent_still_requires_native_retrieval(document, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_open_generation_client", lambda *args: pytest.fail("no model dispatch")
    )
    arguments = ask_arguments(
        document,
        "--route",
        "agent",
        "--backend",
        "api",
        "--api-base",
        "https://provider.invalid/v1",
        "--api-model",
        "cloud",
    )
    assert cli.main(arguments) == 1
    assert "agent requires --native-gpu" in capsys.readouterr().err


def test_api_cli_agent_admits_gpu_retrieval_without_local_generation(
    document, clients, monkeypatch, capsys, tokenizer_directory
):
    from page_cite_rag import application, gpu_retrieval

    store, document_id, _ = document
    build_bm25_index(store, document_id)
    retrieval = []
    reached = []

    def staged(*args, **kwargs):
        retrieval.append(kwargs)
        return object()

    def routed(workflow, question, **kwargs):
        reached.append(workflow)
        raise ModelError("API Agent route reached")

    monkeypatch.setattr(gpu_retrieval, "StagedGpuSearch", staged)
    monkeypatch.setattr(application, "answer_question", routed)
    assert (
        cli.main(
            ask_arguments(
                document,
                "--backend",
                "api",
                "--api-base",
                "https://provider.invalid/v1",
                "--api-model",
                "cloud",
                "--route",
                "agent",
                "--method",
                "hybrid",
                "--native-gpu",
                "--gpu-uuid",
                "GPU-12345678-1234-1234-1234-123456789abc",
                "--model-dir",
                str(tokenizer_directory),
                "--gpu-temperature-limit",
                "88",
            )
        )
        == 1
    )
    assert len(clients) == len(reached) == len(retrieval) == 1
    args, config, _ = clients[0]
    assert args.device == "cpu"  # No local generation GPU is required.
    assert args.backend == "api"
    assert args.token_budget == "conservative"
    assert retrieval[0]["gpu_temperature_limit_c"] == 88
    assert "API Agent route reached" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["serve", "mcp"])
def test_api_service_configuration_accepts_conservative_model(document, monkeypatch, command):
    from page_cite_rag.generation import GenerationConfig

    store, document_id, _ = document
    build_bm25_index(store, document_id)
    args = cli.build_parser().parse_args(
        [
            command,
            document_id,
            "--db",
            str(store.database_path),
            "--method",
            "bm25",
            "--backend",
            "api",
            "--api-base",
            "https://provider.invalid/v1",
            "--api-model",
            "cloud",
        ]
    )
    monkeypatch.setattr(cli, "_open_generation_client", lambda *args: pytest.fail("must be lazy"))
    service = cli._create_document_service(args)
    assert service is not None
    assert args.token_budget == "conservative"
    assert isinstance(cli._generation_config(args), GenerationConfig)


@pytest.mark.parametrize("command", ["serve", "mcp"])
def test_local_service_own_ollama_keeps_original_default_error(command):
    args = cli.build_parser().parse_args([command, "synthetic", "--backend", "ollama"])
    with pytest.raises(ValueError, match="exact token budget requires an owned backend"):
        cli._generation_config(args)
