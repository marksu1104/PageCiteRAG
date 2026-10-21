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
        payload = json.loads(messages[1].content)
        first_evidence = payload["evidence"][0]
        if self.insufficient:
            value = schema_class.model_validate_json(
                json.dumps({"status": "insufficient_evidence", "limitations": ["Not stated."]})
            )
        else:
            citation = {
                "source_id": "S999" if self.invalid_citation else first_evidence["source_id"]
            }
            value = schema_class.model_validate_json(
                json.dumps(
                    {
                        "status": "answered",
                        "answer_claims": [
                            {
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


def ask_arguments(document, *options):
    store, document_id, _ = document
    return ["ask", document_id, "significant", "--db", str(store.database_path), *options]


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
            arguments[-1] = str(tmp_path / "PRIVATE_CANARY.sqlite3")
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
    assert cli.main(ask_arguments(document, "--top-k", "21")) == 1
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
        assert "--agent" in result.stdout
        assert "--native-gpu" in result.stdout


def test_agent_cli_uses_bounded_context_then_shared_answer_path(
    document, clients, monkeypatch, capsys
):
    from page_cite_rag import agent

    store, document_id, _ = document
    build_bm25_index(store, document_id)

    @contextmanager
    def local_model(_client):
        yield object()

    def run_agent(workflow, question, language, result, blocks, _model, **kwargs):
        assert callable(kwargs["count_chat_input_tokens"])
        assert kwargs["delegated"] is False
        context = agent._admit(workflow, question, language, blocks, result.context_groups)
        return agent.AgentRun(
            context,
            {"model_calls_completed": 1, "tool_calls_recorded": 1},
        )

    monkeypatch.setattr(agent, "local_chat_model", local_model)
    monkeypatch.setattr(agent, "run_bounded_reading_agent", run_agent)
    arguments = ask_arguments(
        document,
        "--backend",
        "llama",
        "--token-budget",
        "exact",
        "--agent",
        "--format",
        "json",
    )

    assert cli.main(arguments) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["answer"]["status"] == "answered"
    assert output["answer"]["retrieval"]["method"] == "bm25"
    assert output["agent_trace"] == {
        "model_calls_completed": 1,
        "tool_calls_recorded": 1,
    }
    assert len(clients) == 1 and len(clients[0][2].calls) == 1
    assert clients[0][2].close_count == 1


def test_native_gpu_ask_searches_before_opening_the_generation_model(
    document, clients, monkeypatch, capsys
):
    from page_cite_rag.retrieval import BM25Retriever

    store, document_id, _ = document
    build_bm25_index(store, document_id)
    events = []

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
        ):
            assert passed_store.database_path == store.database_path
            assert passed_document_id == document_id
            assert method == "hybrid"
            assert model_path.name == "Qwen3-Embedding-0.6B-f16.gguf"
            assert gpu_uuid == "GPU-12345678-1234-1234-1234-123456789abc"
            assert gpu_temperature_limit_c == 80
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


def test_native_gpu_agent_uses_owned_gpu_for_followup_search(
    document, clients, monkeypatch, capsys
):
    from contextlib import contextmanager

    from page_cite_rag import agent, gpu_retrieval
    from page_cite_rag.retrieval import BM25Retriever

    store, document_id, _ = document
    build_bm25_index(store, document_id)
    monkeypatch.setattr(
        cli,
        "_load_encoder",
        lambda _args: pytest.fail("native GPU must not silently fall back to CPU"),
    )
    events = []

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
        ):
            assert passed_store.database_path == store.database_path
            assert passed_document_id == document_id
            assert method == "hybrid"
            assert tokenizer_directory.is_dir()
            assert gpu_temperature_limit_c == 88
            assert clients == []
            events.append("GPU search setup")
            self.retriever = BM25Retriever(store, document_id)

        def search(self, query, *, top_k):
            events.append(f"GPU query: {query}")
            return self.retriever.search(query, top_k=top_k)

        @contextmanager
        def session(self):
            events.append("GPU session start")
            try:
                yield self
            finally:
                events.append("GPU session cleanup")

    @contextmanager
    def local_model(_client):
        yield object()

    def run_agent(workflow, question, language, result, blocks, _model, **kwargs):
        assert kwargs["delegated"] is False
        followup, _followup_blocks = workflow.search_sources("independent follow-up query")
        assert followup.query == "independent follow-up query"
        current = agent._admit(workflow, question, language, blocks, result.context_groups)
        return agent.AgentRun(
            current,
            {"model_calls_completed": 2, "tool_calls_recorded": 1},
        )

    monkeypatch.setattr(gpu_retrieval, "StagedGpuSearch", FakeStagedGpuSearch)
    monkeypatch.setattr(agent, "local_chat_model", local_model)
    monkeypatch.setattr(agent, "run_bounded_reading_agent", run_agent)
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
        "--gpu-temperature-limit",
        "88",
        "--agent",
        "--format",
        "json",
    )

    assert cli.main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["answer"]["status"] == "answered"
    assert output["agent_trace"]["model_calls_completed"] == 2
    assert events == [
        "GPU search setup",
        "GPU session start",
        "GPU query: significant",
        "generation startup",
        "GPU query: independent follow-up query",
        "GPU session cleanup",
    ]
    assert len(clients) == 1 and clients[0][2].close_count == 1
