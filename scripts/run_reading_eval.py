"""Prepare and run the frozen reading comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

from page_cite_rag.answer_metrics import AnswerPrediction, AnswerReference, score_prediction
from page_cite_rag.dense import build_dense_index, dense_index_kind
from page_cite_rag.embedding import DEFAULT_MODEL_DIR
from page_cite_rag.embedding_native import NativeEmbeddingEncoder
from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH, NativeEmbeddingServer
from page_cite_rag.generation import ContextBudgetError
from page_cite_rag.gpu_retrieval import StagedGpuSearch
from page_cite_rag.llama import DEFAULT_BUNDLE, LlamaConfig, LlamaModelClient
from page_cite_rag.models import DocumentMetadata, ParsedUnit
from page_cite_rag.qasper import Paper, Passage, ingest_paper
from page_cite_rag.retrieval import (
    SearchResult,
    build_bm25_index,
)
from page_cite_rag.store import SQLiteDocumentStore

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "outputs/reading-eval-20261020-v3"
DEFAULT_RUN_ROOT = FROZEN_ROOT / "executions/unified-reading-v1"
GPU_UUID = os.environ.get("PAGE_CITE_RAG_GPU_UUID", "")
GPU_TEMPERATURE_LIMIT_C = 88
CONTEXT_TOKENS = 16_384
EXACT_INPUT_LIMIT = 12_000
AGENT_FINAL_INPUT_LIMIT = 10_000
OUTPUT_TOKENS = 1_536
TEMPLATE_RESERVE = 1_024
TOP_K = 10
CONTEXT_NEIGHBOR_BLOCKS = 0
PREP_DOCUMENT_EMBED_TOKEN_LIMIT = 500_000
PREP_QUERY_EMBED_TOKEN_LIMIT = 25_000
MAX_INVOCATION_WALL_SECONDS = 50 * 60
ARMS = ("fixed-rag", "single-agent", "delegated-agent")
RUNNER_VERSION = "frozen-reading-eval-v3-context-budget-12k"
_PASSAGE_ID = re.compile(r"/s(\d{4,})/p(\d{4,})$")
_LONG_BENCH_ZH_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃《》「」『』【】〖〗〔〕〘〙〚〛〜〝〞〟〰〾〿"
    "–—‘’‛“”„‟…‧﹏."
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _answer_format_for(case: dict) -> str:
    """Bind each benchmark's scoring unit to the model's output contract."""
    formats = {
        "qasper": "short_answer",
        "longbench-v1-multifieldqa-zh": "short_answer",
        "longbench-v2-single-document-qa": "multiple_choice",
    }
    try:
        return formats[case["dataset"]]
    except (KeyError, TypeError):
        raise ValueError("unknown frozen dataset task") from None


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL object in {path.name}")
    return rows


def atomic_json_once(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as target:
        temporary = Path(target.name)
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.chmod(0o600)
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"refusing to overwrite existing artifact: {path.name}") from None
    temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_jsonl_once(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as target:
        temporary = Path(target.name)
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.chmod(0o600)
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"refusing to overwrite existing artifact: {path.name}") from None
    temporary.unlink()


def _file_hashes(path: Path) -> dict[str, str]:
    if path.is_file():
        return {path.name: sha256(path)}
    return {
        str(item.relative_to(path)): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }


def load_frozen(root: Path) -> tuple[dict, list[dict]]:
    manifest_path = root / "manifest.json"
    inputs_path = root / "inputs.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files"]["inputs.jsonl"]
    if inputs_path.is_symlink() or inputs_path.stat().st_size != expected["bytes"]:
        raise ValueError("frozen file identity differs: inputs.jsonl")
    if sha256(inputs_path) != expected["sha256"]:
        raise ValueError("frozen file checksum differs: inputs.jsonl")
    inputs = read_jsonl(inputs_path)
    expected_ids = [case["case_id"] for case in manifest["cases"]]
    if len(inputs) != 24 or [row.get("case_id") for row in inputs] != expected_ids:
        raise ValueError("frozen input rows do not match the manifest")
    return manifest, inputs


def load_gold(root: Path, manifest: dict) -> list[dict]:
    """Read gold only in the scoring phase, never while preparing or generating."""
    path = root / "gold.jsonl"
    expected = manifest["files"]["gold.jsonl"]
    if path.is_symlink() or path.stat().st_size != expected["bytes"]:
        raise ValueError("frozen file identity differs: gold.jsonl")
    if sha256(path) != expected["sha256"]:
        raise ValueError("frozen file checksum differs: gold.jsonl")
    rows = read_jsonl(path)
    expected_ids = [case["case_id"] for case in manifest["cases"]]
    if len(rows) != 24 or [row.get("case_id") for row in rows] != expected_ids:
        raise ValueError("frozen gold rows do not match the manifest")
    return rows


def _case_document(store: SQLiteDocumentStore, case: dict) -> tuple[str, dict[str, str]]:
    source = case["source"]
    if source.get("format") == "qasper_paragraphs":
        passages = []
        section_count = 0
        for value in source["paragraphs"]:
            match = _PASSAGE_ID.search(value["passage_id"])
            if match is None:
                raise ValueError("frozen QASPER passage ID has an unsupported format")
            section_index, paragraph_index = map(int, match.groups())
            section_count = max(section_count, section_index + 1)
            passages.append(
                Passage(value["passage_id"], section_index, paragraph_index, value["text"])
            )
        if not passages:
            raise ValueError("QASPER document has no nonempty source paragraphs")
        paper = Paper(
            case["source_id"],
            tuple(passages),
            (),
            tuple(None for _ in range(section_count)),
        )
        return ingest_paper(store, paper)

    if source.get("format") != "plain_text" or not isinstance(source.get("text"), str):
        raise ValueError("frozen source format is unsupported")
    text = source["text"]
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    version = "frozen-reading-eval/plain-text-v1"
    document_id = hashlib.sha256(
        f"{RUNNER_VERSION}\0{case['dataset']}\0{case['case_id']}\0{source_hash}".encode()
    ).hexdigest()
    document = DocumentMetadata(
        document_id=document_id,
        source_sha256=source_hash,
        source_name=f"{case['source_id']}.txt",
        media_type="text/plain",
        parser_name="plain_text",
        parser_version=version,
        language=case["language"],
        usage_scope="public benchmark evaluation",
    )
    store.ingest(document, (ParsedUnit(unit_index=1, text=text, status="ok"),))
    return document_id, {}


class _CountingTransport:
    def __init__(self, transport) -> None:
        self._transport = transport
        self.query_prefix: str | None = None
        self.document_tokens = 0
        self.query_tokens = 0
        self.embed_inputs = 0

    def token_ids(self, text: str) -> list[int]:
        return self._transport.token_ids(text)

    def embed(self, texts: list[str]):
        document_count = 0
        query_count = 0
        for text in texts:
            count = len(self._transport.token_ids(text))
            if self.query_prefix is not None and text.startswith(self.query_prefix):
                query_count += count
            else:
                document_count += count
        if self.document_tokens + document_count > PREP_DOCUMENT_EMBED_TOKEN_LIMIT:
            raise ValueError("document embedding input token budget exceeded before dispatch")
        if self.query_tokens + query_count > PREP_QUERY_EMBED_TOKEN_LIMIT:
            raise ValueError("query embedding input token budget exceeded before dispatch")
        self.document_tokens += document_count
        self.query_tokens += query_count
        self.embed_inputs += len(texts)
        return self._transport.embed(texts)


def _source_code_hashes() -> dict[str, str]:
    package_files = sorted((ROOT / "src/page_cite_rag").glob("*.py"))
    paths = [
        ROOT / "scripts/prepare_reading_eval.py",
        ROOT / "scripts/run_reading_eval.py",
        *package_files,
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def static_configuration(
    root: Path,
    manifest: dict,
    selected_case_ids: list[str] | None = None,
    selected_arms: list[str] | None = None,
) -> dict:
    from page_cite_rag.embedding_native import GGUF_REVISION, GGUF_SHA256
    from page_cite_rag.full_answer import PROMPT_VERSION
    from page_cite_rag.llama import LLAMA_COMMIT, MODEL_REVISION, MODEL_SHA256
    from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL

    return {
        "runner_version": RUNNER_VERSION,
        "frozen_inputs_sha256": manifest["files"]["inputs.jsonl"]["sha256"],
        "run_selection": {
            "case_ids": selected_case_ids or [case["case_id"] for case in manifest["cases"]],
            "arms": selected_arms or list(ARMS),
        },
        "source_code_sha256": _source_code_hashes(),
        "models": {
            "generator": {
                "name": "Qwen3.5-9B-Q4_K_M",
                "revision": MODEL_REVISION,
                "sha256": MODEL_SHA256,
                "llama_commit": LLAMA_COMMIT,
            },
            "embedding": {
                "name": "Qwen3-Embedding-0.6B-f16-GGUF",
                "revision": GGUF_REVISION,
                "sha256": GGUF_SHA256,
            },
        },
        "retrieval": {
            "method": "BM25S + Qwen dense + reciprocal-rank fusion",
            **FORMAL_RETRIEVAL.as_dict(),
            "context_neighbor_blocks": CONTEXT_NEIGHBOR_BLOCKS,
            "query_rewrite": False,
            "reranker": False,
        },
        "answer": {
            "contract": PROMPT_VERSION,
            "task_adapter": "benchmark-answer-format/v1",
            "task_formats": {
                "qasper": "short_answer",
                "longbench-v1-multifieldqa-zh": "short_answer",
                "longbench-v2-single-document-qa": "multiple_choice",
            },
            "language": "per frozen case",
            "thinking": False,
            "sampling_profile": "greedy",
            "seed": 42,
            "num_ctx": CONTEXT_TOKENS,
            "max_exact_input_tokens": EXACT_INPUT_LIMIT,
            "num_predict": OUTPUT_TOKENS,
            "template_reserve": TEMPLATE_RESERVE,
            "timeout_seconds": 120,
            "retry": False,
        },
        "agent": {
            "framework": "LangChain create_agent",
            "max_decision_calls": 11,
            "max_total_input_tokens": 50_000,
            "max_total_output_tokens": 2_560,
            "reserved_final_input_tokens": AGENT_FINAL_INPUT_LIMIT,
            "reserved_final_output_tokens": OUTPUT_TOKENS,
            "max_tool_calls": 12,
            "max_readers": 2,
            "sequential": True,
            "shared_budget": True,
        },
        "gpu": {
            "uuid": GPU_UUID,
            "memory_limit_mib": 12_000,
            "temperature_limit_c": GPU_TEMPERATURE_LIMIT_C,
            "headroom_below_driver_max_operating_c": 5,
            "thermal_slowdown_must_be_inactive": True,
            "max_simultaneous_sequences": 1,
        },
        "preparation_budget": {
            "document_embedding_input_tokens_max": PREP_DOCUMENT_EMBED_TOKEN_LIMIT,
            "query_embedding_input_tokens_max": PREP_QUERY_EMBED_TOKEN_LIMIT,
        },
        "dataset_manifest_sha256": sha256(root / "manifest.json"),
        "gold_not_loaded_during_preparation_or_generation": True,
    }


def _ensure_run_config(run_root: Path, static: dict) -> None:
    path = run_root / "configuration.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("static") != static:
            raise ValueError("run configuration/code/data changed; refusing to mix results")
        return
    run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json_once(path, {"static": static, "state": "configuration_frozen"})


def _prepare(root: Path, run_root: Path, manifest: dict, inputs: list[dict], static: dict) -> dict:
    _ensure_run_config(run_root, static)
    prepared_dir = run_root / "prepared"
    ready_path = prepared_dir / "prepared.json"
    if not ready_path.is_file() and prepared_dir.exists() and any(prepared_dir.iterdir()):
        raise ValueError(
            "preparation artifacts are incomplete; preserve them and use a new run-root"
        )
    prepared_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    database_path = prepared_dir / "reading.sqlite3"
    retrieval_path = prepared_dir / "retrieval.jsonl"
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("static_configuration_sha256") != canonical_hash(static):
            raise ValueError("prepared store does not match the frozen run configuration")
        if ready.get("retrieval_sha256") != sha256(retrieval_path):
            raise ValueError("prepared retrieval results changed")
        if ready.get("database_files_sha256") != _file_hashes(database_path):
            raise ValueError("prepared database changed")
        index_root = database_path.with_name(database_path.name + ".indexes")
        if ready.get("index_files_sha256") != _file_hashes(index_root):
            raise ValueError("prepared indexes changed")
        return ready

    store = SQLiteDocumentStore(database_path)
    documents: dict[str, tuple[str, dict[str, str]]] = {}
    for case in inputs:
        if case["case_id"] not in documents:
            documents[case["case_id"]] = _case_document(store, case)

    document_tokens = query_tokens = embed_inputs = 0
    retrieval_rows = []
    with NativeEmbeddingServer(
        DEFAULT_GGUF_PATH,
        GPU_UUID,
        max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C,
    ) as server:
        counting_transport = _CountingTransport(server)
        encoder = NativeEmbeddingEncoder(counting_transport, DEFAULT_MODEL_DIR, batch_size=4)
        counting_transport.query_prefix = encoder.identity["query_prompt"]

        expected_document_tokens = 0
        expected_document_inputs = 0
        for document_id, _passage_by_block in documents.values():
            for block in store.get_blocks(document_id):
                for start, end in encoder.split_document(block.text):
                    expected_document_tokens += len(server.token_ids(block.text[start:end]))
                    expected_document_inputs += 1
        query_token_counts = [
            len(server.token_ids(encoder.identity["query_prompt"] + case["question"]))
            for case in inputs
        ]
        expected_query_tokens = sum(query_token_counts)
        embedding_preflight = {
            "state": "preflight_passed_not_embedded",
            "static_configuration_sha256": canonical_hash(static),
            "document_embedding_input_tokens": expected_document_tokens,
            "query_embedding_input_tokens": expected_query_tokens,
            "embedding_input_count": expected_document_inputs + len(inputs),
            "query_max_tokens": encoder.identity["max_tokens"],
            "document_token_limit": PREP_DOCUMENT_EMBED_TOKEN_LIMIT,
            "query_token_limit": PREP_QUERY_EMBED_TOKEN_LIMIT,
            "gpu_temperature_limit_c": GPU_TEMPERATURE_LIMIT_C,
            "gpu_memory_limit_mib": 12_000,
        }
        if expected_document_tokens > PREP_DOCUMENT_EMBED_TOKEN_LIMIT:
            raise ValueError("document embedding input token preflight exceeds frozen budget")
        if expected_query_tokens > PREP_QUERY_EMBED_TOKEN_LIMIT:
            raise ValueError("query embedding input token preflight exceeds frozen budget")
        if any(count > encoder.identity["max_tokens"] for count in query_token_counts):
            raise ValueError("a frozen query exceeds the embedding model's exact input limit")
        atomic_json_once(prepared_dir / "embedding-preflight.json", embedding_preflight)

        for case in inputs:
            document_id, passage_by_block = documents[case["case_id"]]
            if store.get_retrieval_manifest(document_id, "bm25") is None:
                build_bm25_index(store, document_id)
            if store.get_retrieval_manifest(document_id, dense_index_kind(encoder)) is None:
                build_dense_index(store, document_id, encoder, versioned=True)
        for case in inputs:
            document_id, passage_by_block = documents[case["case_id"]]
            from page_cite_rag.application import document_retriever

            retriever = document_retriever(
                store, document_id, "hybrid", encoder=encoder, versioned_dense=True
            )
            result = retriever.search(case["question"], top_k=TOP_K)
            block_to_passage = passage_by_block
            candidate_passage_ids = list(
                dict.fromkeys(
                    passage_id
                    for branch in result.branches
                    for block_id in branch.block_ids
                    if (passage_id := block_to_passage.get(block_id)) is not None
                )
            )
            retrieval_rows.append(
                {
                    "case_id": case["case_id"],
                    "document_id": document_id,
                    "question_sha256": hashlib.sha256(case["question"].encode("utf-8")).hexdigest(),
                    "retrieved_passage_ids": [
                        passage_by_block[hit.block.block_id]
                        for hit in result.hits
                        if hit.block.block_id in passage_by_block
                    ],
                    "candidate_passage_ids": candidate_passage_ids,
                    "block_to_passage_id": passage_by_block,
                    "result": result.model_dump(mode="json"),
                }
            )
        gpu_trace = server.diagnostics()
        document_tokens = counting_transport.document_tokens
        query_tokens = counting_transport.query_tokens
        embed_inputs = counting_transport.embed_inputs

        if document_tokens != expected_document_tokens or query_tokens != expected_query_tokens:
            raise ValueError("actual embedding token usage differs from exact preflight")

    atomic_jsonl_once(retrieval_path, retrieval_rows)
    for path in prepared_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o600)
    index_root = database_path.with_name(database_path.name + ".indexes")
    ready = {
        "state": "prepared_not_run",
        "case_count": len(inputs),
        "document_count": len(documents),
        "retrieval_sha256": sha256(retrieval_path),
        "database_files_sha256": _file_hashes(database_path),
        "index_files_sha256": _file_hashes(index_root),
        "embedding_identity": encoder.identity,
        "embedding_inputs": {
            "document_tokens": document_tokens,
            "query_tokens": query_tokens,
            "embed_inputs": embed_inputs,
            "document_token_limit": PREP_DOCUMENT_EMBED_TOKEN_LIMIT,
            "query_token_limit": PREP_QUERY_EMBED_TOKEN_LIMIT,
        },
        "gpu_trace": gpu_trace,
        "static_configuration_sha256": canonical_hash(static),
        "embedding_preflight_sha256": sha256(prepared_dir / "embedding-preflight.json"),
    }
    atomic_json_once(ready_path, ready)
    return ready


def _load_prepared(run_root: Path, static: dict) -> tuple[SQLiteDocumentStore, dict[str, dict]]:
    configured = json.loads((run_root / "configuration.json").read_text(encoding="utf-8"))
    if configured.get("static") != static:
        raise ValueError("configuration changed after freezing; refusing to run")
    prepared_dir = run_root / "prepared"
    ready_path = prepared_dir / "prepared.json"
    retrieval_path = prepared_dir / "retrieval.jsonl"
    if not ready_path.is_file() or not retrieval_path.is_file():
        raise ValueError("run `--phase prepare` before preflight or generation")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if (
        ready.get("static_configuration_sha256") != canonical_hash(static)
        or ready.get("retrieval_sha256") != sha256(retrieval_path)
        or ready.get("embedding_preflight_sha256")
        != sha256(prepared_dir / "embedding-preflight.json")
    ):
        raise ValueError("prepared artifacts differ from the frozen run")
    database_path = prepared_dir / "reading.sqlite3"
    if ready.get("database_files_sha256") != _file_hashes(database_path):
        raise ValueError("prepared database changed")
    index_root = database_path.with_name(database_path.name + ".indexes")
    if ready.get("index_files_sha256") != _file_hashes(index_root):
        raise ValueError("prepared indexes changed")
    retrieval = {row["case_id"]: row for row in read_jsonl(retrieval_path)}
    return SQLiteDocumentStore(database_path, read_only=True), retrieval


def _search_result_from_json(value: dict) -> SearchResult:
    """Validate a JSON-decoded result while preserving strict nested schemas."""
    return SearchResult.model_validate_json(
        json.dumps(value, ensure_ascii=False, allow_nan=False), strict=True
    )


def _llama_config() -> LlamaConfig:
    return LlamaConfig(
        bundle_dir=DEFAULT_BUNDLE,
        device="cuda",
        gpu_uuid=GPU_UUID,
        num_ctx=CONTEXT_TOKENS,
        num_predict=OUTPUT_TOKENS,
        template_reserve=TEMPLATE_RESERVE,
        timeout_seconds=120,
        max_gpu_memory_mib=12_000,
        max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C,
    )


def _preflight(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    store, retrieval = _load_prepared(run_root, static)
    preflight_path = run_root / "preflight.json"
    if preflight_path.exists():
        saved = json.loads(preflight_path.read_text(encoding="utf-8"))
        if saved.get("static_configuration_sha256") != canonical_hash(static):
            raise ValueError("preflight was made with different settings")
        return saved

    from page_cite_rag.application import FrozenSearchRetriever, document_workflow

    report = {}
    started = time.monotonic()
    with LlamaModelClient(
        _llama_config(), budget_policy="exact", recover_output_errors=False
    ) as model:
        generator_identity = model.identity
        for case in inputs:
            row = retrieval[case["case_id"]]
            search = _search_result_from_json(row["result"])
            workflow = document_workflow(
                store,
                row["document_id"],
                FrozenSearchRetriever(search, top_k=TOP_K),
                model,
                model.config,
                top_k=TOP_K,
                answer_format=_answer_format_for(case),
            )
            blocks = workflow._context_blocks(search, workflow._read_hits(search, case["question"]))
            admitted, omitted, _messages, budget = workflow._context(
                case["question"], case["language"], "local", blocks, None
            )
            if budget.exact_input_tokens is None or budget.exact_input_tokens > EXACT_INPUT_LIMIT:
                raise ContextBudgetError("fixed final-answer prompt failed exact preflight")
            report[case["case_id"]] = {
                "document_id": row["document_id"],
                "exact_final_input_tokens": budget.exact_input_tokens,
                "admitted_block_ids": [block.block_id for block in admitted],
                "retrieved_block_ids": [hit.block.block_id for hit in search.hits],
                "omitted_block_ids": list(omitted),
            }
        gpu = {
            "device_uuid": GPU_UUID,
            "max_gpu_temperature_c": GPU_TEMPERATURE_LIMIT_C,
            **_safe_model_diagnostics(model),
        }
    preflight = {
        "state": "preflight_complete_no_generation",
        "static_configuration_sha256": canonical_hash(static),
        "generator_identity": generator_identity,
        "cases": report,
        "gpu": gpu,
        "wall_seconds": time.monotonic() - started,
        "completion_calls": 0,
    }
    atomic_json_once(preflight_path, preflight)
    return preflight


def _safe_model_diagnostics(model) -> dict:
    value = model.diagnostics()
    samples = value.get("monitor_samples", [])
    health_samples = [sample for sample in samples if isinstance(sample, dict)]
    max_operating_values = [
        sample["max_operating_c"]
        for sample in health_samples
        if type(sample.get("max_operating_c")) is int
    ]
    return {
        "guard_error": value.get("guard_error"),
        "monitor_sample_count": value.get("monitor_sample_count", len(samples)),
        "observed_gpu_peak": value.get("observed_gpu_peak"),
        "max_operating_c": max(max_operating_values, default=None),
        "hw_thermal_slowdown_seen": any(
            sample.get("hw_thermal_slowdown") is True for sample in health_samples
        ),
        "sw_thermal_slowdown_seen": any(
            sample.get("sw_thermal_slowdown") is True for sample in health_samples
        ),
        "last_completion": value.get("last_completion"),
        "server_exit_code": value.get("exit_code_before_termination"),
    }


class _CountingModelClient:
    """Count calls entering the shared final-generation API without touching responses."""

    def __init__(self, model) -> None:
        self._model = model
        self.generate_calls = 0

    def __getattr__(self, name):
        return getattr(self._model, name)

    def generate(self, *args, **kwargs):
        self.generate_calls += 1
        return self._model.generate(*args, **kwargs)


def _run_arm(
    run_root: Path,
    static: dict,
    inputs: list[dict],
    arm: str,
    case_id: str | None,
) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown evaluation arm")
    if arm not in static.get("run_selection", {}).get("arms", ARMS):
        raise ValueError("run arm is outside the frozen comparison scope")
    preflight_path = run_root / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError("run exact-token preflight before model generation")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("static_configuration_sha256") != canonical_hash(static):
        raise ValueError("preflight does not match the frozen run configuration")
    store, retrieval = _load_prepared(run_root, static)
    selected = [case for case in inputs if case_id is None or case["case_id"] == case_id]
    if not selected or (case_id is not None and selected[0]["case_id"] != case_id):
        raise ValueError("requested case ID is not in the frozen input manifest")

    from page_cite_rag.agent import (
        AgentRunFailure,
        _admit,
        local_chat_model,
        run_bounded_reading_agent,
    )
    from page_cite_rag.application import FrozenSearchRetriever, document_workflow
    from page_cite_rag.workflows import RetrievalTrace

    arm_dir = run_root / arm
    arm_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt_dir = arm_dir / ".attempts"
    attempt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = time.monotonic()
    attempted = 0
    completed = 0
    stopped = None
    with LlamaModelClient(
        _llama_config(),
        budget_policy="exact",
        recover_output_errors=False,
        sampling_profile="greedy",
        sampling_seed=42,
    ) as model:
        if model.identity != preflight["generator_identity"]:
            raise ValueError("generator identity differs from the frozen preflight")
        counted_model = _CountingModelClient(model)
        for case in selected:
            target = arm_dir / f"{case['case_id']}.json"
            if target.exists():
                continue
            attempt_marker = attempt_dir / f"{case['case_id']}.json"
            if attempt_marker.exists():
                continue
            if time.monotonic() - started >= MAX_INVOCATION_WALL_SECONDS:
                stopped = "invocation_wall_limit"
                break
            case_started = time.monotonic()
            row = retrieval[case["case_id"]]
            search = _search_result_from_json(row["result"])
            retriever = FrozenSearchRetriever(search, top_k=TOP_K)
            if arm != "fixed-rag":
                retriever = StagedGpuSearch(
                    store,
                    row["document_id"],
                    "hybrid",
                    DEFAULT_GGUF_PATH,
                    GPU_UUID,
                    tokenizer_directory=DEFAULT_MODEL_DIR,
                    gpu_temperature_limit_c=GPU_TEMPERATURE_LIMIT_C,
                )
            workflow = document_workflow(
                store,
                row["document_id"],
                retriever,
                counted_model,
                model.config,
                top_k=TOP_K,
                answer_format=_answer_format_for(case),
            )
            blocks = workflow._context_blocks(search, workflow._read_hits(search, case["question"]))
            context = _admit(workflow, case["question"], case["language"], blocks, ())
            if (
                context.exact_input_tokens
                != preflight["cases"][case["case_id"]]["exact_final_input_tokens"]
                or [block.block_id for block in context.blocks]
                != preflight["cases"][case["case_id"]]["admitted_block_ids"]
                or [
                    block.block_id
                    for block in blocks
                    if block.block_id not in {item.block_id for item in context.blocks}
                ]
                != preflight["cases"][case["case_id"]]["omitted_block_ids"]
            ):
                raise ValueError("runtime initial context differs from exact preflight")

            atomic_json_once(
                attempt_marker,
                {
                    "state": "attempt_started_no_retry",
                    "arm": arm,
                    "case_id": case["case_id"],
                    "static_configuration_sha256": canonical_hash(static),
                    "initial_search_sha256": canonical_hash(row["result"]),
                    "started_monotonic_seconds": time.monotonic(),
                },
            )
            agent_run = None
            final_attempted = False
            result = None
            failure = None
            final_calls_before = counted_model.generate_calls
            attempted += 1
            try:
                if arm == "fixed-rag":
                    final_context = context
                else:
                    with local_chat_model(model) as chat_model:
                        with retriever.session():
                            agent_run = run_bounded_reading_agent(
                                workflow,
                                case["question"],
                                case["language"],
                                search,
                                blocks,
                                chat_model,
                                count_chat_input_tokens=model.count_chat_input_tokens,
                                delegated=arm == "delegated-agent",
                            )
                    final_context = agent_run.context
                result = workflow.answer_from_context(
                    case["question"],
                    final_context.blocks,
                    language=case["language"],
                    context_groups=final_context.context_groups,
                )
                final_attempted = counted_model.generate_calls > final_calls_before
                result = result.model_copy(
                    update={
                        "retrieval": RetrievalTrace(
                            method=search.method,
                            index_id=search.index_id,
                            query=search.query,
                            hit_count=len(search.hits),
                            elapsed_ms=search.elapsed_ms,
                            query_plan=None,
                            branches=search.branches,
                        )
                    }
                )
                completed += 1
            except Exception as exc:
                final_attempted = counted_model.generate_calls > final_calls_before
                failure = {
                    "error_type": exc.error_type
                    if isinstance(exc, AgentRunFailure)
                    else type(exc).__name__,
                    "agent_trace": exc.trace if isinstance(exc, AgentRunFailure) else None,
                    "final_answer_attempted": final_attempted,
                }
                if accounting := getattr(exc, "accounting", None):
                    failure["accounting"] = accounting.model_dump(mode="json")
                if output := getattr(exc, "raw_model_output", None):
                    failure["raw_model_output"] = output
                if errors := getattr(exc, "validation_errors", ()):
                    failure["validation_errors"] = list(errors)

            diagnostics = _safe_model_diagnostics(model)
            generation_stats = (
                result.generation_stats.model_dump(mode="json")
                if result is not None and result.generation_stats is not None
                else None
            )
            agent_trace = (
                agent_run.trace
                if agent_run is not None
                else (failure.get("agent_trace") if failure else None)
            )
            agent_input = agent_trace.get("agent_input_tokens_exact") if agent_trace else 0
            agent_call_records = agent_trace.get("model_call_records", []) if agent_trace else []
            agent_output_lower_bound = sum(
                call.get("output_tokens_observed", 0)
                for call in agent_call_records
                if type(call.get("output_tokens_observed")) is int
            )
            agent_output_unknown = any(
                call.get("status") != "not_dispatched_cooldown_failed"
                and type(call.get("output_tokens_observed")) is not int
                for call in agent_call_records
            )
            failure_accounting = failure.get("accounting") if failure else None
            final_input = (
                generation_stats.get("prompt_eval_count")
                if generation_stats is not None
                else failure_accounting.get("prompt_eval_count")
                if isinstance(failure_accounting, dict)
                else None
            )
            final_output = (
                generation_stats.get("eval_count")
                if generation_stats is not None
                else failure_accounting.get("eval_count")
                if isinstance(failure_accounting, dict)
                else None
            )
            agent_calls = sum(
                call.get("status") != "not_dispatched_cooldown_failed"
                for call in agent_call_records
            )
            input_lower_bound = (agent_input if isinstance(agent_input, int) else 0) + (
                final_input if isinstance(final_input, int) else 0
            )
            output_lower_bound = agent_output_lower_bound + (
                final_output if isinstance(final_output, int) else 0
            )
            final_usage_unknown = final_attempted and (final_input is None or final_output is None)
            record = {
                "arm": arm,
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "answer_format": _answer_format_for(case),
                "execution": "ok" if result is not None else "failed",
                "failure": failure,
                "result": result.model_dump(mode="json") if result is not None else None,
                "agent_trace": agent_trace,
                "agent_input_tokens_exact": agent_input,
                "agent_output_tokens_observed": agent_trace.get("agent_output_tokens_observed")
                if agent_trace
                else 0,
                "agent_output_tokens_lower_bound": agent_output_lower_bound,
                "agent_output_tokens_reserved": (
                    agent_trace.get("agent_output_tokens_reserved") if agent_trace else 0
                ),
                "agent_model_call_records": agent_call_records,
                "final_answer_attempted": final_attempted,
                "final_input_tokens_observed": final_input,
                "final_output_tokens_observed": final_output,
                "input_tokens_lower_bound": input_lower_bound,
                "input_tokens_exact_or_observed": (
                    input_lower_bound if not final_usage_unknown else None
                ),
                "output_tokens_lower_bound": output_lower_bound,
                "output_usage_unknown": final_usage_unknown or agent_output_unknown,
                "completion_calls_attempted": agent_calls
                + counted_model.generate_calls
                - final_calls_before,
                "initial_search_sha256": canonical_hash(row["result"]),
                "block_to_passage_id": row.get("block_to_passage_id", {}),
                "candidate_passage_ids": row.get("candidate_passage_ids", []),
                "initial_top10_passage_ids": row.get("retrieved_passage_ids", []),
                "initial_admitted_block_ids": preflight["cases"][case["case_id"]][
                    "admitted_block_ids"
                ],
                "elapsed_ms": (time.monotonic() - case_started) * 1000,
                "model_call_elapsed_ms": [
                    call["elapsed_ms"]
                    for call in agent_call_records
                    if type(call.get("elapsed_ms")) in (int, float)
                ]
                + (
                    [generation_stats["elapsed_ms"]]
                    if generation_stats is not None
                    else [failure_accounting["elapsed_ms"]]
                    if isinstance(failure_accounting, dict)
                    and type(failure_accounting.get("elapsed_ms")) in (int, float)
                    else []
                ),
                "generation_stats": generation_stats,
                "gpu_diagnostics": diagnostics,
                "gpu_retrieval_trace": (
                    retriever.last_trace if isinstance(retriever, StagedGpuSearch) else None
                ),
                "no_retry": True,
                "gold_not_loaded": True,
            }
            atomic_json_once(target, record)
            print(
                f"{arm} {case['case_id']}: {record['execution']}, "
                f"{record['completion_calls_attempted']} generation calls",
                flush=True,
            )
            if diagnostics.get("guard_error"):
                stopped = "gpu_guard_stopped"
                break
            if time.monotonic() - started >= MAX_INVOCATION_WALL_SECONDS:
                stopped = "invocation_wall_limit"
                break

    result_state = {
        "arm": arm,
        "attempted_slots_this_invocation": attempted,
        "successful_slots_this_invocation": completed,
        "elapsed_seconds": time.monotonic() - started,
        "stopped": stopped,
        "remaining_slots": sum(
            not (arm_dir / f"{case['case_id']}.json").is_file() for case in selected
        ),
        "unresolved_attempt_slots": [
            case["case_id"]
            for case in selected
            if not (arm_dir / f"{case['case_id']}.json").is_file()
            and (attempt_dir / f"{case['case_id']}.json").is_file()
        ],
    }
    return result_state


def _longbench_zh_f1(prediction: str, references: list[str]) -> float:
    import string

    import jieba

    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()

    def tokens(text: str) -> list[str]:
        remove = set(string.punctuation + _LONG_BENCH_ZH_PUNCTUATION)
        return [
            token
            for raw in tokenizer.cut(text, cut_all=False)
            if (
                token := "".join(
                    "".join(char for char in raw.lower() if char not in remove).split()
                )
            )
        ]

    predicted = tokens(prediction)
    best = 0.0
    for reference in references:
        expected = tokens(reference)
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        score = 2 * overlap / (len(predicted) + len(expected)) if overlap else 0.0
        best = max(best, score)
    return best


def _longbench_v2_choice(prediction: str) -> str | None:
    response = prediction.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def _qasper_evidence_texts(
    result: dict | None,
    block_to_passage: dict[str, str],
    passage_texts: dict[str, str],
) -> tuple[str, ...]:
    passage_ids = _qasper_cited_passage_ids(result, block_to_passage)
    if any(passage_id not in passage_texts for passage_id in passage_ids):
        raise ValueError("QASPER answer citation has no original paragraph text")
    return tuple(passage_texts[passage_id] for passage_id in passage_ids)


def _qasper_cited_passage_ids(
    result: dict | None,
    block_to_passage: dict[str, str],
) -> tuple[str, ...]:
    if result is None:
        return ()
    passage_ids = []
    for claim in result.get("answer_claims", []):
        for citation in claim.get("citations", []):
            block_id = citation.get("block_id")
            passage_id = block_to_passage.get(block_id)
            if passage_id is None:
                raise ValueError("QASPER answer citation has no original paragraph mapping")
            if passage_id not in passage_ids:
                passage_ids.append(passage_id)
    return tuple(passage_ids)


def score_run(
    run_root: Path,
    manifest: dict,
    inputs: list[dict],
    gold: list[dict],
    arms: tuple[str, ...] = ARMS,
) -> dict:
    if (run_root / "scores.json").exists():
        raise ValueError("scores.json already exists; never overwrite a frozen score")
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("score arms must be a nonempty unique subset of the frozen arms")
    expected_cases = {case["case_id"] for case in manifest["cases"]}
    result_records = {}
    missing = {}
    for arm in arms:
        directory = run_root / arm
        records = {}
        for path in directory.glob("*.json") if directory.is_dir() else ():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("arm") != arm or value.get("case_id") != path.stem:
                raise ValueError("result path does not match its frozen arm/case identity")
            if path.stem in records:
                raise ValueError("duplicate result case")
            records[path.stem] = value
        unexpected = records.keys() - expected_cases
        if unexpected:
            raise ValueError("run contains result cases outside the selected evaluation scope")
        result_records[arm] = records
        missing[arm] = sorted(expected_cases - records.keys())

    if any(missing.values()):
        interrupted = {
            arm: sorted(
                case_id
                for case_id in missing[arm]
                if (run_root / arm / ".attempts" / f"{case_id}.json").is_file()
            )
            for arm in ARMS
        }
        report = {
            "state": "incomplete",
            "missing_slots_by_arm": missing,
            "unresolved_attempt_slots_by_arm": interrupted,
            "score_artifact_written": False,
        }
        return report

    gold_by_id = {row["case_id"]: row for row in gold}
    inputs_by_id = {row["case_id"]: row for row in inputs}
    if set(gold_by_id) != expected_cases or set(inputs_by_id) != expected_cases:
        raise ValueError("gold set does not match frozen case IDs")
    scores = {}
    for arm, records in result_records.items():
        datasets: dict[str, list[dict]] = {}
        attempted_generation_calls = total_input = total_output = 0
        unknown_output_rows = 0
        case_elapsed_ms = []
        model_call_elapsed_ms = []
        gpu_peak_memory = []
        gpu_peak_temperature = []
        for case in manifest["cases"]:
            case_id = case["case_id"]
            raw = records[case_id]
            prediction_result = raw.get("result") if raw.get("execution") == "ok" else None
            prediction_text = prediction_result.get("answer", "") if prediction_result else ""
            gold_row = gold_by_id[case_id]
            if case["dataset"] == "qasper":
                source = inputs_by_id[case_id]["source"]
                passage_texts = {item["passage_id"]: item["text"] for item in source["paragraphs"]}
                references = tuple(
                    AnswerReference(
                        answer=item["answer"],
                        evidence=tuple(item["evidence"]),
                        answer_type=item["answer_type"],
                    )
                    for item in gold_row["references"]
                )
                prediction = (
                    AnswerPrediction(
                        "Unanswerable"
                        if prediction_result
                        and prediction_result.get("status") == "insufficient_evidence"
                        else prediction_text,
                        _qasper_evidence_texts(
                            prediction_result,
                            raw.get("block_to_passage_id", {}),
                            passage_texts,
                        ),
                    )
                    if prediction_result is not None
                    else None
                )
                metrics = score_prediction(references, prediction)
                aligned = bool(gold_row["aligned_evidence_passage_ids"]) and not bool(
                    gold_row["evidence_alignment_issues"]
                )
                gold_evidence_ids = set(gold_row["aligned_evidence_passage_ids"])

                def evidence_recall(passage_ids: list[str] | tuple[str, ...]) -> float | None:
                    if not aligned:
                        return None
                    return len(gold_evidence_ids & set(passage_ids)) / len(gold_evidence_ids)

                cited_passage_ids = _qasper_cited_passage_ids(
                    prediction_result,
                    raw.get("block_to_passage_id", {}),
                )
                agent_trace = raw.get("agent_trace")
                final_block_ids = (
                    prediction_result.get("included_block_ids", [])
                    if prediction_result
                    else agent_trace.get("selected_block_ids", [])
                    if isinstance(agent_trace, dict)
                    else []
                )
                final_context_ids = [
                    raw.get("block_to_passage_id", {}).get(block_id) for block_id in final_block_ids
                ]
                tool_read_ids = []
                if isinstance(agent_trace, dict):
                    for event in agent_trace.get("tool_trace", []):
                        if event.get("status") == "ok" and event.get("tool") == "read_source":
                            tool_read_ids.append(event.get("block_id"))
                        tool_read_ids.extend(event.get("newly_admitted_block_ids", []))
                block_to_passage = raw.get("block_to_passage_id", {})
                coverage = {
                    "candidate_union_recall": evidence_recall(raw.get("candidate_passage_ids", [])),
                    "initial_top10_recall": evidence_recall(
                        raw.get("initial_top10_passage_ids", [])
                    ),
                    "initial_admitted_recall": evidence_recall(
                        [
                            block_to_passage[block_id]
                            for block_id in raw.get("initial_admitted_block_ids", [])
                            if block_id in block_to_passage
                        ]
                    ),
                    "tool_read_recall": (
                        evidence_recall(
                            [
                                block_to_passage[block_id]
                                for block_id in tool_read_ids
                                if block_id in block_to_passage
                            ]
                        )
                        if isinstance(agent_trace, dict)
                        else None
                    ),
                    "final_context_recall": evidence_recall(
                        [passage_id for passage_id in final_context_ids if passage_id]
                    ),
                    "answer_citation_recall": evidence_recall(list(cited_passage_ids)),
                }
                question_score = {
                    "case_id": case_id,
                    "answer_f1": metrics["answer_f1"],
                    "evidence_f1": metrics["evidence_f1"] if aligned else None,
                    "evidence_precision": metrics["evidence_precision"] if aligned else None,
                    "evidence_recall": metrics["evidence_recall"] if aligned else None,
                    "evidence_applicable": aligned,
                    "evidence_coverage": coverage,
                    "execution": raw["execution"],
                }
                if prediction_result is not None:
                    no_answer = gold_row["stratum"] == "no_answer"
                    question_score["correct_refusal"] = bool(
                        no_answer and prediction_result.get("status") == "insufficient_evidence"
                    )
                    question_score["false_refusal"] = bool(
                        not no_answer and prediction_result.get("status") == "insufficient_evidence"
                    )
            elif case["dataset"] == "longbench-v2-single-document-qa":
                choice = _longbench_v2_choice(prediction_text) if prediction_result else None
                question_score = {
                    "case_id": case_id,
                    "accuracy": float(choice == gold_row["answer"]),
                    "extracted_choice": choice,
                    "execution": raw["execution"],
                }
            elif case["dataset"] == "longbench-v1-multifieldqa-zh":
                question_score = {
                    "case_id": case_id,
                    "answer_f1": _longbench_zh_f1(prediction_text, gold_row["answers"]),
                    "execution": raw["execution"],
                }
            else:
                raise ValueError("unknown frozen dataset task")
            datasets.setdefault(case["dataset"], []).append(question_score)
            attempted_generation_calls += raw.get("completion_calls_attempted", 0)
            total_input += raw.get("input_tokens_lower_bound", 0)
            total_output += raw.get("output_tokens_lower_bound", 0)
            if raw.get("output_usage_unknown"):
                unknown_output_rows += 1
            if type(raw.get("elapsed_ms")) in (int, float):
                case_elapsed_ms.append(raw["elapsed_ms"])
            model_call_elapsed_ms.extend(raw.get("model_call_elapsed_ms", []))
            peak = raw.get("gpu_diagnostics", {}).get("observed_gpu_peak", {})
            if type(peak.get("memory_mib")) is int:
                gpu_peak_memory.append(peak["memory_mib"])
            if type(peak.get("temperature_c")) is int:
                gpu_peak_temperature.append(peak["temperature_c"])

        dataset_scores = {}
        for dataset, rows in datasets.items():
            metrics = {}
            for key in ("answer_f1", "accuracy"):
                values = [row[key] for row in rows if row.get(key) is not None]
                if values:
                    metrics[key] = sum(values) / len(values)
                    metrics[f"{key}_n"] = len(values)
            evidence_rows = [row for row in rows if row.get("evidence_applicable")]
            if evidence_rows:
                for key in ("evidence_f1", "evidence_precision", "evidence_recall"):
                    metrics[key] = sum(row[key] for row in evidence_rows) / len(evidence_rows)
                metrics["evidence_n"] = len(evidence_rows)
            coverage_rows = [row for row in evidence_rows if row.get("evidence_coverage")]
            if coverage_rows:
                for name in (
                    "candidate_union_recall",
                    "initial_top10_recall",
                    "initial_admitted_recall",
                    "tool_read_recall",
                    "final_context_recall",
                    "answer_citation_recall",
                ):
                    values = [
                        row["evidence_coverage"][name]
                        for row in coverage_rows
                        if row["evidence_coverage"][name] is not None
                    ]
                    if values:
                        metrics[f"{name}_mean"] = sum(values) / len(values)
                        metrics[f"{name}_n"] = len(values)
            refusal_rows = [row for row in rows if "correct_refusal" in row]
            if refusal_rows:
                metrics["correct_refusals"] = sum(row["correct_refusal"] for row in refusal_rows)
                metrics["false_refusals"] = sum(row["false_refusal"] for row in refusal_rows)
                metrics["refusal_n"] = len(refusal_rows)
            dataset_scores[dataset] = {
                "n": len(rows),
                "successful": sum(row["execution"] == "ok" for row in rows),
                "metrics": metrics,
                "per_case": rows,
            }
        scores[arm] = {
            "datasets": dataset_scores,
            "generation_calls_attempted": attempted_generation_calls,
            "input_tokens_lower_bound": total_input,
            "output_tokens_lower_bound": total_output,
            "rows_with_unknown_output_usage": unknown_output_rows,
            "latency_percentile_method": "nearest-rank",
            "case_elapsed_ms_p50": _percentile(case_elapsed_ms, 0.50),
            "case_elapsed_ms_p95": _percentile(case_elapsed_ms, 0.95),
            "model_call_elapsed_ms_p50": _percentile(model_call_elapsed_ms, 0.50),
            "model_call_elapsed_ms_p95": _percentile(model_call_elapsed_ms, 0.95),
            "observed_gpu_peak_memory_mib": max(gpu_peak_memory, default=None),
            "observed_gpu_peak_temperature_c": max(gpu_peak_temperature, default=None),
        }
    report = {
        "state": "scored_complete",
        "evaluation_scope": {
            "kind": "selected_case_and_arm_slice",
            "case_ids": [case["case_id"] for case in manifest["cases"]],
            "arms": list(arms),
            "case_count": len(expected_cases),
            "frozen_dataset_case_count": manifest.get("files", {})
            .get("inputs.jsonl", {})
            .get("rows", len(expected_cases)),
        },
        "dataset_metrics_are_not_combined": True,
        "semantic_claim_support_judged": False,
        "independent_human_review": False,
        "arms": scores,
    }
    atomic_json_once(run_root / "scores.json", report)
    return report


def _percentile(values: list[int | float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "preflight", "run", "score"), required=True)
    parser.add_argument("--frozen-root", type=Path, default=FROZEN_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--case-ids",
        nargs="+",
        help="freeze a selected case slice for prepare, preflight, run, and score",
    )
    parser.add_argument(
        "--compare-arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        help="arms included in this run configuration and score (default: all arms)",
    )
    args = parser.parse_args()
    try:
        if args.phase != "score" and not GPU_UUID:
            raise ValueError("set PAGE_CITE_RAG_GPU_UUID to an explicit GPU UUID")
        os.umask(0o077)
        root = args.frozen_root.expanduser().resolve()
        run_root = args.run_root.expanduser().resolve()
        manifest, inputs = load_frozen(root)
        all_case_ids = [case["case_id"] for case in manifest["cases"]]
        selected_case_ids = args.case_ids or all_case_ids
        if len(set(selected_case_ids)) != len(selected_case_ids):
            raise ValueError("selected case IDs must be unique")
        unknown_case_ids = set(selected_case_ids) - set(all_case_ids)
        if unknown_case_ids:
            raise ValueError("selected case ID is not in the frozen input manifest")
        if not args.compare_arms or len(set(args.compare_arms)) != len(args.compare_arms):
            raise ValueError("comparison arms must be a nonempty unique selection")
        input_by_id = {case["case_id"]: case for case in inputs}
        selected_inputs = [input_by_id[case_id] for case_id in selected_case_ids]
        if args.case_id is not None and args.case_id not in selected_case_ids:
            raise ValueError("run case ID must be included in the selected case slice")
        if args.phase == "run" and args.arm is None:
            raise ValueError("--phase run requires --arm")
        if args.phase == "run" and args.arm not in args.compare_arms:
            raise ValueError("run arm is not included in --compare-arms")
        static = static_configuration(root, manifest, selected_case_ids, args.compare_arms)
        if args.phase == "prepare":
            result = _prepare(root, run_root, manifest, selected_inputs, static)
        elif args.phase == "preflight":
            result = _preflight(run_root, static, selected_inputs)
        elif args.phase == "run":
            if args.arm is None:
                raise ValueError("--phase run requires --arm")
            result = _run_arm(run_root, static, selected_inputs, args.arm, args.case_id)
        else:
            gold = load_gold(root, manifest)
            selected_manifest = {
                **manifest,
                "cases": [
                    case
                    for case_id in selected_case_ids
                    for case in manifest["cases"]
                    if case["case_id"] == case_id
                ],
            }
            selected_case_id_set = set(selected_case_ids)
            selected_gold = [row for row in gold if row["case_id"] in selected_case_id_set]
            result = score_run(
                run_root,
                selected_manifest,
                selected_inputs,
                selected_gold,
                tuple(args.compare_arms),
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ContextBudgetError) as exc:
        print(f"reading-eval: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
