"""Prepare and run the frozen reading comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import zipfile
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path

from page_cite_rag.agent import (
    MAX_AGENT_MODEL_CALLS,
    MAX_AGENT_TOOL_CALLS,
    MAX_AGENT_TOTAL_INPUT_TOKENS,
    MAX_AGENT_TOTAL_OUTPUT_TOKENS,
    MAX_EXACT_INPUT_TOKENS,
    MAX_INTEGRATED_AGENT_CALL_INPUT_TOKENS,
    MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
    MAX_INTEGRATED_AGENT_TOTAL_INPUT_TOKENS,
    MAX_READER_TASKS,
)
from page_cite_rag.answer_metrics import AnswerPrediction, AnswerReference, score_prediction
from page_cite_rag.dense import DenseRetriever, build_dense_index, dense_index_kind
from page_cite_rag.embedding import DEFAULT_MODEL_DIR
from page_cite_rag.embedding_native import NativeEmbeddingEncoder
from page_cite_rag.embedding_server import DEFAULT_GGUF_PATH, NativeEmbeddingServer
from page_cite_rag.generation import ContextBudgetError, ModelError, OutputValidationError
from page_cite_rag.gpu_processes import MAX_GPU_MEMORY_MIB
from page_cite_rag.gpu_retrieval import StagedGpuSearch
from page_cite_rag.hybrid import reciprocal_rank_fusion
from page_cite_rag.ingest import prepare_source
from page_cite_rag.llama import (
    DEFAULT_BUNDLE,
    LLAMA_COMMIT,
    LlamaConfig,
    LlamaModelClient,
    model_spec,
)
from page_cite_rag.models import DocumentMetadata, ParsedUnit
from page_cite_rag.qasper import Paper, Passage, ingest_paper
from page_cite_rag.reading_plan import (
    MAX_QUERIES_PER_REQUIREMENT,
    MAX_READING_REQUIREMENTS,
    ReadingPlan,
    reading_plan_messages,
)
from page_cite_rag.retrieval import (
    BM25Retriever,
    SearchContextGroup,
    SearchHit,
    SearchResult,
    build_bm25_index,
)
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL, MAX_CONTEXT_GROUPS
from page_cite_rag.store import SQLiteDocumentStore

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "outputs/reading-eval-20261020-v3"
DEFAULT_RUN_ROOT = FROZEN_ROOT / "executions/unified-reading-v1"
GPU_UUID = os.environ.get("PAGE_CITE_RAG_GPU_UUID", "")
GPU_TEMPERATURE_LIMIT_C = 88
CONTEXT_TOKENS = 16_384
EXACT_INPUT_LIMIT = 12_000
AGENT_FINAL_INPUT_LIMIT = MAX_EXACT_INPUT_TOKENS
OUTPUT_TOKENS = 3_072
TEMPLATE_RESERVE = 1_024
TOP_K = FORMAL_RETRIEVAL.final_k
FIXED_FOLLOWUP_BLOCKS = MAX_CONTEXT_GROUPS
CONTEXT_NEIGHBOR_BLOCKS = 0
PREP_DOCUMENT_EMBED_TOKEN_LIMIT = 1_000_000
PREP_DOCUMENT_EMBED_TOKEN_GLOBAL_LIMIT = 40_000_000
PREP_QUERY_EMBED_TOKEN_LIMIT = 25_000
PREP_QUERY_EMBED_TOKEN_GLOBAL_LIMIT = 100_000
MAX_INVOCATION_WALL_SECONDS = 50 * 60
MAX_CASE_WALL_SECONDS = 600
MAX_WORKFLOW_COMPARISON_CASES = 24
LONG_BENCH_MAX_INVOCATION_WALL_SECONDS = 90 * 60
LONG_BENCH_OFFICIAL_ARM = "longbench-v2-official-rag"
DEFAULT_ARMS = ("fixed-rag", "single-agent")
QUALITY_V1_ARMS = (
    "fixed-rag",
    "fixed-rag-followup-once",
    "decomposition-fixed-rag",
    "decomposition-agent",
)
REVIEW_ARMS = {
    "repair-once": "once",
    "reading-agent": "agent",
    "delegated-reading-agent": "delegated",
}
PROMPT_COMPARISON_ARMS = {
    "fixed-rag-prompt-task-aligned": "task-aligned-v5",
}
ARMS = (
    *DEFAULT_ARMS,
    "fixed-rag-followup-once",
    "single-agent-no-tools",
    "decomposition-fixed-rag",
    "decomposition-agent",
    *PROMPT_COMPARISON_ARMS,
    "fixed-rag-reranked",
    *REVIEW_ARMS,
    LONG_BENCH_OFFICIAL_ARM,
)
ARM_ARCHITECTURES = {
    "fixed-rag": "fixed_hybrid_retrieval_then_one_structured_answer",
    "fixed-rag-followup-once": (
        "shared_reranked_initial_context_then_one_deterministic_hybrid_pool_followup"
    ),
    "single-agent": "single_agent_navigation_then_shared_answer_writer",
    "decomposition-fixed-rag": "question_plan_then_multi_query_hybrid_rrf_then_shared_writer",
    "decomposition-agent": (
        "question_plan_multi_query_hybrid_rrf_then_bounded_agent_and_shared_writer"
    ),
    "single-agent-no-tools": "single_agent_structured_answer_over_frozen_initial_evidence",
    "delegated-agent": "legacy_bounded_navigation_then_separate_writer",
    "fixed-rag-prompt-task-aligned": "fixed_hybrid_retrieval_then_one_structured_answer",
    "fixed-rag-reranked": "fixed_hybrid_retrieval_then_one_structured_answer",
    "repair-once": "structured_review_then_at_most_one_answer_repair",
    "reading-agent": "bounded_reader_review_then_shared_answer_workflow",
    "delegated-reading-agent": "bounded_delegated_reader_review_then_shared_answer_workflow",
    LONG_BENCH_OFFICIAL_ARM: "official_longbench_prompt_over_fixed_retrieval_context",
}
RUNNER_VERSION = "frozen-reading-eval-v30-agent-control-followup-v1"
MAX_RERANKER_PAIR_TOKENS = 200_000_000
QUALITY_RERANKER_PAIR_LIMIT = 7_200
QUALITY_RERANKER_PAIR_TOKEN_LIMIT = 30_000_000
MAX_RERANKER_GRID_WALL_SECONDS = 60 * 60
SUMMARY_DATASETS = {
    "longbench-v1-qmsum": "en",
    "longbench-v1-vcsum": "zh",
}
_PASSAGE_ID = re.compile(r"/s(\d{4,})/p(\d{4,})$")
_LONG_BENCH_ZH_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃》「」『』【】〖〗〔〕〘〙〚〛〜〝〞〟〰〾〿"
    "–—‘’‛“”„‟…‧﹏."
)
LONG_BENCH_V2_OFFICIAL_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONG_BENCH_V2_RAG_PROMPT = (
    "Please read the following retrieved text chunks and answer the question below.\n\n"
    "<text>\n$DOC$\n</text>\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".\n'
)
LONG_BENCH_V2_RAG_PROMPT_SHA256 = "df1ed7a498962d37d1de051dfe063fde10875c530d94b33f71f9cc2c0ce2134e"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_protocol_configuration_hash(
    static: dict, expected_hash: str | None, *, required: bool
) -> None:
    if expected_hash is None:
        if required:
            raise ValueError("LongBench runs require the protocol-frozen static configuration hash")
        return
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError(
            "expected static configuration hash must be a lowercase SHA-256 hex digest"
        )
    if canonical_hash(static) != expected_hash:
        raise ValueError("run static configuration does not match the protocol-frozen hash")


def _answer_format_for(case: dict) -> str:
    """Bind each benchmark's scoring unit to the model's output contract."""
    if "task_format" in case:
        value = case["task_format"]
        if value not in ("general", "short_answer", "multiple_choice"):
            raise ValueError("frozen task_format must be general, short_answer or multiple_choice")
        return value
    formats = {
        "qasper": "short_answer",
        "longbench-v1-multifieldqa-zh": "short_answer",
        "longbench-v2-single-document-qa": "multiple_choice",
        "xrag-zh-mono-crosslingual": "short_answer",
        "derived-crosslingual": "short_answer",
        **{dataset: "general" for dataset in SUMMARY_DATASETS},
    }
    try:
        return formats[case["dataset"]]
    except (KeyError, TypeError):
        raise ValueError("unknown frozen dataset task") from None


def _prompt_revision_for_arm(arm: str) -> str:
    if arm == LONG_BENCH_OFFICIAL_ARM:
        return "longbench-v2-official-rag-prompt"
    return PROMPT_COMPARISON_ARMS.get(arm, "contract-consistent")


def _is_agent_arm(arm: str) -> bool:
    return arm in (
        "single-agent",
        "decomposition-agent",
        "single-agent-no-tools",
        "delegated-agent",
        "reading-agent",
        "delegated-reading-agent",
    )


def _agent_uses_live_retrieval(arm: str) -> bool:
    return _is_agent_arm(arm) and arm != "single-agent-no-tools"


def _is_fixed_rag_arm(arm: str) -> bool:
    return not _is_agent_arm(arm) and arm not in REVIEW_ARMS


def _longbench_v2_official_prompt(context: str, question: str, choices: dict[str, str]) -> str:
    if set(choices) != set("ABCD"):
        raise ValueError("LongBench-v2 choices must contain exactly A, B, C and D")
    values = {
        "$DOC$": context.strip(),
        "$Q$": question.strip(),
        **{f"$C_{letter}$": choices[letter].strip() for letter in "ABCD"},
    }
    prompt = LONG_BENCH_V2_RAG_PROMPT
    for placeholder, value in values.items():
        if not isinstance(value, str) or not value:
            raise ValueError("LongBench-v2 prompt fields must be nonempty text")
        prompt = prompt.replace(placeholder, value)
    if re.search(r"\$(?:DOC|Q|C_[A-D])\$", prompt):
        raise ValueError("LongBench-v2 prompt contains an unresolved placeholder")
    return prompt


def _longbench_v2_official_choice(response: str) -> str | None:
    """Match LongBench's pinned extract_answer implementation exactly."""
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def _common_admitted_block_ids(
    prompt_preflights: dict[str, dict], retrieved_block_ids: list[str]
) -> list[str]:
    """Lock prompt comparisons to evidence that every revision can admit."""
    if not prompt_preflights:
        raise ValueError("prompt comparison has no context preflight")
    admitted_sets = [set(value["admitted_block_ids"]) for value in prompt_preflights.values()]
    common = set.intersection(*admitted_sets)
    ordered = [block_id for block_id in retrieved_block_ids if block_id in common]
    if not ordered:
        raise ValueError("prompt revisions have no common admitted evidence")
    for value in prompt_preflights.values():
        if [block_id for block_id in value["admitted_block_ids"] if block_id in common] != ordered:
            raise ValueError("prompt revisions admit common evidence in different orders")
    return ordered


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


def append_jsonl_record(path: Path, row: dict) -> None:
    """Durably append one immutable private result row for resumable batches."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("refusing to append through a symlink")
    payload = (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_unique_jsonl(path: Path, key: str = "case_id") -> dict[str, dict]:
    """Load append-only progress while rejecting duplicate or partial records."""
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    values = {}
    for row in rows:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in values:
            raise ValueError(f"invalid or duplicate {key} in {path.name}")
        values[identity] = row
    return values


def reject_interrupted_attempts(
    attempts: dict[str, dict], completed: dict[str, dict], operation: str
) -> None:
    """Prevent silently retrying a case whose model-backed operation already started."""
    unfinished = sorted(attempts.keys() - completed.keys())
    if unfinished:
        raise ValueError(
            f"a {operation} request already started without a successful result; "
            f"preserve it and do not retry case {unfinished[0]}"
        )


def _duplicate_observer_call_indices(
    rows: list[dict],
    reservations: dict[int, dict],
    usage_by_index: dict[int, dict],
) -> set[int]:
    """Validate and return audited call reservations duplicated by an observer."""
    corrections = [row for row in rows if row.get("state") == "duplicate_observer_correction"]
    duplicate_indices = [row.get("duplicate_call_index") for row in corrections]
    actual_indices = [row.get("actual_call_index") for row in corrections]
    if (
        any(type(index) is not int for index in duplicate_indices + actual_indices)
        or len(duplicate_indices) != len(set(duplicate_indices))
        or len(actual_indices) != len(set(actual_indices))
        or set(duplicate_indices) & set(actual_indices)
    ):
        raise ModelError("global model-call ledger has invalid duplicate-observer corrections")
    for correction in corrections:
        duplicate_index = correction["duplicate_call_index"]
        actual_index = correction["actual_call_index"]
        if (
            duplicate_index not in reservations
            or actual_index not in reservations
            or duplicate_index not in usage_by_index
            or actual_index not in usage_by_index
        ):
            raise ModelError("duplicate-observer correction has no completed dispatch pair")
        duplicate = reservations[duplicate_index]
        actual = reservations[actual_index]
        duplicate_usage = usage_by_index[duplicate_index]
        actual_usage = usage_by_index[actual_index]
        if (
            duplicate.get("state") != "reserved_before_dispatch"
            or actual.get("state") != "reserved_before_dispatch"
            or any(
                duplicate.get(key) != actual.get(key) for key in ("arm", "case_id", "budget_phase")
            )
            or duplicate_usage.get("usage_unknown") is True
            or actual_usage.get("usage_unknown") is True
            or any(
                duplicate_usage.get(key) != actual_usage.get(key)
                for key in ("input_tokens", "output_tokens")
            )
        ):
            raise ModelError("duplicate-observer correction does not match its actual dispatch")
    return set(duplicate_indices)


def _pending_grid_cases(inputs, attempts, completed, failures):
    """Keep failed attempts as missing observations; never dispatch them again."""
    ids = {case["case_id"] for case in inputs}
    if (attempts.keys() | completed.keys() | failures.keys()) - ids:
        raise ValueError("reranker records contain unknown case IDs")
    if (failures.keys() | completed.keys()) - attempts.keys() or failures.keys() & completed.keys():
        raise ValueError("reranker failure records do not match attempts")
    reject_interrupted_attempts(attempts, completed | failures, "rewritten reranker")
    return [case for case in inputs if case["case_id"] not in attempts]


def _validate_stopped_grid_configuration(run_root: Path, current: dict) -> dict:
    """Keep the original configuration while recording a narrow continuation change."""
    frozen = json.loads((run_root / "configuration.json").read_text(encoding="utf-8"))["static"]
    saved_sources = frozen["source_code_sha256"]
    current_sources = current["source_code_sha256"]
    changed = {
        name
        for name in saved_sources.keys() | current_sources.keys()
        if saved_sources.get(name) != current_sources.get(name)
    }
    allowed = {
        "scripts/run_reading_eval.py",
        "src/page_cite_rag/cli.py",
        "src/page_cite_rag/service.py",
    }
    if not changed <= allowed:
        raise ValueError("retrieval implementation changed since the stopped grid")
    equivalent = json.loads(json.dumps(current))
    equivalent["source_code_sha256"] = saved_sources
    equivalent["answer"].pop("case_deadline_seconds", None)
    if equivalent != frozen:
        raise ValueError("configuration changed beyond the recorded answer deadline")
    archive = run_root / "source-at-revalidation.zip"
    with zipfile.ZipFile(archive) as snapshot:
        original = snapshot.read("scripts/run_reading_eval.py")
    if hashlib.sha256(original).hexdigest() != saved_sources["scripts/run_reading_eval.py"]:
        raise ValueError("original runner snapshot does not match the frozen grid")
    if not (run_root / "GPU-STOP.json").is_file():
        raise ValueError("no recorded GPU stop to continue")
    return frozen


def write_state_json(path: Path, value: object) -> None:
    """Atomically replace a private progress marker; result artifacts stay append-only."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("refusing to replace a symlinked progress file")
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as target:
        temporary = Path(target.name)
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _file_hashes(path: Path) -> dict[str, str]:
    if path.is_file():
        return {path.name: sha256(path)}
    return {
        str(item.relative_to(path)): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }


def _frozen_case_ids(manifest: dict) -> list[str]:
    case_ids = [case["case_id"] for case in manifest["cases"]]
    if (
        not case_ids
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise ValueError("frozen manifest case IDs must be nonempty strings and unique")
    return case_ids


def _manifest_path(root: Path) -> Path:
    path = root / "manifest.json"
    return path if path.is_file() else root.parent / "manifest.json"


def load_frozen(root: Path) -> tuple[dict, list[dict]]:
    manifest_path = _manifest_path(root)
    inputs_path = root / "inputs.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path.parent != root:
        partition = manifest.get("partitions", {}).get(root.name)
        if not isinstance(partition, dict):
            raise ValueError("master manifest has no matching frozen partition")
        manifest = {**manifest, **partition, "partition": root.name}
    expected = manifest["files"]["inputs.jsonl"]
    if inputs_path.is_symlink() or inputs_path.stat().st_size != expected["bytes"]:
        raise ValueError("frozen file identity differs: inputs.jsonl")
    if sha256(inputs_path) != expected["sha256"]:
        raise ValueError("frozen file checksum differs: inputs.jsonl")
    inputs = read_jsonl(inputs_path)
    expected_ids = _frozen_case_ids(manifest)
    if len(inputs) != len(expected_ids) or [row.get("case_id") for row in inputs] != expected_ids:
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
    expected_ids = _frozen_case_ids(manifest)
    if len(rows) != len(expected_ids) or [row.get("case_id") for row in rows] != expected_ids:
        raise ValueError("frozen gold rows do not match the manifest")
    return rows


def _case_document(
    store: SQLiteDocumentStore, case: dict, *, chunker=None
) -> tuple[str, dict[str, str]]:
    source = case["source"]
    if source.get("format") == "file":
        relative_path = source.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError("frozen PDF source path is invalid")
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("frozen PDF source path must be project-relative")
        path = ROOT / path
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise ValueError("frozen PDF source is missing or not a regular project file")
        if sha256(path) != source.get("sha256"):
            raise ValueError("frozen PDF source checksum differs")
        prepared = prepare_source(
            path,
            language=case.get("source_language"),
            usage_scope="private public-document evaluation",
            chunker=chunker,
        )
        store.ingest(
            prepared.document,
            prepared.iter_units(),
            block_factory=prepared.block_factory,
        )
        return prepared.document.document_id, {}

    if source.get("format") == "qasper_paragraphs":
        passages = []
        section_count = 0
        section_names = source.get("section_names")
        if section_names is not None and (
            not isinstance(section_names, list)
            or not all(isinstance(name, str) or name is None for name in section_names)
        ):
            raise ValueError("frozen QASPER section names are invalid")
        for value in source["paragraphs"]:
            match = _PASSAGE_ID.search(value["passage_id"])
            if match is None:
                raise ValueError("frozen QASPER passage ID has an unsupported format")
            section_index, paragraph_index = map(int, match.groups())
            frozen_section_index = value.get("section_index", section_index)
            if type(frozen_section_index) is not int or frozen_section_index != section_index:
                raise ValueError("frozen QASPER section index differs from its passage ID")
            if not isinstance(value.get("text"), str):
                raise ValueError("frozen QASPER passage text is invalid")
            section_count = max(section_count, section_index + 1)
            passages.append(
                Passage(value["passage_id"], section_index, paragraph_index, value["text"])
            )
        if not passages:
            raise ValueError("QASPER document has no nonempty source paragraphs")
        if section_names is not None and len(section_names) < section_count:
            raise ValueError("frozen QASPER section names do not cover all passages")
        names = (
            tuple(section_names)
            if section_names is not None
            else tuple(None for _ in range(section_count))
        )
        paper = Paper(
            case["source_id"],
            tuple(passages),
            (),
            names,
        )
        return ingest_paper(store, paper, chunker=chunker)

    if source.get("format") not in {"plain_text", "xrag_article_bundle"} or not isinstance(
        source.get("text"), str
    ):
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
        source_name=f"benchmark-{source_hash[:16]}.txt",
        media_type="text/plain",
        parser_name="plain_text",
        parser_version=version,
        # The same source may have several translated query variants. Its
        # document metadata must describe the source, not the query/answer arm.
        language=case.get("source_language", case["language"]),
        usage_scope="public benchmark evaluation",
    )
    if chunker is not None:
        document = chunker.bind_document(document)
    options = {"block_factory": chunker} if chunker is not None else {}
    store.ingest(document, (ParsedUnit(unit_index=1, text=text, status="ok"),), **options)
    return document.document_id, {}


def _xrag_block_article_map(
    store: SQLiteDocumentStore, document_id: str, source: dict
) -> dict[str, str]:
    """Map retrieval blocks to article identity using frozen source character offsets."""
    if source.get("format") != "xrag_article_bundle":
        return {}
    ranges = source.get("article_ranges")
    if not isinstance(ranges, list) or len(ranges) != 8:
        raise ValueError("frozen XRAG source lacks eight article ranges")
    mapping = {}
    for block in store.get_blocks(document_id):
        overlapping = [
            item
            for item in ranges
            if item["char_start"] < block.char_end and block.char_start < item["char_end"]
        ]
        if len(overlapping) > 1:
            raise ValueError("an indexed block crosses XRAG article boundaries")
        if overlapping:
            item = overlapping[0]
            if block.char_start < item["char_start"] or block.char_end > item["char_end"]:
                raise ValueError("an indexed block is not contained by one XRAG article")
            mapping[block.block_id] = item["article_id"]
    if not mapping:
        raise ValueError("no indexed text blocks map to XRAG articles")
    return mapping


class _CountingTransport:
    def __init__(
        self,
        transport,
        *,
        document_token_limit: int | None = None,
        query_token_limit: int | None = None,
    ) -> None:
        document_token_limit = (
            PREP_DOCUMENT_EMBED_TOKEN_LIMIT
            if document_token_limit is None
            else document_token_limit
        )
        query_token_limit = (
            PREP_QUERY_EMBED_TOKEN_LIMIT if query_token_limit is None else query_token_limit
        )
        if (
            type(document_token_limit) is not int
            or not 1 <= document_token_limit <= PREP_DOCUMENT_EMBED_TOKEN_GLOBAL_LIMIT
            or type(query_token_limit) is not int
            or query_token_limit < 1
        ):
            raise ValueError("invalid embedding token budget")
        self._transport = transport
        self._document_token_limit = document_token_limit
        self._query_token_limit = query_token_limit
        self.query_prefix: str | None = None
        self.document_tokens = 0
        self.query_tokens = 0
        self.embed_inputs = 0
        self.requests_attempted = 0
        self.requests_succeeded = 0
        self.last_request: dict | None = None

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
        if self.document_tokens + document_count > self._document_token_limit:
            raise ValueError("document embedding input token budget exceeded before dispatch")
        if self.query_tokens + query_count > self._query_token_limit:
            raise ValueError("query embedding input token budget exceeded before dispatch")
        self.document_tokens += document_count
        self.query_tokens += query_count
        self.embed_inputs += len(texts)
        self.requests_attempted += 1
        self.last_request = {
            "input_count": len(texts),
            "document_input_tokens": document_count,
            "query_input_tokens": query_count,
            "state": "dispatched",
        }
        try:
            result = self._transport.embed(texts)
        except BaseException as exc:
            self.last_request["state"] = "failed_or_interrupted"
            self.last_request["error_type"] = type(exc).__name__
            raise
        self.requests_succeeded += 1
        self.last_request["state"] = "succeeded"
        return result


def _embedding_request_accounting(transport: _CountingTransport | None) -> dict | None:
    if transport is None:
        return None
    return {
        "requests_attempted": transport.requests_attempted,
        "requests_succeeded": transport.requests_succeeded,
        "requests_failed_or_interrupted": (
            transport.requests_attempted - transport.requests_succeeded
        ),
        "input_count_attempted": transport.embed_inputs,
        "document_input_tokens_attempted": transport.document_tokens,
        "query_input_tokens_attempted": transport.query_tokens,
        "last_request": transport.last_request,
    }


def _record_preparation_failure(
    run_root: Path,
    static: dict,
    progress: dict,
    exc: BaseException,
    server: NativeEmbeddingServer | None,
    transport: _CountingTransport | None,
) -> None:
    preflight_path = run_root / "prepared" / "embedding-preflight.json"
    try:
        gpu_trace = server.diagnostics() if server is not None else None
    except Exception as diagnostic_error:
        gpu_trace = {"diagnostic_capture_error": type(diagnostic_error).__name__}
    if gpu_trace is None and isinstance(getattr(exc, "sample", None), dict):
        gpu_trace = {
            "guard_error": getattr(exc, "category", "gpu_guard"),
            "last_gpu_sample": exc.sample,
        }
    record = {
        "state": "preparation_failed",
        "static_configuration_sha256": canonical_hash(static),
        "failure_stage": progress.get("stage"),
        "document_ordinal": progress.get("document_ordinal"),
        "document_count": progress.get("document_count"),
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
        "embedding_request_accounting": _embedding_request_accounting(transport),
        "gpu_trace": gpu_trace,
        "embedding_preflight_sha256": sha256(preflight_path) if preflight_path.is_file() else None,
    }
    try:
        atomic_json_once(run_root / "PREPARATION-FAILURE.json", record)
    except ValueError:
        # Preserve the first failure artifact for this run root.
        pass


@contextmanager
def _preparation_embedding_session(run_root: Path, static: dict, progress: dict):
    server = None
    transport = None
    try:
        server = NativeEmbeddingServer(
            DEFAULT_GGUF_PATH,
            GPU_UUID,
            max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C,
            request_timeout_seconds=static["embedding_input"]["request_timeout_seconds"],
        )
        with server:
            transport = _CountingTransport(
                server,
                document_token_limit=static["preparation_budget"][
                    "document_embedding_input_tokens_max"
                ],
                query_token_limit=static["preparation_budget"]["query_embedding_input_tokens_max"],
            )
            yield server, transport
    except BaseException as exc:
        _record_preparation_failure(run_root, static, progress, exc, server, transport)
        raise


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
    sampling_profile: str = "greedy",
    document_embedding_token_limit: int = PREP_DOCUMENT_EMBED_TOKEN_LIMIT,
    query_embedding_token_limit: int = PREP_QUERY_EMBED_TOKEN_LIMIT,
    generator_model_size: str = "9b",
    retrieval_grid: bool = False,
    shared_initial_reranker: bool = False,
) -> dict:
    from page_cite_rag.embedding_native import GGUF_REVISION, GGUF_SHA256
    from page_cite_rag.full_answer import (
        PROMPT_VERSION,
        TASK_ALIGNED_PROMPT_VERSION,
    )
    from page_cite_rag.reranker_server import MAX_RERANK_DOCUMENTS_PER_REQUEST
    from page_cite_rag.reranker_server import MODEL_REVISION as RERANK_REVISION
    from page_cite_rag.reranker_server import MODEL_SHA256 as RERANK_SHA256
    from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL, MAX_CONTEXT_GROUPS
    from page_cite_rag.text import TokenBlockFactory

    if type(retrieval_grid) is not bool:
        raise ValueError("retrieval_grid must be a boolean")
    if type(shared_initial_reranker) is not bool:
        raise ValueError("shared_initial_reranker must be a boolean")
    arms = (
        ["fixed-rag", "fixed-rag-reranked"]
        if retrieval_grid
        else selected_arms or list(DEFAULT_ARMS)
    )
    is_longbench_official = LONG_BENCH_OFFICIAL_ARM in arms
    if is_longbench_official and arms != [LONG_BENCH_OFFICIAL_ARM]:
        raise ValueError("the official LongBench prompt protocol must run in its own frozen batch")
    if sampling_profile not in ("greedy", "qwen-general"):
        raise ValueError("unsupported sampling profile")
    if generator_model_size not in ("9b", "4b"):
        raise ValueError("unsupported pinned generator model size")
    if is_longbench_official and generator_model_size != "9b":
        raise ValueError("the frozen LongBench official-adapter arm uses the pinned 9B model")
    if (
        type(document_embedding_token_limit) is not int
        or not 1 <= document_embedding_token_limit <= PREP_DOCUMENT_EMBED_TOKEN_GLOBAL_LIMIT
    ):
        raise ValueError("document embedding token limit exceeds the approved global ceiling")
    if (
        type(query_embedding_token_limit) is not int
        or not 1 <= query_embedding_token_limit <= PREP_QUERY_EMBED_TOKEN_GLOBAL_LIMIT
    ):
        raise ValueError("query embedding token limit exceeds the approved global ceiling")
    case_ids = selected_case_ids or [case["case_id"] for case in manifest["cases"]]
    workflow_call_limits = {
        "fixed-rag": 1,
        "fixed-rag-reranked": 1,
        "fixed-rag-followup-once": 2,
        "single-agent": None,
        "decomposition-fixed-rag": 1,
        "decomposition-agent": None,
    }
    reading_plan_comparison = tuple(arms) == QUALITY_V1_ARMS
    if shared_initial_reranker and not reading_plan_comparison:
        raise ValueError("shared initial reranking is reserved for the four reading-plan arms")
    selected_workflow_arms = {arm for arm in arms if arm in workflow_call_limits}
    answer_call_budget = None
    reading_plan_budget = None
    if reading_plan_comparison:
        if len(case_ids) > 72:
            raise ValueError("reading-plan comparison exceeds its frozen 72-case ceiling")
        answer_call_budget = {
            "cases": len(case_ids),
            "per_case_model_call_limit_by_arm": {
                arm: workflow_call_limits[arm] for arm in QUALITY_V1_ARMS
            },
            "single_agent_per_case_call_limit": None,
            "phase_generation_dispatch_limit": 720,
            "case_deadline_seconds": MAX_CASE_WALL_SECONDS,
        }
        reading_plan_budget = {
            "planner_calls_max": len(case_ids),
            "planner_calls_limit": 72,
            "planner_phase_generation_dispatch_limit": 72,
            "model_calls_per_agent_case_limit": None,
            "case_deadline_seconds": MAX_CASE_WALL_SECONDS,
        }
    elif "single-agent" in selected_workflow_arms and len(selected_workflow_arms) > 1:
        if len(case_ids) > MAX_WORKFLOW_COMPARISON_CASES:
            raise ValueError("workflow comparison exceeds its frozen 24-case development ceiling")
        answer_call_budget = {
            "cases": len(case_ids),
            "per_case_model_call_limit_by_arm": {
                arm: workflow_call_limits[arm] for arm in arms if arm in selected_workflow_arms
            },
            "single_agent_per_case_call_limit": None,
            "case_deadline_seconds": MAX_CASE_WALL_SECONDS,
            "batch_generation_dispatch_limit": "frozen_global_and_phase_ledgers",
        }
    effective_sampling_profile = "longbench-official" if is_longbench_official else sampling_profile
    generator_spec = model_spec(generator_model_size)
    auxiliary_metrics = _frozen_auxiliary_metrics(manifest)
    return {
        **({"auxiliary_metrics": auxiliary_metrics} if auxiliary_metrics else {}),
        "runner_version": RUNNER_VERSION,
        "frozen_inputs_sha256": manifest["files"]["inputs.jsonl"]["sha256"],
        "run_selection": {
            "case_ids": case_ids,
            "arms": arms,
        },
        "reading_plan": (
            {
                "protocol": "question-needs-and-bilingual-query-plan/v2",
                "prompt_revision": "neutral-needs-source-language-query/v1",
                "schema_sha256": canonical_hash(ReadingPlan.model_json_schema()),
                "num_predict": 1024,
                "max_requirements": MAX_READING_REQUIREMENTS,
                "max_queries_per_requirement": MAX_QUERIES_PER_REQUIREMENT,
                "retrieval_candidates_per_branch": FORMAL_RETRIEVAL.candidates_per_branch,
                "retrieval_candidate_count_per_query": 2 * FORMAL_RETRIEVAL.candidates_per_branch,
                "fused_candidate_count": MAX_CONTEXT_GROUPS,
                "context_group_limit": MAX_CONTEXT_GROUPS,
                "context_groups_after_fusion": MAX_CONTEXT_GROUPS,
                "planned_query_input_tokens_max": query_embedding_token_limit,
                "planner_budget": reading_plan_budget,
            }
            if reading_plan_comparison
            else None
        ),
        "reading_plan_call_budget": reading_plan_budget,
        "source_code_sha256": _source_code_hashes(),
        "models": {
            "generator": {
                "name": generator_spec["file"].removesuffix(".gguf"),
                "size": generator_model_size,
                "revision": generator_spec["revision"],
                "file": generator_spec["file"],
                "sha256": generator_spec["sha256"],
                "llama_commit": LLAMA_COMMIT,
            },
            "embedding": {
                "name": "Qwen3-Embedding-0.6B-f16-GGUF",
                "revision": GGUF_REVISION,
                "sha256": GGUF_SHA256,
            },
            "reranker": (
                {
                    "name": "Qwen3-Reranker-0.6B-Q8_0",
                    "revision": RERANK_REVISION,
                    "sha256": RERANK_SHA256,
                }
                if "fixed-rag-reranked" in arms or shared_initial_reranker
                else None
            ),
        },
        "embedding_input": {
            "document_max_tokens": 512,
            "query_max_tokens": 1024 if is_longbench_official else 512,
            "request_timeout_seconds": 60 if is_longbench_official else 15,
            "long_query_policy": "keep complete query; no silent truncation",
        },
        "retrieval": {
            "method": "BM25S + Qwen dense + reciprocal-rank fusion",
            **FORMAL_RETRIEVAL.as_dict(),
            "context_neighbor_blocks": CONTEXT_NEIGHBOR_BLOCKS,
            "query_rewrite": (
                {
                    "version": "full-sentence-document-language/v1",
                    "prompt_id": "retrieval.full-sentence.v1",
                    "mode": "one bounded full-sentence search query in source language",
                    "max_calls": 240,
                    "num_predict": 128,
                    "sampling_profile": effective_sampling_profile,
                }
                if retrieval_grid
                else False
            ),
            "reranker": "fixed-rag-reranked" in arms,
            "shared_initial_reranker": shared_initial_reranker,
            "reranker_scope": (
                "one rerank per shared initial evidence pool; Agent follow-up search stays hybrid"
                if shared_initial_reranker
                else None
            ),
            "reranker_request": (
                {
                    "documents_per_request": MAX_RERANK_DOCUMENTS_PER_REQUEST,
                    "pair_max_tokens": 4096,
                    "request_timeout_seconds": 60,
                }
                if "fixed-rag-reranked" in arms or shared_initial_reranker
                else None
            ),
            "retrieval_grid": (
                {
                    "protocol": "full-sentence-rewrite-routing/v1",
                    "variants": [
                        "original_hybrid",
                        "rewrite_bm25_only",
                        "rewrite_dense_only",
                        "rewrite_both",
                    ],
                    "candidate_per_branch": FORMAL_RETRIEVAL.candidates_per_branch,
                    "rrf_constant": FORMAL_RETRIEVAL.rrf_constant,
                    "context_groups": TOP_K,
                    "candidate_pool": "top-50-per-branch-union",
                }
                if retrieval_grid
                else None
            ),
            # Source chunking is part of the common retrieval corpus, not a
            # side effect of choosing the optional reranker arm.
            "token_chunking": True,
            "text_chunking": TokenBlockFactory(DEFAULT_MODEL_DIR).identity,
        },
        "answer": {
            "default_contract": PROMPT_VERSION,
            "prompt_revisions_by_arm": {arm: _prompt_revision_for_arm(arm) for arm in arms},
            "contracts_by_arm": {
                arm: (
                    TASK_ALIGNED_PROMPT_VERSION
                    if _prompt_revision_for_arm(arm) == "task-aligned-v5"
                    else PROMPT_VERSION
                )
                for arm in arms
            },
            "task_adapter": "benchmark-answer-format/v1",
            "task_formats": {
                "qasper": "short_answer",
                "longbench-v1-multifieldqa-zh": "short_answer",
                "longbench-v2-single-document-qa": "multiple_choice",
                "xrag-zh-mono-crosslingual": "short_answer",
                **{dataset: "general" for dataset in SUMMARY_DATASETS},
            },
            "task_format_override": "validated frozen case task_format, when supplied",
            "summary_scoring_text": "answer plus detail_claim texts; no citations or limitations",
            "language": "per frozen case",
            "thinking": False,
            "sampling_profile": effective_sampling_profile,
            "seed": 42,
            "num_ctx": CONTEXT_TOKENS,
            "max_exact_input_tokens": EXACT_INPUT_LIMIT,
            "num_predict": 128 if is_longbench_official else OUTPUT_TOKENS,
            "prompt_template": (
                {
                    "source": "THUDM/LongBench",
                    "revision": LONG_BENCH_V2_OFFICIAL_REVISION,
                    "path": "prompts/0shot_rag.txt",
                    "sha256": LONG_BENCH_V2_RAG_PROMPT_SHA256,
                    "protocol_scope": (
                        "official prompt/extractor adapted to local retrieval; not official "
                        "LongBench-v2 leaderboard RAG execution"
                    ),
                    "retrieval_context_top_k": TOP_K,
                    "context_order": "original source block ordinal (LongBench c_idx order)",
                    "context_admission": (
                        "complete blocks; drop lowest-ranked blocks until exact fit"
                    ),
                    "temperature": 0.1,
                    "retry": False,
                }
                if is_longbench_official
                else None
            ),
            "template_reserve": TEMPLATE_RESERVE,
            "timeout_seconds": 600,
            "case_deadline_seconds": None if is_longbench_official else MAX_CASE_WALL_SECONDS,
            "retry": False,
        },
        "agent": {
            "framework": "LangChain create_agent",
            "architecture_by_arm": {
                arm: ARM_ARCHITECTURES[arm] for arm in arms if arm in ARM_ARCHITECTURES
            },
            "resource_profiles_by_arm": {
                "fixed-rag": {
                    "model_calls_per_case_max": 1,
                    "data_tool_events_per_case_max": 0,
                    "total_input_tokens_per_case_max": EXACT_INPUT_LIMIT,
                    "total_output_tokens_per_case_max": OUTPUT_TOKENS,
                },
                "fixed-rag-reranked": {
                    "model_calls_per_case_max": 1,
                    "data_tool_events_per_case_max": 0,
                    "total_input_tokens_per_case_max": EXACT_INPUT_LIMIT,
                    "total_output_tokens_per_case_max": OUTPUT_TOKENS,
                    "initial_evidence": "shared_hybrid_then_rerank",
                },
                "fixed-rag-followup-once": {
                    "model_calls_per_case_max": 2,
                    "data_tool_events_per_case_max": 0,
                    "followup_trigger": ["partial", "insufficient_evidence"],
                    "followup_candidate_block_limit": FIXED_FOLLOWUP_BLOCKS,
                    "total_input_tokens_per_call_max": EXACT_INPUT_LIMIT,
                    "total_output_tokens_per_case_max": OUTPUT_TOKENS * 2,
                },
                "single-agent": {
                    "model_call_limit_per_case": None,
                    "common_writer_calls_per_case": 1,
                    "data_tool_event_limit_per_case": None,
                    "call_input_tokens_max": MAX_INTEGRATED_AGENT_CALL_INPUT_TOKENS,
                    "total_input_tokens_per_case_limit": None,
                    "call_output_tokens_max": MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
                    "total_output_tokens_per_case_limit": None,
                    "final_input_tokens_max": AGENT_FINAL_INPUT_LIMIT,
                    "final_output_tokens_max": MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
                    "case_deadline_seconds": MAX_CASE_WALL_SECONDS,
                    "budget_scope": "single_agent_navigation_and_common_writer_shared_ledger",
                },
                "decomposition-fixed-rag": {
                    "model_calls_per_case_max": 1,
                    "data_tool_events_per_case_max": 0,
                    "planner_is_shared_and_reported_separately": True,
                    "total_input_tokens_per_call_max": EXACT_INPUT_LIMIT,
                    "total_output_tokens_per_case_max": OUTPUT_TOKENS,
                },
                "decomposition-agent": {
                    "model_call_limit_per_case": None,
                    "common_writer_calls_per_case": 1,
                    "planner_is_shared_and_reported_separately": True,
                    "data_tool_event_limit_per_case": None,
                    "call_input_tokens_max": MAX_INTEGRATED_AGENT_CALL_INPUT_TOKENS,
                    "total_input_tokens_per_case_limit": None,
                    "call_output_tokens_max": MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
                    "total_output_tokens_per_case_limit": None,
                    "final_input_tokens_max": AGENT_FINAL_INPUT_LIMIT,
                    "final_output_tokens_max": MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
                    "case_deadline_seconds": MAX_CASE_WALL_SECONDS,
                    "budget_scope": "navigation_and_shared_writer; one shared planner call",
                },
                "single-agent-no-tools": {
                    "model_calls_per_case_max": 1,
                    "data_tool_events_per_case_max": 0,
                    "total_input_tokens_per_case_max": MAX_INTEGRATED_AGENT_TOTAL_INPUT_TOKENS,
                    "call_output_tokens_max": MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
                    "total_output_tokens_per_case_max": MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
                    "budget_scope": "one_structured_answer_over_frozen_initial_evidence",
                },
                "delegated-agent": {
                    "navigation_model_calls_per_case_max": MAX_AGENT_MODEL_CALLS,
                    "separate_final_writer_calls_per_case_max": 1,
                    "total_model_calls_per_case_max": MAX_AGENT_MODEL_CALLS + 1,
                    "navigation_input_tokens_max": MAX_AGENT_TOTAL_INPUT_TOKENS,
                    "final_writer_input_tokens_max": AGENT_FINAL_INPUT_LIMIT,
                    "total_input_tokens_per_case_max": MAX_AGENT_TOTAL_INPUT_TOKENS
                    + AGENT_FINAL_INPUT_LIMIT,
                    "navigation_output_tokens_max": MAX_AGENT_TOTAL_OUTPUT_TOKENS,
                    "final_writer_output_tokens_max": OUTPUT_TOKENS,
                    "total_output_tokens_per_case_max": MAX_AGENT_TOTAL_OUTPUT_TOKENS
                    + OUTPUT_TOKENS,
                    "data_tool_events_per_case_max": MAX_AGENT_TOOL_CALLS,
                    "readers_per_case_max": MAX_READER_TASKS,
                    "budget_scope": "legacy_navigation_and_readers_then_separate_writer",
                },
            },
            "sequential": True,
            "shared_budget_by_arm": {
                "single-agent": True,
                "single-agent-no-tools": True,
                "delegated-agent": False,
            },
            "legacy_delegated_arm_is_default": False,
        },
        "answer_call_budget": answer_call_budget,
        "source_review": {
            "implementation": "page_cite_rag.review.answer_with_review",
            "checker_is_quality_evaluator": False,
            "strategies_by_arm": REVIEW_ARMS,
            "structured_call_limit_per_case": 4,
            "navigation_call_limit_per_case": 4,
            "total_call_limits_by_arm": {
                "repair-once": 4,
                "reading-agent": 8,
                "delegated-reading-agent": 8,
            },
            "max_repair_cycles": 1,
            "generation_records": "actual messages, schema, raw output and usage for every stage",
        },
        "gpu": {
            "uuid": GPU_UUID,
            "memory_limit_mib": MAX_GPU_MEMORY_MIB,
            "temperature_limit_c": GPU_TEMPERATURE_LIMIT_C,
            "headroom_below_driver_max_operating_c": 5,
            "thermal_slowdown_must_be_inactive": True,
            "max_simultaneous_sequences": 1,
        },
        "request_scheduler": {
            "minimum_wait_seconds": 3,
            "maximum_wait_seconds": 120,
            "dispatch_temperature": "min(driver_target - 15C, approved_limit - 3C)",
            "thermal_guards": "unchanged; active HW/SW slowdown stops the server",
            "timing": "waiting included in case wall time, excluded from inference stats",
        },
        "preparation_budget": {
            "document_embedding_input_tokens_max": document_embedding_token_limit,
            "query_embedding_input_tokens_max": query_embedding_token_limit,
            "global_document_embedding_input_tokens_max": PREP_DOCUMENT_EMBED_TOKEN_GLOBAL_LIMIT,
            "global_query_embedding_input_tokens_max": PREP_QUERY_EMBED_TOKEN_GLOBAL_LIMIT,
            "reranker_pair_tokens_max": (
                QUALITY_RERANKER_PAIR_TOKEN_LIMIT
                if shared_initial_reranker
                else MAX_RERANKER_PAIR_TOKENS
            ),
            "reranker_pairs_max": QUALITY_RERANKER_PAIR_LIMIT if shared_initial_reranker else None,
        },
        "dataset_manifest_sha256": sha256(_manifest_path(root)),
        "frozen_partition": manifest.get("partition"),
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


def _unique_document_passages(
    documents: dict[str, tuple[str, dict[str, str]]],
) -> dict[str, dict[str, str]]:
    """Count and index content-addressed documents once even when cases repeat them."""
    unique: dict[str, dict[str, str]] = {}
    for document_id, passage_map in documents.values():
        unique.setdefault(document_id, passage_map)
    return unique


def _copy_preparation_checkpoint(run_root: Path, source_root: Path, static: dict) -> dict:
    """Copy verified source/index artifacts without changing the previous run."""
    source_root = source_root.resolve()
    if run_root.resolve().is_relative_to(source_root):
        raise ValueError("checkpoint destination must be outside the source run")
    configuration_path = source_root / "configuration.json"
    previous = json.loads(configuration_path.read_text(encoding="utf-8"))["static"]
    source_prepared = source_root / "prepared"
    source_preflight_path = source_prepared / "embedding-preflight.json"
    if not source_preflight_path.is_file() or source_preflight_path.is_symlink():
        raise ValueError("checkpoint is missing its verified embedding preflight")
    source_preflight = json.loads(source_preflight_path.read_text(encoding="utf-8"))
    actual_document_tokens = source_preflight.get(
        "new_document_embedding_input_tokens_expected",
        source_preflight.get("document_embedding_input_tokens"),
    )
    actual_query_tokens = source_preflight.get("query_embedding_input_tokens")
    embedding_limits = static.get("preparation_budget", {})
    if (
        type(actual_document_tokens) is not int
        or type(actual_query_tokens) is not int
        or actual_document_tokens > embedding_limits.get("document_embedding_input_tokens_max", -1)
        or actual_query_tokens > embedding_limits.get("query_embedding_input_tokens_max", -1)
    ):
        raise ValueError("checkpoint embedding usage exceeds the active preparation budget")

    def checkpoint_identity(value: dict) -> dict:
        identity = json.loads(json.dumps(value))
        identity.pop("source_code_sha256", None)
        identity.pop("runner_version", None)
        # The same byte-identical frozen inputs may carry scoring-only metadata
        # (for example an auxiliary metric or source-review rubric). Those fields
        # must be frozen into the new run, but they do not invalidate retrieval.
        identity.pop("dataset_manifest_sha256", None)
        identity.pop("auxiliary_metrics", None)
        # Answer prompts, Agent budgets/roles, and the selected generation arms
        # do not change the already-built document or retrieval artifacts. The
        # selected case IDs and effective retrieval/model identities stay bound.
        identity.pop("answer", None)
        identity.pop("agent", None)
        identity.pop("answer_call_budget", None)
        identity.pop("reading_plan", None)
        identity.pop("reading_plan_call_budget", None)
        identity.pop("source_review", None)
        selection = identity.get("run_selection")
        if isinstance(selection, dict):
            selection.pop("arms", None)
        retrieval = identity.get("retrieval", {})
        # Reranking is downstream of the frozen BM25+dense candidate pool.
        # The checkpoint copies only that base pool, so enabling/disabling the
        # reranker must not force document parsing or embedding to run again.
        retrieval.pop("reranker", None)
        for key in (
            "query_rewrite",
            "retrieval_grid",
            "reranker_request",
            "shared_initial_reranker",
            "reranker_scope",
        ):
            retrieval.pop(key, None)
        identity.get("models", {}).pop("reranker", None)
        preparation_budget = identity.get("preparation_budget", {})
        for key in (
            "document_embedding_input_tokens_max",
            "query_embedding_input_tokens_max",
            "reranker_pair_tokens_max",
            "reranker_pairs_max",
        ):
            preparation_budget.pop(key, None)
        return identity

    if checkpoint_identity(previous) != checkpoint_identity(static):
        raise ValueError("checkpoint data, model or effective configuration changed")
    old_code = previous["source_code_sha256"]
    new_code = static["source_code_sha256"]
    changed = {p for p in old_code.keys() | new_code.keys() if old_code.get(p) != new_code.get(p)}
    index_contract_files = {
        "src/page_cite_rag/dense.py",
        "src/page_cite_rag/embedding_native.py",
        "src/page_cite_rag/models.py",
        "src/page_cite_rag/qasper.py",
        "src/page_cite_rag/retrieval.py",
        "src/page_cite_rag/store.py",
        "src/page_cite_rag/text.py",
        "src/page_cite_rag/tokenize.py",
    }
    if changed & index_contract_files:
        raise ValueError("checkpoint index or source-block implementation changed")
    safe_experiment_changes = {
        # These changes affect frozen-case selection metadata and query/result
        # candidate handling only. The checkpoint identity independently binds
        # the exact input hashes and unchanged index/source-block contract.
        "scripts/prepare_reading_eval.py",
        "scripts/run_reading_eval.py",
        "src/page_cite_rag/agent.py",
        "src/page_cite_rag/api.py",
        "src/page_cite_rag/application.py",
        "src/page_cite_rag/cli.py",
        "src/page_cite_rag/docling_adapter.py",
        "src/page_cite_rag/docling_document.py",
        "src/page_cite_rag/docling_source.py",
        "src/page_cite_rag/embedding_server.py",
        "src/page_cite_rag/full_answer.py",
        "src/page_cite_rag/gpu_processes.py",
        "src/page_cite_rag/gpu_retrieval.py",
        "src/page_cite_rag/hybrid.py",
        "src/page_cite_rag/llama.py",
        "src/page_cite_rag/query.py",
        "src/page_cite_rag/reading_plan.py",
        "src/page_cite_rag/review.py",
        "src/page_cite_rag/retrieval_profile.py",
        "src/page_cite_rag/service.py",
        "src/page_cite_rag/workflows.py",
    }
    if changed - safe_experiment_changes:
        raise ValueError("checkpoint source code changed outside the reviewed experiment surface")
    source_db = source_root / "prepared/reading.sqlite3"
    source_ready_path = source_prepared / "prepared.json"
    source_retrieval_path = source_prepared / "retrieval.jsonl"
    target_db = run_root / "prepared/reading.sqlite3"
    if (
        source_db.is_symlink()
        or not source_ready_path.is_file()
        or not source_retrieval_path.is_file()
        or target_db.exists()
    ):
        raise ValueError("checkpoint database must be regular and destination must be new")
    source_indexes = source_db.with_name(source_db.name + ".indexes")
    target_indexes = target_db.with_name(target_db.name + ".indexes")
    source_ready = json.loads(source_ready_path.read_text(encoding="utf-8"))
    if (
        source_ready.get("retrieval_sha256") != sha256(source_retrieval_path)
        or source_ready.get("database_files_sha256") != _file_hashes(source_db)
        or source_ready.get("index_files_sha256") != _file_hashes(source_indexes)
    ):
        raise ValueError("checkpoint preparation artifacts failed checksum verification")
    source_retrieval = read_jsonl(source_retrieval_path)
    required_retrieval_key = "candidate_result" if static["retrieval"].get("reranker") else "result"
    if [row.get("case_id") for row in source_retrieval] != static["run_selection"][
        "case_ids"
    ] or any(not isinstance(row.get(required_retrieval_key), dict) for row in source_retrieval):
        raise ValueError("checkpoint does not contain the required frozen retrieval results")
    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as source:
        if source.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("checkpoint database failed integrity verification")
        index_ids = set()
        for kind, raw in source.execute("SELECT kind, manifest FROM retrieval_indexes"):
            if kind == "bm25" or kind.startswith("dense:"):
                index_id = json.loads(raw)["index_id"]
                if not isinstance(index_id, str) or not re.fullmatch("[0-9a-f]{32}", index_id):
                    raise ValueError("invalid checkpoint index directory")
                index_ids.add(index_id)
        target_indexes.mkdir(mode=0o700)
        for index_id in sorted(index_ids):
            directory = source_indexes / index_id
            if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob("*")):
                raise ValueError("checkpoint indexes must not contain symlinks")
            destination = target_indexes / index_id
            shutil.copytree(directory, destination)
            if _file_hashes(directory) != _file_hashes(destination):
                raise ValueError("checkpoint copy checksum mismatch")
        with sqlite3.connect(target_db) as target:
            source.backup(target)
    target_retrieval = run_root / "prepared/retrieval.jsonl"
    shutil.copy2(source_retrieval_path, target_retrieval)
    if sha256(target_retrieval) != source_ready["retrieval_sha256"]:
        raise ValueError("checkpoint retrieval results changed during copy")
    target_preflight = run_root / "prepared/embedding-preflight.json"
    preflight = source_preflight
    preflight["reused_from_static_configuration_sha256"] = preflight.get(
        "static_configuration_sha256"
    )
    preflight["static_configuration_sha256"] = canonical_hash(static)
    write_state_json(target_preflight, preflight)
    ready = dict(source_ready)
    ready["static_configuration_sha256"] = canonical_hash(static)
    ready["database_files_sha256"] = _file_hashes(target_db)
    ready["index_files_sha256"] = _file_hashes(target_indexes)
    ready["retrieval_sha256"] = sha256(target_retrieval)
    ready["embedding_preflight_sha256"] = sha256(target_preflight)
    ready["checkpoint_reuse"] = {
        "source_configuration_sha256": sha256(configuration_path),
        "source_preparation_sha256": sha256(source_ready_path),
        "base_retrieval_reused": True,
        "index_contract_source_unchanged": True,
        "source_documents_and_effective_index_identity_match": True,
    }
    write_state_json(run_root / "prepared/prepared.json", ready)
    target_db.chmod(0o600)
    (target_db.parent / "prepared.json").chmod(0o600)
    target_retrieval.chmod(0o600)
    target_preflight.chmod(0o600)
    for path in target_indexes.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)
        elif path.is_file() and not path.is_symlink():
            path.chmod(0o600)
    report = {
        "source_configuration_sha256": sha256(configuration_path),
        "source_database_sha256": sha256(source_db),
        "copied_database_sha256": sha256(target_db),
        "copied_index_directories": len(index_ids),
        "changed_code_files": sorted(changed),
        "data_model_and_effective_configuration_match": True,
        "indexes_require_validation_against_the_active_encoder_before_reuse": True,
        "base_retrieval_results_reused": True,
        "index_contract_source_unchanged": True,
    }
    atomic_json_once(run_root / "checkpoint-copy.json", report)
    return report


def _prepare(
    root: Path,
    run_root: Path,
    manifest: dict,
    inputs: list[dict],
    static: dict,
    *,
    reuse_from: Path | None = None,
) -> dict:
    _ensure_run_config(run_root, static)
    prepared_dir = run_root / "prepared"
    ready_path = prepared_dir / "prepared.json"
    if not ready_path.is_file() and prepared_dir.exists() and any(prepared_dir.iterdir()):
        raise ValueError(
            "preparation artifacts are incomplete; preserve them and use a new run-root"
        )
    prepared_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if reuse_from is not None:
        _copy_preparation_checkpoint(run_root, reuse_from, static)
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
    chunker = None
    if static["retrieval"].get("token_chunking"):
        from page_cite_rag.text import TokenBlockFactory

        chunker = TokenBlockFactory(DEFAULT_MODEL_DIR)
    documents: dict[str, tuple[str, dict[str, str]]] = {}
    for case in inputs:
        if case["case_id"] not in documents:
            documents[case["case_id"]] = _case_document(store, case, chunker=chunker)
    unique_documents = _unique_document_passages(documents)
    article_mappings = {
        case["case_id"]: _xrag_block_article_map(
            store, documents[case["case_id"]][0], case["source"]
        )
        for case in inputs
        if case["source"].get("format") == "xrag_article_bundle"
    }

    document_tokens = query_tokens = embed_inputs = 0
    retrieval_rows = []
    progress = {
        "stage": "exact_embedding_preflight",
        "document_ordinal": None,
        "document_count": len(unique_documents),
    }
    with _preparation_embedding_session(run_root, static, progress) as (server, counting_transport):
        encoder = NativeEmbeddingEncoder(
            counting_transport,
            DEFAULT_MODEL_DIR,
            max_tokens=static["embedding_input"]["document_max_tokens"],
            query_max_tokens=static["embedding_input"]["query_max_tokens"],
            batch_size=4,
        )
        counting_transport.query_prefix = encoder.identity["query_prompt"]

        document_tokens_by_id = {}
        document_inputs_by_id = {}
        for document_id in unique_documents:
            expected_document_tokens = 0
            expected_document_inputs = 0
            for block in store.get_blocks(document_id):
                for start, end in encoder.split_document(block.text):
                    expected_document_tokens += len(server.token_ids(block.text[start:end]))
                    expected_document_inputs += 1
            document_tokens_by_id[document_id] = expected_document_tokens
            document_inputs_by_id[document_id] = expected_document_inputs
        expected_document_tokens = sum(document_tokens_by_id.values())
        expected_document_inputs = sum(document_inputs_by_id.values())
        reused_document_ids = []
        for document_id in unique_documents:
            if store.get_retrieval_manifest(document_id, "bm25") is not None:
                BM25Retriever(store, document_id)
            if store.get_retrieval_manifest(document_id, dense_index_kind(encoder)) is not None:
                DenseRetriever(store, document_id, encoder, versioned=True)
                reused_document_ids.append(document_id)
        reused_document_tokens = sum(document_tokens_by_id[d] for d in reused_document_ids)
        expected_new_document_tokens = expected_document_tokens - reused_document_tokens
        query_token_counts = [
            (
                case["case_id"],
                len(server.token_ids(encoder.identity["query_prompt"] + case["question"])),
            )
            for case in inputs
        ]
        expected_query_tokens = sum(count for _, count in query_token_counts)
        overlong_queries = [
            {"case_id": case_id, "input_tokens": count}
            for case_id, count in query_token_counts
            if count > encoder.identity["query_max_tokens"]
        ]
        failed_checks = []
        if (
            expected_document_tokens
            > static["preparation_budget"]["document_embedding_input_tokens_max"]
        ):
            failed_checks.append("document_embedding_token_budget")
        if expected_query_tokens > static["preparation_budget"]["query_embedding_input_tokens_max"]:
            failed_checks.append("query_embedding_token_budget")
        if overlong_queries:
            failed_checks.append("query_encoder_context")
        embedding_preflight = {
            "state": "preflight_passed_not_embedded" if not failed_checks else "preflight_failed",
            "static_configuration_sha256": canonical_hash(static),
            "document_embedding_input_tokens": expected_document_tokens,
            "reused_document_ids": reused_document_ids,
            "reused_document_tokens": reused_document_tokens,
            "new_document_embedding_input_tokens_expected": expected_new_document_tokens,
            "document_embedding_input_tokens_by_document": document_tokens_by_id,
            "query_embedding_input_tokens": expected_query_tokens,
            "query_token_counts_by_case": dict(query_token_counts),
            "overlong_queries": overlong_queries,
            "failed_checks": failed_checks,
            "embedding_input_count": expected_document_inputs + len(inputs),
            "document_max_tokens": encoder.identity["max_tokens"],
            "query_max_tokens": encoder.identity["query_max_tokens"],
            "request_timeout_seconds": static["embedding_input"]["request_timeout_seconds"],
            "document_token_limit": static["preparation_budget"][
                "document_embedding_input_tokens_max"
            ],
            "query_token_limit": static["preparation_budget"]["query_embedding_input_tokens_max"],
            "gpu_temperature_limit_c": GPU_TEMPERATURE_LIMIT_C,
            "gpu_memory_limit_mib": MAX_GPU_MEMORY_MIB,
        }
        atomic_json_once(prepared_dir / "embedding-preflight.json", embedding_preflight)
        if "document_embedding_token_budget" in failed_checks:
            raise ValueError("document embedding input token preflight exceeds frozen budget")
        if "query_embedding_token_budget" in failed_checks:
            raise ValueError("query embedding input token preflight exceeds frozen budget")
        if overlong_queries:
            first = overlong_queries[0]
            raise ValueError(
                "a frozen query exceeds the embedding model's exact input limit: "
                f"{first['case_id']} uses {first['input_tokens']} tokens, "
                f"limit {encoder.identity['query_max_tokens']}"
            )

        progress["stage"] = "building_document_indexes"
        for ordinal, document_id in enumerate(unique_documents, start=1):
            progress["document_ordinal"] = ordinal
            if store.get_retrieval_manifest(document_id, "bm25") is None:
                build_bm25_index(store, document_id)
            if store.get_retrieval_manifest(document_id, dense_index_kind(encoder)) is None:
                build_dense_index(store, document_id, encoder, versioned=True)
        progress["stage"] = "building_query_retrieval"
        progress["document_ordinal"] = None
        for case in inputs:
            document_id, passage_by_block = documents[case["case_id"]]
            from page_cite_rag.application import document_retriever

            retriever = document_retriever(
                store, document_id, "hybrid", encoder=encoder, versioned_dense=True
            )
            result = retriever.search(
                case["question"], top_k=40 if static["retrieval"].get("reranker") else TOP_K
            )
            candidates = result
            result = _limit_search_result_to_context_groups(result, TOP_K)
            block_to_passage = passage_by_block
            block_to_article = article_mappings.get(case["case_id"], {})
            candidate_passage_ids = list(
                dict.fromkeys(
                    passage_id
                    for branch in result.branches
                    for block_id in branch.block_ids
                    if (passage_id := block_to_passage.get(block_id)) is not None
                )
            )
            candidate_article_ids = list(
                dict.fromkeys(
                    article_id
                    for branch in result.branches
                    for block_id in branch.block_ids
                    if (article_id := block_to_article.get(block_id)) is not None
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
                    "block_to_article_id": block_to_article,
                    "retrieved_article_ids": list(
                        dict.fromkeys(
                            block_to_article[hit.block.block_id]
                            for hit in result.hits
                            if hit.block.block_id in block_to_article
                        )
                    ),
                    "candidate_article_ids": candidate_article_ids,
                    "result": _search_result_to_json(result),
                    **(
                        {"candidate_result": _search_result_to_json(candidates)}
                        if static["retrieval"].get("reranker")
                        else {}
                    ),
                }
            )
        gpu_trace = server.diagnostics()
        document_tokens = counting_transport.document_tokens
        query_tokens = counting_transport.query_tokens
        embed_inputs = counting_transport.embed_inputs

        if document_tokens != expected_new_document_tokens or query_tokens != expected_query_tokens:
            raise ValueError("actual embedding token usage differs from exact preflight")

    # Preparation freezes the source, indexes and unre-ranked candidates only.
    # Model-backed reranking runs in the resumable retrieval-grid phase so one
    # slow query cannot discard successful rows from the entire corpus.
    reranker_trace = None
    atomic_jsonl_once(retrieval_path, retrieval_rows)
    for path in prepared_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o600)
    index_root = database_path.with_name(database_path.name + ".indexes")
    ready = {
        "state": "prepared_not_run",
        "case_count": len(inputs),
        "document_count": len(unique_documents),
        "retrieval_sha256": sha256(retrieval_path),
        "database_files_sha256": _file_hashes(database_path),
        "index_files_sha256": _file_hashes(index_root),
        "embedding_identity": encoder.identity,
        "embedding_inputs": {
            "document_tokens": document_tokens,
            "reused_document_tokens": reused_document_tokens,
            "corpus_document_tokens": expected_document_tokens,
            "reused_document_count": len(reused_document_ids),
            "query_tokens": query_tokens,
            "embed_inputs": embed_inputs,
            "embedding_requests": _embedding_request_accounting(counting_transport),
            "document_token_limit": static["preparation_budget"][
                "document_embedding_input_tokens_max"
            ],
            "query_token_limit": static["preparation_budget"]["query_embedding_input_tokens_max"],
        },
        "gpu_trace": gpu_trace,
        "reranker_gpu_trace": reranker_trace,
        "text_chunking": chunker.identity if chunker is not None else None,
        "static_configuration_sha256": canonical_hash(static),
        "embedding_preflight_sha256": sha256(prepared_dir / "embedding-preflight.json"),
    }
    atomic_json_once(ready_path, ready)
    return ready


def _load_prepared(
    run_root: Path,
    static: dict,
    *,
    include_reranked: bool = True,
) -> tuple[SQLiteDocumentStore, dict[str, dict]]:
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
    if static["retrieval"].get("reranker") and include_reranked:
        reranking_dir = prepared_dir / "reranked"
        results_path = reranking_dir / "results.jsonl"
        manifest_path = reranking_dir / "reranked.json"
        if not results_path.is_file() or not manifest_path.is_file():
            raise ValueError("run `--phase rerank` before reranked preflight or generation")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = {row["case_id"]: row for row in read_jsonl(results_path)}
        if (
            manifest.get("state") != "complete"
            or manifest.get("static_configuration_sha256") != canonical_hash(static)
            or manifest.get("prepared_retrieval_sha256") != sha256(retrieval_path)
            or manifest.get("results_sha256") != sha256(results_path)
            or set(rows) != set(retrieval)
        ):
            raise ValueError("reranked retrieval artifacts differ from the frozen run")
        for case_id, row in retrieval.items():
            ranked = rows[case_id]
            candidates = row.get("candidate_result")
            if candidates is None or ranked.get("candidate_sha256") != canonical_hash(
                _search_result_from_json(candidates).model_dump(mode="json")
            ):
                raise ValueError("reranked retrieval no longer matches its frozen candidates")
            row["reranked_result"] = ranked["result"]
    return SQLiteDocumentStore(database_path, read_only=True), retrieval


def _load_reading_plans(run_root: Path, static: dict, inputs: list[dict]) -> dict[str, dict]:
    if static.get("reading_plan") is None:
        raise ValueError("the frozen configuration does not include a reading planner")
    directory = run_root / "reading-plans"
    rows_path = directory / "results.jsonl"
    ready_path = directory / "ready.json"
    if not rows_path.is_file() or not ready_path.is_file():
        raise ValueError("run `--phase plan` before planned retrieval or generation")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    rows = load_unique_jsonl(rows_path)
    expected_ids = [case["case_id"] for case in inputs]
    if (
        ready.get("static_configuration_sha256") != canonical_hash(static)
        or ready.get("results_sha256") != sha256(rows_path)
        or ready.get("case_ids") != expected_ids
        or list(rows) != expected_ids
    ):
        raise ValueError("reading-plan artifacts differ from the frozen comparison")
    for row in rows.values():
        if row.get("status") == "complete":
            ReadingPlan.model_validate_json(
                json.dumps(row["plan"], ensure_ascii=False), strict=True
            )
        elif row.get("status") == "degraded" and row.get("plan") is None:
            if row.get("degraded_reason") not in {
                "original_query_fallback",
                "planned_query_token_limit_original_query_fallback",
            }:
                raise ValueError("degraded reading plan has no declared fallback")
        elif row.get("status") != "failed":
            raise ValueError("reading-plan artifact has an invalid terminal status")
    return rows


def _global_generation_budget(run_root: Path, static: dict) -> tuple[Path | None, int | None]:
    """Use the frozen batch ledger for every model-backed evaluation phase."""
    resolved = run_root.resolve()
    batch_root = _generation_batch_root(resolved)
    if batch_root is not None:
        progress_path = batch_root / "progress.json"
        if progress_path.is_symlink() or not progress_path.is_file():
            raise ValueError("quality batch progress is missing or unsafe; no model dispatch")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        scope = progress.get("scope")
        revision = progress.get("revision", progress.get("plan_revision"))
        if revision in {"agent-production-alignment-v4", "agent-control-followup-v1"}:
            limits = progress.get("generation_dispatch_limits")
            limit = limits.get("total") if isinstance(limits, dict) else None
            expected_limit = {
                "agent-production-alignment-v4": 1440,
                "agent-control-followup-v1": 1299,
            }[revision]
            if revision == "agent-control-followup-v1" and limits != {
                "development": 123,
                "regression_planners": 72,
                "regression_answers": 864,
                "new_diagnostics": 240,
                "total": 1299,
            }:
                raise ValueError("follow-up phase dispatch ceilings differ from the approved plan")
        else:
            limit = (
                scope.get("maximum_new_generation_dispatches") if isinstance(scope, dict) else None
            )
            expected_limit = {
                "quality-repair-closure-v2": 696,
                "agent-quality-closure-v3": 1134,
            }.get(revision)
        if expected_limit is None or progress.get("state") != "in_progress":
            raise ValueError("quality batch revision or state is not approved for dispatch")
        if type(limit) is not int or limit != expected_limit:
            raise ValueError(
                "quality batch dispatch ceiling differs from its approved frozen limit"
            )
        return batch_root / "global-model-calls.jsonl", limit

    reading_plan = static.get("reading_plan")
    if isinstance(reading_plan, dict):
        limit = reading_plan.get("planner_budget", {}).get("total_calls_limit")
        if type(limit) is not int or limit < 1:
            raise ValueError("reading-plan model-call ceiling is invalid")
        return run_root.parent / "global-model-calls.jsonl", limit
    return None, None


def _generation_batch_root(path: Path) -> Path | None:
    return next(
        (
            parent
            for parent in (path.resolve(), *path.resolve().parents)
            if parent.name
            in {
                "quality-repair-closure-v2",
                "agent-quality-closure-v3",
                "agent-production-alignment-v4",
                "agent-control-followup-v1",
            }
        ),
        None,
    )


def _global_generation_phase(run_root: Path, operation: str) -> str | None:
    """Map a frozen experiment directory to its approved dispatch phase."""
    batch_root = _generation_batch_root(run_root)
    if batch_root is None:
        return None
    relative_parts = run_root.resolve().relative_to(batch_root).parts
    if operation not in {"plan", "run"}:
        raise ValueError("unexpected generation operation")
    if batch_root.name in {"agent-production-alignment-v4", "agent-control-followup-v1"}:
        joined = "/".join(part.casefold().replace("_", "-") for part in relative_parts)
        if "protocol" in joined:
            return "protocol"
        if "development" in joined:
            return "development"
        if "regression" in joined:
            return "regression_planners" if operation == "plan" else "regression_answers"
        if "new-diagnostic" in joined:
            return "new_diagnostics"
        raise ValueError("quality generation run root must name its frozen phase")
    if batch_root.name != "agent-quality-closure-v3":
        return None
    if any("protocol" in part for part in relative_parts):
        return "protocol"
    if any("development" in part for part in relative_parts):
        return "development"
    if any("regression" in part for part in relative_parts):
        return "regression_planners" if operation == "plan" else "regression_answers"
    if any("holdout" in part for part in relative_parts):
        return "new_holdout_planners_and_two_flows"
    raise ValueError("v3 generation run root must name its frozen phase")


def _run_reading_plans(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    """Generate one bounded, answer-free plan per case and preserve failed attempts."""
    if static.get("reading_plan") is None:
        raise ValueError("select all four reading-plan comparison arms before planning")
    _ensure_run_config(run_root, static)
    store, retrieval = _load_prepared(run_root, static)
    directory = run_root / "reading-plans"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempts_dir = directory / ".attempts"
    attempts_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    results_path = directory / "results.jsonl"
    ready_path = directory / "ready.json"
    if ready_path.exists():
        _load_reading_plans(run_root, static, inputs)
        return json.loads(ready_path.read_text(encoding="utf-8"))
    completed = load_unique_jsonl(results_path)
    attempts = load_unique_jsonl(attempts_dir / "ledger.jsonl")
    reject_interrupted_attempts(attempts, completed, "reading plan")
    expected_ids = [case["case_id"] for case in inputs]
    if (completed.keys() | attempts.keys()) - set(expected_ids):
        raise ValueError("reading-plan artifacts contain unknown case IDs")

    from page_cite_rag.application import FrozenSearchRetriever, document_workflow

    started = time.monotonic()
    terminal_error = None
    global_call_ledger, global_call_limit = _global_generation_budget(run_root, static)
    global_call_phase = _global_generation_phase(run_root, "plan")
    with LlamaModelClient(
        _llama_config(static, num_predict=static["reading_plan"]["num_predict"]),
        budget_policy="exact",
        recover_output_errors=True,
        sampling_profile=static["answer"]["sampling_profile"],
        sampling_seed=42,
    ) as model:
        generator_identity = model.identity
        counted_model = _CountingModelClient(
            model,
            global_call_ledger=global_call_ledger,
            global_call_limit=global_call_limit,
            global_call_phase=global_call_phase,
        )
        for case in inputs:
            case_id = case["case_id"]
            if case_id in completed:
                continue
            attempt_path = attempts_dir / f"{case_id}.json"
            if attempt_path.exists():
                raise ValueError("reading-plan attempt exists without a result; no retry")
            row = retrieval[case_id]
            search = _search_result_from_json(row["result"])
            document = store.get_document(row["document_id"])
            if document is None:
                raise ValueError("prepared source document is missing")
            workflow = document_workflow(
                store,
                row["document_id"],
                FrozenSearchRetriever(search, top_k=TOP_K),
                model,
                model.config,
                top_k=TOP_K,
                answer_format=_answer_format_for(case),
            )
            outline = workflow.source_outline()
            titles = tuple(
                section.title
                for section in (outline.sections if outline is not None else ())
                if section.title
            )[:80]
            source_language = document.language
            source_language = (
                source_language if source_language in {"en", "zh-Hant", "zh-Hans"} else "unknown"
            )
            messages = reading_plan_messages(
                case["question"],
                source_language=source_language,
                output_language=case["language"],
                outline_titles=titles,
            )
            marker = {
                "case_id": case_id,
                "state": "attempt_started_no_retry",
                "static_configuration_sha256": canonical_hash(static),
                "prompt_sha256": canonical_hash([item.model_dump() for item in messages]),
            }
            atomic_json_once(attempt_path, marker)
            append_jsonl_record(attempts_dir / "ledger.jsonl", marker)
            counted_model.begin_case(
                structured_call_limit=1,
                call_context={
                    "phase": "plan",
                    "arm": "shared-planner",
                    "case_id": case_id,
                },
            )
            attempt_started = time.monotonic()
            try:
                result = counted_model.generate(messages, ReadingPlan)
                record = {
                    "case_id": case_id,
                    "status": "complete",
                    "plan": result.value.model_dump(mode="json"),
                    "prompt_sha256": marker["prompt_sha256"],
                    "response_schema_sha256": canonical_hash(ReadingPlan.model_json_schema()),
                    "generator_identity": generator_identity,
                    "generation_stats": result.stats.model_dump(mode="json"),
                    "raw_model_output": result.raw_model_output,
                    "call_records": counted_model.records,
                    "elapsed_ms": (time.monotonic() - attempt_started) * 1000,
                }
            except Exception as exc:
                record = {
                    "case_id": case_id,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:500],
                    "call_records": counted_model.records,
                    "elapsed_ms": (time.monotonic() - attempt_started) * 1000,
                }
                if isinstance(exc, ModelError) and not isinstance(exc, OutputValidationError):
                    terminal_error = exc
            append_jsonl_record(results_path, record)
            completed[case_id] = record
            if terminal_error is not None:
                break
        diagnostics = _safe_model_diagnostics(model)
    if list(completed) != expected_ids:
        if terminal_error is not None:
            remaining = len(expected_ids) - len(completed)
            raise ModelError(
                "reading-plan phase stopped after a terminal model error; "
                f"{remaining} unstarted case(s) remain; resume will not retry the failed case"
            ) from None
        raise ValueError("reading-plan phase ended with unresolved case slots")
    planner_usage = _structured_usage(
        [call for row in completed.values() for call in row.get("call_records", ())]
    )
    ready = {
        "state": "complete_with_recorded_failures"
        if any(row["status"] == "failed" for row in completed.values())
        else "complete",
        "static_configuration_sha256": canonical_hash(static),
        "case_ids": expected_ids,
        "successful_plans": sum(row["status"] == "complete" for row in completed.values()),
        "failed_plans": sum(row["status"] == "failed" for row in completed.values()),
        "planner_calls_attempted": sum(
            len(row.get("call_records", ())) for row in completed.values()
        ),
        "input_tokens": planner_usage["input_tokens_lower_bound"],
        "output_tokens": planner_usage["output_tokens_lower_bound"],
        "usage_unknown": planner_usage["usage_unknown"],
        "generator_identity": generator_identity,
        "results_sha256": sha256(results_path),
        "gpu_diagnostics": diagnostics,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json_once(ready_path, ready)
    return ready


def _load_planned_retrieval(run_root: Path, static: dict, inputs: list[dict]) -> dict[str, dict]:
    directory = run_root / "planned-retrieval"
    results_path = directory / "results.jsonl"
    ready_path = directory / "ready.json"
    if not results_path.is_file() or not ready_path.is_file():
        raise ValueError("run `--phase planned-retrieval` before preflight or generation")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    rows = load_unique_jsonl(results_path)
    expected_ids = [case["case_id"] for case in inputs]
    if (
        ready.get("static_configuration_sha256") != canonical_hash(static)
        or ready.get("results_sha256") != sha256(results_path)
        or ready.get("case_ids") != expected_ids
        or list(rows) != expected_ids
    ):
        raise ValueError("planned-retrieval artifacts differ from the frozen comparison")
    for row in rows.values():
        if row.get("status") == "complete":
            _search_result_from_json(row["result"])
            _search_result_from_json(row["candidate_result"])
            _search_result_from_json(row["original_query_result"])
            _search_result_from_json(row["original_query_candidate_result"])
        elif row.get("status") == "degraded":
            _search_result_from_json(row["result"])
            _search_result_from_json(row["candidate_result"])
            _search_result_from_json(row["original_query_result"])
            _search_result_from_json(row["original_query_candidate_result"])
            if row.get("degraded_reason") not in {
                "original_query_fallback",
                "planned_query_token_limit_original_query_fallback",
            }:
                raise ValueError("planned retrieval has an unknown degraded path")
        elif row.get("status") != "failed":
            raise ValueError("planned-retrieval artifact has an invalid terminal status")
    return rows


def _run_planned_retrieval(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    """Search original and plan queries with one shared native embedding session."""
    if static.get("reading_plan") is None:
        raise ValueError("the frozen configuration does not include planned retrieval")
    store, base_retrieval = _load_prepared(run_root, static)
    plans = _load_reading_plans(run_root, static, inputs)
    directory = run_root / "planned-retrieval"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempts_dir = directory / ".attempts"
    attempts_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    results_path = directory / "results.jsonl"
    ready_path = directory / "ready.json"
    if ready_path.exists():
        _load_planned_retrieval(run_root, static, inputs)
        return json.loads(ready_path.read_text(encoding="utf-8"))
    completed = load_unique_jsonl(results_path)
    attempts = load_unique_jsonl(attempts_dir / "ledger.jsonl")
    reject_interrupted_attempts(attempts, completed, "planned retrieval")
    expected_ids = [case["case_id"] for case in inputs]
    if (completed.keys() | attempts.keys()) - set(expected_ids):
        raise ValueError("planned-retrieval artifacts contain unknown case IDs")

    from page_cite_rag.application import document_retriever, search_planned_hybrid
    from page_cite_rag.reading_plan import planned_query_routes

    query_sets = {}
    for case in inputs:
        plan_row = plans[case["case_id"]]
        if plan_row["status"] != "complete":
            query_sets[case["case_id"]] = {
                "queries": (case["question"],),
                "query_to_needs": {},
                "planner_degraded": True,
            }
            continue
        plan = ReadingPlan.model_validate_json(
            json.dumps(plan_row["plan"], ensure_ascii=False), strict=True
        )
        queries, query_need_ids = planned_query_routes(case["question"], plan)
        query_to_needs = {query: list(ids) for query, ids in query_need_ids.items()}
        query_sets[case["case_id"]] = {
            "queries": queries,
            "query_to_needs": query_to_needs,
            "planner_degraded": False,
        }

    progress = {"stage": "planned_query_preflight", "document_ordinal": None}
    total_query_tokens = 0
    expected_queries = 0
    query_token_counts = {}
    overlong_queries = []
    query_fallback_reasons = {}
    query_tokens_by_case: dict[str, list[int]] = {}
    embedding_queries_by_case: dict[str, tuple[str, ...]] = {}
    planned_retrieval_rows = dict(completed)
    fatal_error = None
    started = time.monotonic()
    try:
        with _preparation_embedding_session(run_root, static, progress) as (server, transport):
            encoder = NativeEmbeddingEncoder(
                transport,
                DEFAULT_MODEL_DIR,
                max_tokens=static["embedding_input"]["document_max_tokens"],
                query_max_tokens=static["embedding_input"]["query_max_tokens"],
                batch_size=4,
            )
            transport.query_prefix = encoder.identity["query_prompt"]
            for case_id, query_set in query_sets.items():
                case_tokens = []
                overlong = False
                for query in query_set["queries"]:
                    exact_tokens = len(server.token_ids(transport.query_prefix + query))
                    query_token_counts[f"{case_id}:{len(query_token_counts)}"] = exact_tokens
                    if exact_tokens > encoder.identity["query_max_tokens"]:
                        overlong = True
                        overlong_queries.append(
                            {
                                "case_id": case_id,
                                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                                "input_tokens": exact_tokens,
                                "limit": encoder.identity["query_max_tokens"],
                            }
                        )
                    case_tokens.append(exact_tokens)
                query_tokens_by_case[case_id] = case_tokens
                if overlong:
                    query_fallback_reasons[case_id] = (
                        "planned_query_token_limit_original_query_fallback"
                    )
                    if case_tokens[0] <= encoder.identity["query_max_tokens"]:
                        embedding_queries_by_case[case_id] = query_set["queries"][:1]
                        total_query_tokens += case_tokens[0]
                        expected_queries += 1
                    else:
                        embedding_queries_by_case[case_id] = ()
                    continue
                embedding_queries_by_case[case_id] = query_set["queries"]
                total_query_tokens += sum(case_tokens)
                expected_queries += len(case_tokens)
            query_limit = static["reading_plan"]["planned_query_input_tokens_max"]
            if total_query_tokens > query_limit:
                raise ValueError("planned retrieval exceeds its frozen exact query-token budget")
            query_preflight = {
                "state": "passed_before_query_embedding",
                "static_configuration_sha256": canonical_hash(static),
                "query_count": expected_queries,
                "query_input_tokens": total_query_tokens,
                "query_token_limit": query_limit,
                "query_tokens_by_case_and_ordinal": query_token_counts,
                "overlong_queries": overlong_queries,
                "fallback_case_ids": sorted(query_fallback_reasons),
                "embedding_identity": encoder.identity,
            }
            preflight_path = directory / "query-preflight.json"
            if preflight_path.exists():
                if json.loads(preflight_path.read_text(encoding="utf-8")) != query_preflight:
                    raise ValueError("planned query preflight changed after it was frozen")
            else:
                atomic_json_once(preflight_path, query_preflight)

            for case in inputs:
                case_id = case["case_id"]
                if case_id in completed:
                    continue
                query_set = query_sets[case_id]
                queries_to_run = embedding_queries_by_case[case_id]
                if not queries_to_run:
                    base_row = base_retrieval[case_id]
                    base_search = _expand_cached_hybrid_candidates(
                        _search_result_from_json(base_row["result"]),
                        store,
                        top_k=(2 * static["retrieval"]["candidates_per_branch"]),
                        constant=static["retrieval"]["rrf_constant"],
                    )
                    block_to_passage = base_row.get("block_to_passage_id", {})
                    fallback_passage_ids = list(
                        dict.fromkeys(
                            block_to_passage[hit.block.block_id]
                            for hit in base_search.hits
                            if hit.block.block_id in block_to_passage
                        )
                    )
                    fallback = {
                        "case_id": case_id,
                        "status": "degraded",
                        "degraded_reason": query_fallback_reasons.get(
                            case_id, "original_query_fallback"
                        ),
                        "planner_status": plans[case_id]["status"],
                        "planner_failure_type": plans[case_id].get("original_planner_failure_type"),
                        "overlong_query_records": [
                            item for item in overlong_queries if item["case_id"] == case_id
                        ],
                        "document_id": base_row["document_id"],
                        "queries": [],
                        "query_to_need_ids": {},
                        "candidate_result": _search_result_to_json(base_search),
                        "result": _search_result_to_json(base_search),
                        "original_query_candidate_result": _search_result_to_json(base_search),
                        "original_query_result": _search_result_to_json(base_search),
                        "candidate_passage_ids": fallback_passage_ids,
                        "retrieved_passage_ids": fallback_passage_ids,
                        "original_query_candidate_passage_ids": fallback_passage_ids,
                        "original_query_retrieved_passage_ids": fallback_passage_ids,
                        "fallback_source_block_ids": [
                            hit.block.block_id for hit in base_search.hits
                        ],
                        "elapsed_ms": 0.0,
                        "new_query_embedding_requests": 0,
                    }
                    append_jsonl_record(results_path, fallback)
                    completed[case_id] = fallback
                    planned_retrieval_rows[case_id] = fallback
                    continue
                attempt_path = attempts_dir / f"{case_id}.json"
                if attempt_path.exists():
                    raise ValueError("planned-retrieval attempt exists without a result; no retry")
                marker = {
                    "case_id": case_id,
                    "state": "attempt_started_no_retry",
                    "static_configuration_sha256": canonical_hash(static),
                }
                atomic_json_once(attempt_path, marker)
                append_jsonl_record(attempts_dir / "ledger.jsonl", marker)
                case_started = time.monotonic()
                try:
                    base_row = base_retrieval[case_id]
                    document_id = base_row["document_id"]
                    retriever = document_retriever(
                        store,
                        document_id,
                        "hybrid",
                        encoder=encoder,
                        versioned_dense=True,
                        candidates=static["reading_plan"]["retrieval_candidates_per_branch"],
                    )
                    candidate, query_results, shared_search_trace = search_planned_hybrid(
                        retriever,
                        case["question"],
                        ReadingPlan.model_validate_json(
                            json.dumps(plans[case_id]["plan"], ensure_ascii=False), strict=True
                        ),
                        candidates_per_query=(
                            static["reading_plan"]["retrieval_candidate_count_per_query"]
                        ),
                        fused_pool_size=static["reading_plan"]["fused_candidate_count"],
                        queries_override=tuple(queries_to_run),
                        query_to_needs_override={
                            query: tuple(query_set["query_to_needs"].get(query, ()))
                            for query in queries_to_run
                        },
                    )
                    query_trace = [
                        {
                            "query": query,
                            "need_ids": query_set["query_to_needs"].get(query, []),
                            "index_id": search.index_id,
                            "elapsed_ms": search.elapsed_ms,
                            "ranked_block_ids": [hit.block.block_id for hit in search.hits],
                            "branches": [
                                branch.model_dump(mode="json") for branch in search.branches
                            ],
                        }
                        for query, search in zip(queries_to_run, query_results, strict=True)
                    ]
                    result = _limit_search_result_to_context_groups(
                        candidate, _context_group_limit(static)
                    )
                    original_candidate = query_results[0]
                    original_result = _limit_search_result_to_context_groups(
                        original_candidate, _context_group_limit(static)
                    )
                    block_to_passage = base_row.get("block_to_passage_id", {})
                    candidate_passage_ids = list(
                        dict.fromkeys(
                            block_to_passage[hit.block.block_id]
                            for hit in candidate.hits
                            if hit.block.block_id in block_to_passage
                        )
                    )
                    record = {
                        "case_id": case_id,
                        "status": ("degraded" if case_id in query_fallback_reasons else "complete"),
                        "degraded_reason": query_fallback_reasons.get(case_id),
                        "planner_status": plans[case_id]["status"],
                        "planner_degraded": query_set["planner_degraded"],
                        "document_id": document_id,
                        "queries": query_trace,
                        "query_to_need_ids": query_set["query_to_needs"],
                        "shared_planned_search_trace": shared_search_trace,
                        "candidate_result": _search_result_to_json(candidate),
                        "result": _search_result_to_json(result),
                        "original_query_candidate_result": _search_result_to_json(
                            original_candidate
                        ),
                        "original_query_result": _search_result_to_json(original_result),
                        "candidate_passage_ids": candidate_passage_ids,
                        "retrieved_passage_ids": list(
                            dict.fromkeys(
                                block_to_passage[hit.block.block_id]
                                for hit in result.hits
                                if hit.block.block_id in block_to_passage
                            )
                        ),
                        "original_query_candidate_passage_ids": _passage_ids_from_search(
                            original_candidate, block_to_passage
                        ),
                        "original_query_retrieved_passage_ids": list(
                            dict.fromkeys(
                                block_to_passage[hit.block.block_id]
                                for hit in original_result.hits
                                if hit.block.block_id in block_to_passage
                            )
                        ),
                        "elapsed_ms": (time.monotonic() - case_started) * 1000,
                    }
                except Exception as exc:
                    record = {
                        "case_id": case_id,
                        "status": "failed",
                        "failure_type": type(exc).__name__,
                        "elapsed_ms": (time.monotonic() - case_started) * 1000,
                    }
                    try:
                        case_gpu = server.diagnostics()
                    except Exception as diagnostic_error:
                        case_gpu = {"diagnostic_error": type(diagnostic_error).__name__}
                    if (
                        getattr(exc, "category", None)
                        or isinstance(exc, (ModelError, TimeoutError))
                        or _gpu_guard_detected(case_gpu)
                    ):
                        fatal_error = type(exc).__name__
                append_jsonl_record(results_path, record)
                completed[case_id] = record
                if fatal_error is not None:
                    break
            embedding_usage = _embedding_request_accounting(transport)
            gpu_diagnostics = server.diagnostics()
            if (
                not fatal_error
                and not any(row.get("status") == "failed" for row in completed.values())
                and transport.query_tokens != total_query_tokens
            ):
                raise ValueError("planned query embedding usage differs from exact preflight")
    except Exception as exc:
        fatal_error = fatal_error or type(exc).__name__
        embedding_usage = None
        gpu_diagnostics = {"error_type": type(exc).__name__}

    for case in inputs:
        if case["case_id"] not in completed:
            continue
        planned_retrieval_rows[case["case_id"]] = completed[case["case_id"]]
    complete_slots = list(planned_retrieval_rows) == expected_ids
    ready = {
        "state": (
            "stopped_unresolved"
            if not complete_slots
            else "complete_with_recorded_failures"
            if any(row.get("status") == "failed" for row in planned_retrieval_rows.values())
            else "complete"
        ),
        "static_configuration_sha256": canonical_hash(static),
        "case_ids": expected_ids,
        "completed_slots": len(planned_retrieval_rows),
        "failed_slots": sum(
            row.get("status") == "failed" for row in planned_retrieval_rows.values()
        ),
        "query_preflight_sha256": (
            sha256(directory / "query-preflight.json")
            if (directory / "query-preflight.json").is_file()
            else None
        ),
        "embedding_usage": embedding_usage,
        "gpu_diagnostics": gpu_diagnostics,
        "fatal_error_type": fatal_error,
        "results_sha256": sha256(results_path) if results_path.is_file() else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    if ready_path.exists():
        raise ValueError("planned-retrieval ready marker already exists")
    atomic_json_once(ready_path, ready)
    if not complete_slots:
        raise ValueError("planned retrieval stopped with unresolved case slots; no retry")
    return ready


def _validate_downstream_stage_configuration(run_root: Path, current: dict, *, stage: str) -> dict:
    """Permit a downstream-only runner edit without changing frozen preparation inputs."""
    configuration_path = run_root / "configuration.json"
    if not configuration_path.is_file():
        raise ValueError("downstream stage requires a frozen run configuration")
    frozen = json.loads(configuration_path.read_text(encoding="utf-8")).get("static")
    if not isinstance(frozen, dict):
        raise ValueError("frozen run configuration is malformed")

    frozen_sources = frozen.get("source_code_sha256")
    current_sources = current.get("source_code_sha256")
    if not isinstance(frozen_sources, dict) or not isinstance(current_sources, dict):
        raise ValueError("frozen run source hashes are missing")
    changed = {
        path
        for path in frozen_sources.keys() | current_sources.keys()
        if frozen_sources.get(path) != current_sources.get(path)
    }
    if not changed:
        if current != frozen:
            raise ValueError("downstream stage configuration differs from the frozen run")
        return frozen
    if changed != {"scripts/run_reading_eval.py"}:
        raise ValueError("downstream stage source changed beyond the evaluation runner")

    equivalent = json.loads(json.dumps(current))
    equivalent["source_code_sha256"] = frozen_sources
    if equivalent != frozen:
        raise ValueError("downstream stage configuration differs beyond the runner source hash")

    audit_dir = run_root / "stage-continuation-audits"
    audit_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    audit_path = audit_dir / f"{stage}.json"
    audit = {
        "stage": stage,
        "state": "frozen_preparation_reused_after_runner_only_change",
        "frozen_static_configuration_sha256": canonical_hash(frozen),
        "current_static_configuration_sha256": canonical_hash(current),
        "changed_source_paths": sorted(changed),
        "frozen_runner_sha256": frozen_sources["scripts/run_reading_eval.py"],
        "current_runner_sha256": current_sources["scripts/run_reading_eval.py"],
        "policy": (
            "preserve original configuration and prepared artifacts; downstream stage uses "
            "the frozen static configuration"
        ),
    }
    if audit_path.exists():
        if json.loads(audit_path.read_text(encoding="utf-8")) != audit:
            raise ValueError("downstream continuation audit changed after it was recorded")
    else:
        atomic_json_once(audit_path, audit)
    return frozen


def _rank_planned_initial_retrieval(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    """Rerank each shared initial pool once; live Agent tool searches remain hybrid."""
    if not static["retrieval"].get("shared_initial_reranker"):
        raise ValueError("the frozen run did not select shared initial reranking")
    _store, _base = _load_prepared(run_root, static, include_reranked=False)
    planned = _load_planned_retrieval(run_root, static, inputs)
    directory = run_root / "planned-retrieval" / "reranked"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    preflight_path = directory / "preflight.json"
    attempts_path = directory / "attempts.jsonl"
    results_path = directory / "results.jsonl"
    failures_path = directory / "failures.jsonl"
    manifest_path = directory / "reranked.json"
    from page_cite_rag.hybrid import rerank_search_result
    from page_cite_rag.reranker_server import (
        DEFAULT_MODEL_PATH,
        MAX_PAIR_TOKENS,
        NativeRerankerServer,
    )

    pools = []
    for case in inputs:
        row = planned[case["case_id"]]
        if row.get("status") not in ("complete", "degraded"):
            raise ValueError("reranker input contains a failed planned-retrieval case")
        for pool_name, key in (
            ("original_query", "original_query_candidate_result"),
            ("decomposed", "candidate_result"),
        ):
            candidate = _search_result_from_json(row[key])
            if candidate.document_id != row["document_id"] or candidate.query != case["question"]:
                raise ValueError("reranker candidate pool has a mismatched source or query")
            count = len(candidate.context_groups) or len(candidate.hits)
            if not 1 <= count <= 100:
                raise ValueError("reranker candidate pool is outside the frozen 1–100 range")
            slot_id = f"{case['case_id']}::{pool_name}"
            pools.append(
                {
                    "slot_id": slot_id,
                    "case_id": case["case_id"],
                    "pool": pool_name,
                    "candidate_sha256": canonical_hash(candidate.model_dump(mode="json")),
                    "candidate_count": count,
                    "candidate": candidate,
                }
            )
    if len(pools) != 2 * len(inputs):
        raise ValueError("reranker preflight did not produce exactly two initial pools per case")

    batch_root = next(
        (
            parent
            for parent in (run_root.resolve(), *run_root.resolve().parents)
            if parent.name == "quality-repair-closure-v2"
        ),
        None,
    )
    if batch_root is None:
        raise ValueError("shared initial reranking requires the frozen v2 batch root")
    progress_path = batch_root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    counters = progress.setdefault("counters", {})
    used_pairs = int(counters.get("reranker_pairs", 0))
    used_tokens = int(counters.get("reranker_pair_tokens", 0))
    pair_limit = static["preparation_budget"].get("reranker_pairs_max")
    token_limit = static["preparation_budget"].get("reranker_pair_tokens_max")
    total_pairs = sum(pool["candidate_count"] for pool in pools)
    upper_tokens = total_pairs * MAX_PAIR_TOKENS
    if (
        pair_limit != QUALITY_RERANKER_PAIR_LIMIT
        or token_limit != QUALITY_RERANKER_PAIR_TOKEN_LIMIT
    ):
        raise ValueError("shared initial reranker limits differ from the frozen v2 policy")
    attempts = load_unique_jsonl(attempts_path, key="slot_id")
    results = load_unique_jsonl(results_path, key="slot_id")
    failures = load_unique_jsonl(failures_path, key="slot_id")
    slot_ids = {pool["slot_id"] for pool in pools}
    if (attempts.keys() | results.keys() | failures.keys()) - slot_ids:
        raise ValueError("shared reranker artifacts contain an unknown candidate pool")
    if (results.keys() | failures.keys()) - attempts.keys() or results.keys() & failures.keys():
        raise ValueError("shared reranker outcomes do not match their attempt ledger")
    reject_interrupted_attempts(attempts, results | failures, "shared initial reranker")
    pools_by_id = {pool["slot_id"]: pool for pool in pools}
    for slot_id, result in results.items():
        pool = pools_by_id[slot_id]
        if result.get("candidate_sha256") != pool["candidate_sha256"]:
            raise ValueError("saved reranked result no longer matches its frozen candidate pool")
    pending = [pool for pool in pools if pool["slot_id"] not in attempts]
    if failures:
        raise ValueError("shared reranker contains a recorded failure; do not retry")

    completed_pairs = sum(int(item["candidate_count"]) for item in results.values())
    completed_tokens = sum(
        sum(int(value) for value in item.get("pair_tokens", [])) for item in results.values()
    )
    # An atomic result row is authoritative if the process stopped before its
    # progress-counter update. Reconcile those already-completed pairs once.
    used_pairs = max(used_pairs, completed_pairs)
    used_tokens = max(used_tokens, completed_tokens)
    prior_pairs = used_pairs - completed_pairs
    prior_tokens = used_tokens - completed_tokens
    pending_pairs = sum(pool["candidate_count"] for pool in pending)
    pending_token_upper_bound = pending_pairs * MAX_PAIR_TOKENS
    if (
        prior_pairs + completed_pairs + pending_pairs > pair_limit
        or prior_tokens + completed_tokens + pending_token_upper_bound > token_limit
    ):
        raise ValueError("complete initial reranker pool does not fit the frozen v2 budget")
    counters["reranker_pairs"] = used_pairs
    counters["reranker_pair_tokens"] = used_tokens
    plan = {
        "state": "complete_exact_candidate_preflight_no_reranker_dispatch",
        "static_configuration_sha256": canonical_hash(static),
        "planned_retrieval_sha256": sha256(run_root / "planned-retrieval" / "results.jsonl"),
        "case_count": len(inputs),
        "pool_count": len(pools),
        "candidate_pairs": total_pairs,
        "pair_token_upper_bound": upper_tokens,
        "max_pairs_per_pool": 100,
        "max_pair_tokens": MAX_PAIR_TOKENS,
        "frozen_pair_limit": pair_limit,
        "frozen_pair_token_limit": token_limit,
        "rerank_agent_tool_searches": False,
        "candidate_pools": [
            {key: value for key, value in pool.items() if key != "candidate"} for pool in pools
        ],
    }
    if preflight_path.exists():
        if json.loads(preflight_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("shared initial reranker preflight changed after freezing")
    else:
        atomic_json_once(preflight_path, plan)

    write_state_json(progress_path, progress)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("static_configuration_sha256") != canonical_hash(static)
            or manifest.get("preflight_sha256") != sha256(preflight_path)
            or manifest.get("results_sha256") != sha256(results_path)
            or len(results) != len(pools)
        ):
            raise ValueError("shared reranker manifest does not match frozen inputs")
        return manifest

    gpu_started = None
    gpu_diagnostics = None
    try:
        if pending:
            # Count model startup and shutdown in owned GPU time, not only request time.
            gpu_started = time.monotonic()
            with NativeRerankerServer(
                DEFAULT_MODEL_PATH,
                GPU_UUID,
                max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C,
            ) as ranker:
                for pool in pending:
                    candidate = pool["candidate"]
                    slot_id = pool["slot_id"]
                    marker = {
                        "slot_id": slot_id,
                        "case_id": pool["case_id"],
                        "pool": pool["pool"],
                        "state": "started_no_retry",
                        "candidate_count": pool["candidate_count"],
                        "candidate_sha256": pool["candidate_sha256"],
                    }
                    append_jsonl_record(attempts_path, marker)
                    pool_started = time.monotonic()
                    try:
                        ranked = rerank_search_result(candidate, ranker, top_k=TOP_K, store=_store)
                        pair_tokens = list(ranker.last_pair_tokens)
                        if len(pair_tokens) != pool["candidate_count"]:
                            raise ValueError(
                                "reranker pair usage differs from the exact candidate count"
                            )
                        record = {
                            "slot_id": slot_id,
                            "case_id": pool["case_id"],
                            "pool": pool["pool"],
                            "candidate_count": pool["candidate_count"],
                            "candidate_sha256": pool["candidate_sha256"],
                            "result": ranked.model_dump(mode="json"),
                            "pair_tokens": pair_tokens,
                            "batch_sizes": list(ranker.last_batch_sizes),
                            "tokenization_ms": ranker.last_tokenization_ms,
                            "forward_ms": ranker.last_forward_ms,
                            "ranker_total_ms": ranker.last_total_ms,
                            "wall_ms": (time.monotonic() - pool_started) * 1000,
                        }
                    except Exception as exc:
                        # Persist exactly what the tokenizer observed. A failed pool is never
                        # retried; unknown tokenized pairs are charged at the frozen maximum.
                        known_pair_tokens = list(ranker.last_pair_tokens)
                        unknown_pairs = max(0, pool["candidate_count"] - len(known_pair_tokens))
                        pair_tokens_unknown_upper_bound = unknown_pairs * MAX_PAIR_TOKENS
                        failure = {
                            **marker,
                            "state": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                            "pair_tokens_known": known_pair_tokens,
                            "pair_tokens_unknown_upper_bound": pair_tokens_unknown_upper_bound,
                            "gpu": ranker.diagnostics(),
                        }
                        append_jsonl_record(failures_path, failure)
                        current_completed_pairs = sum(
                            int(item["candidate_count"]) for item in results.values()
                        )
                        current_completed_tokens = sum(
                            sum(int(value) for value in item.get("pair_tokens", []))
                            for item in results.values()
                        )
                        progress["counters"]["reranker_pair_tokens"] = (
                            prior_tokens
                            + current_completed_tokens
                            + sum(int(item) for item in known_pair_tokens)
                        )
                        progress["counters"]["reranker_pairs"] = (
                            prior_pairs + current_completed_pairs + len(known_pair_tokens)
                        )
                        progress["counters"]["unknown_reranker_pair_tokens_upper_bound"] = (
                            int(counters.get("unknown_reranker_pair_tokens_upper_bound", 0))
                            + pair_tokens_unknown_upper_bound
                        )
                        progress["counters"]["unknown_reranker_pairs_upper_bound"] = (
                            int(counters.get("unknown_reranker_pairs_upper_bound", 0))
                            + unknown_pairs
                        )
                        progress["last_reranker_gpu_diagnostics"] = ranker.diagnostics()
                        write_state_json(progress_path, progress)
                        raise
                    exact_pairs = (
                        prior_pairs
                        + sum(int(item["candidate_count"]) for item in results.values())
                        + int(record["candidate_count"])
                    )
                    exact_tokens = (
                        prior_tokens
                        + sum(
                            sum(int(value) for value in item.get("pair_tokens", []))
                            for item in results.values()
                        )
                        + sum(int(value) for value in pair_tokens)
                    )
                    if exact_pairs > pair_limit or exact_tokens > token_limit:
                        raise ValueError("shared initial reranker exceeded its frozen v2 budget")
                    append_jsonl_record(results_path, record)
                    results[slot_id] = record
                    progress["counters"]["reranker_pairs"] = exact_pairs
                    progress["counters"]["reranker_pair_tokens"] = exact_tokens
                    progress["last_reranker_gpu_diagnostics"] = ranker.diagnostics()
                    write_state_json(progress_path, progress)
                gpu_diagnostics = ranker.diagnostics()
    finally:
        if gpu_started is not None:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress["counters"]["owned_gpu_seconds"] = float(
                progress["counters"].get("owned_gpu_seconds", 0.0)
            ) + (time.monotonic() - gpu_started)
            if gpu_diagnostics is not None:
                progress["last_reranker_gpu_diagnostics"] = gpu_diagnostics
            write_state_json(progress_path, progress)
    if len(results) != len(pools):
        raise ValueError("shared initial reranking stopped before every frozen pool completed")
    pair_tokens_total = sum(sum(row.get("pair_tokens", [])) for row in results.values())
    manifest = {
        "state": "complete",
        "static_configuration_sha256": canonical_hash(static),
        "planned_retrieval_sha256": sha256(run_root / "planned-retrieval" / "results.jsonl"),
        "preflight_sha256": sha256(preflight_path),
        "results_sha256": sha256(results_path),
        "pool_count": len(results),
        "candidate_pairs": sum(row["candidate_count"] for row in results.values()),
        "pair_tokens": pair_tokens_total,
        "pair_token_upper_bound": upper_tokens,
        "wall_seconds": sum(float(row.get("wall_ms", 0.0)) for row in results.values()) / 1000,
        "gpu": gpu_diagnostics,
        "agent_follow_up_searches_reranked": False,
    }
    atomic_json_once(manifest_path, manifest)
    return manifest


def _load_planned_initial_reranks(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    if not static["retrieval"].get("shared_initial_reranker"):
        raise ValueError("the frozen run did not select shared initial reranking")
    directory = run_root / "planned-retrieval" / "reranked"
    results_path = directory / "results.jsonl"
    manifest_path = directory / "reranked.json"
    preflight_path = directory / "preflight.json"
    planned_path = run_root / "planned-retrieval" / "results.jsonl"
    if not results_path.is_file() or not manifest_path.is_file() or not preflight_path.is_file():
        raise ValueError("run shared initial reranking before exact preflight or generation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_unique_jsonl(results_path, key="slot_id")
    expected = {
        f"{case['case_id']}::{pool}" for case in inputs for pool in ("original_query", "decomposed")
    }
    if (
        manifest.get("state") != "complete"
        or manifest.get("static_configuration_sha256") != canonical_hash(static)
        or manifest.get("planned_retrieval_sha256") != sha256(planned_path)
        or manifest.get("preflight_sha256") != sha256(preflight_path)
        or manifest.get("results_sha256") != sha256(results_path)
        or set(rows) != expected
    ):
        raise ValueError("shared initial reranker artifacts do not match the frozen run")
    planned = _load_planned_retrieval(run_root, static, inputs)
    for case in inputs:
        for pool, key in (
            ("original_query", "original_query_candidate_result"),
            ("decomposed", "candidate_result"),
        ):
            slot_id = f"{case['case_id']}::{pool}"
            candidate = _search_result_from_json(planned[case["case_id"]][key])
            result = rows[slot_id]
            if result.get("candidate_sha256") != canonical_hash(candidate.model_dump(mode="json")):
                raise ValueError("shared initial reranker result differs from its candidate pool")
            _search_result_from_json(result["result"])
    return rows


def _rank_prepared_retrieval(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    """Rerank each frozen top-40 candidate list once and persist each result."""
    if static["retrieval"].get("shared_initial_reranker"):
        return _rank_planned_initial_retrieval(run_root, static, inputs)
    if not static["retrieval"].get("reranker"):
        raise ValueError("select the fixed-rag-reranked arm before reranker preparation")
    store, retrieval = _load_prepared(run_root, static, include_reranked=False)
    prepared_dir = run_root / "prepared"
    retrieval_path = prepared_dir / "retrieval.jsonl"
    reranking_dir = prepared_dir / "reranked"
    reranking_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    results_path = reranking_dir / "results.jsonl"
    attempts_path = reranking_dir / "attempts.jsonl"
    failures_path = reranking_dir / "failures.jsonl"
    manifest_path = reranking_dir / "reranked.json"
    if manifest_path.exists():
        raise ValueError("reranker manifest already exists; do not overwrite a prepared run")

    from page_cite_rag.hybrid import rerank_search_result
    from page_cite_rag.reranker_server import (
        DEFAULT_MODEL_PATH,
        MAX_PAIR_TOKENS,
        NativeRerankerServer,
    )

    result_rows = load_unique_jsonl(results_path)
    attempt_rows = load_unique_jsonl(attempts_path)
    if set(result_rows) - set(retrieval) or set(attempt_rows) - set(retrieval):
        raise ValueError("prepared reranker artifacts contain unknown cases")
    reject_interrupted_attempts(attempt_rows, result_rows, "prepared reranker")
    missing = [case for case in inputs if case["case_id"] not in result_rows]
    progress_path = run_root.parent / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    usage = progress.setdefault("usage", {})
    run_key = str(run_root.relative_to(run_root.parent))
    confirmed_before = int(usage.get("reranker_pair_tokens_confirmed", 0))
    unknown_before = int(usage.get("reranker_pair_tokens_unknown_upper_bound", 0))
    local_completed = sum(sum(row.get("pair_tokens", [])) for row in result_rows.values())
    accounted_by_run = usage.setdefault("reranker_pair_tokens_by_run", {})
    accounted_for_run = int(accounted_by_run.get(run_key, 0))
    if accounted_for_run > local_completed:
        raise ValueError("global reranker ledger exceeds this run's persisted results")
    confirmed_before += local_completed - accounted_for_run
    accounted_by_run[run_key] = local_completed
    usage["reranker_pair_tokens_confirmed"] = confirmed_before
    inflight_by_run = usage.setdefault("reranker_pair_tokens_inflight_by_run", {})
    inflight_by_run.pop(run_key, None)
    usage["reranker_pair_tokens_inflight_upper_bound"] = sum(
        int(value) for value in inflight_by_run.values()
    )
    write_state_json(progress_path, progress)
    remaining_bound = 0
    for case in missing:
        candidate = _search_result_from_json(retrieval[case["case_id"]]["candidate_result"])
        candidate_count = len(candidate.context_groups) or len(candidate.hits)
        remaining_bound += min(40, candidate_count) * MAX_PAIR_TOKENS
    if (
        confirmed_before
        + unknown_before
        + usage["reranker_pair_tokens_inflight_upper_bound"]
        + remaining_bound
        > MAX_RERANKER_PAIR_TOKENS
    ):
        raise ValueError("prepared reranker would exceed the cumulative pair-token budget")

    started = time.monotonic()
    gpu = None
    if missing:
        with NativeRerankerServer(
            DEFAULT_MODEL_PATH,
            GPU_UUID,
            max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C,
        ) as ranker:
            for case in missing:
                case_id = case["case_id"]
                row = retrieval[case_id]
                candidates = _search_result_from_json(row["candidate_result"])
                candidate_sha = canonical_hash(candidates.model_dump(mode="json"))
                candidate_count = len(candidates.context_groups) or len(candidates.hits)
                append_jsonl_record(
                    attempts_path,
                    {
                        "case_id": case_id,
                        "state": "started",
                        "candidate_count": candidate_count,
                        "candidate_sha256": candidate_sha,
                    },
                )
                request_started = time.monotonic()
                in_flight_upper = min(40, candidate_count) * MAX_PAIR_TOKENS
                inflight_by_run[run_key] = in_flight_upper
                usage["reranker_pair_tokens_inflight_upper_bound"] = sum(
                    int(value) for value in inflight_by_run.values()
                )
                write_state_json(progress_path, progress)
                try:
                    ranked = rerank_search_result(candidates, ranker, top_k=TOP_K, store=store)
                except Exception as exc:
                    pair_tokens = list(ranker.last_pair_tokens)
                    pair_upper = (
                        sum(pair_tokens)
                        + max(0, candidate_count - len(pair_tokens)) * MAX_PAIR_TOKENS
                    )
                    append_jsonl_record(
                        failures_path,
                        {
                            "case_id": case_id,
                            "state": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                            "candidate_count": candidate_count,
                            "candidate_sha256": candidate_sha,
                            "pair_tokens_known": pair_tokens,
                            "pair_tokens_unknown_upper_bound": pair_upper,
                            "batch_sizes": list(ranker.last_batch_sizes),
                            "attempted_batch_count": ranker.last_attempted_batch_count,
                            "completed_batch_count": ranker.last_completed_batch_count,
                            "wall_ms": (time.monotonic() - request_started) * 1000,
                            "gpu": ranker.diagnostics(),
                        },
                    )
                    inflight_by_run.pop(run_key, None)
                    usage["reranker_pair_tokens_inflight_upper_bound"] = sum(
                        int(value) for value in inflight_by_run.values()
                    )
                    usage["reranker_pair_tokens_unknown_upper_bound"] = unknown_before + pair_upper
                    unknown_by_run = usage.setdefault("reranker_pair_tokens_unknown_by_run", {})
                    unknown_by_run[run_key] = pair_upper
                    write_state_json(progress_path, progress)
                    raise
                pair_tokens = list(ranker.last_pair_tokens)
                record = {
                    "case_id": case_id,
                    "candidate_sha256": candidate_sha,
                    "result": ranked.model_dump(mode="json"),
                    "pair_tokens": pair_tokens,
                    "batch_sizes": list(ranker.last_batch_sizes),
                    "tokenization_ms": ranker.last_tokenization_ms,
                    "forward_ms": ranker.last_forward_ms,
                    "total_ms": ranker.last_total_ms,
                    "wall_ms": (time.monotonic() - request_started) * 1000,
                }
                append_jsonl_record(results_path, record)
                result_rows[case_id] = record
                current_local_total = sum(
                    sum(item.get("pair_tokens", [])) for item in result_rows.values()
                )
                usage["reranker_pair_tokens_confirmed"] = (
                    confirmed_before + current_local_total - local_completed
                )
                accounted_by_run[run_key] = current_local_total
                inflight_by_run.pop(run_key, None)
                usage["reranker_pair_tokens_inflight_upper_bound"] = sum(
                    int(value) for value in inflight_by_run.values()
                )
                progress.setdefault("stages", {}).setdefault("reranker_preparation", {})[
                    "completed_cases"
                ] = len(result_rows)
                write_state_json(progress_path, progress)
            gpu = ranker.diagnostics()
    if len(result_rows) != len(inputs):
        raise ValueError("prepared reranker did not produce one row per frozen input")
    manifest = {
        "state": "complete",
        "static_configuration_sha256": canonical_hash(static),
        "prepared_retrieval_sha256": sha256(retrieval_path),
        "results_sha256": sha256(results_path),
        "case_count": len(result_rows),
        "pair_tokens": sum(sum(row.get("pair_tokens", [])) for row in result_rows.values()),
        "wall_seconds": time.monotonic() - started,
        "gpu": gpu,
        "reused_completed_results_without_new_gpu_work": not missing,
    }
    atomic_json_once(manifest_path, manifest)
    progress.setdefault("stages", {}).setdefault("reranker_preparation", {}).update(
        {"status": "complete", "case_count": len(result_rows), "pending_cases": 0}
    )
    progress["usage"]["reranker_pair_tokens_inflight_upper_bound"] = sum(
        int(value)
        for value in progress["usage"].get("reranker_pair_tokens_inflight_by_run", {}).values()
    )
    write_state_json(progress_path, progress)
    return manifest


def _search_result_from_json(value: dict) -> SearchResult:
    """Validate a JSON-decoded result while preserving strict nested schemas."""
    return SearchResult.model_validate_json(
        json.dumps(value, ensure_ascii=False, allow_nan=False), strict=True
    )


def _search_result_to_json(result: SearchResult) -> dict:
    """Persist Docling context groups alongside hits; they are excluded by default."""
    value = result.model_dump(mode="json")
    if result.context_groups:
        value["context_groups"] = [group.model_dump(mode="json") for group in result.context_groups]
    return value


def _expand_cached_hybrid_candidates(
    result: SearchResult,
    store: SQLiteDocumentStore,
    *,
    top_k: int,
    constant: int,
) -> SearchResult:
    """Reconstruct the cached lexical/dense RRF pool from its ordered branch IDs."""
    if result.method != "hybrid" or result.context_groups or not result.branches:
        return result
    if (
        type(top_k) is not int
        or top_k < len(result.hits)
        or type(constant) is not int
        or constant < 1
        or {branch.name for branch in result.branches} != {"lexical", "dense"}
    ):
        raise ValueError("cached hybrid branches cannot reconstruct the requested candidate pool")
    branch_results = []
    for branch in result.branches:
        hits = []
        for rank, block_id in enumerate(branch.block_ids, 1):
            block = store.get_block(result.document_id, block_id)
            if block is None or block.document_id != result.document_id:
                raise ValueError("cached hybrid branch references a missing scoped block")
            hits.append(SearchHit(rank=rank, score=0.0, block=block))
        branch_results.append(
            SearchResult(
                document_id=result.document_id,
                index_id=branch.index_id,
                method="hybrid",
                query=result.query,
                hits=tuple(hits),
                elapsed_ms=branch.elapsed_ms,
            )
        )
    hits = reciprocal_rank_fusion(branch_results, top_k=top_k, constant=constant)
    original_ids = [hit.block.block_id for hit in result.hits]
    if [hit.block.block_id for hit in hits[: len(original_ids)]] != original_ids:
        raise ValueError("reconstructed cached RRF does not preserve the frozen top-k order")
    return SearchResult(
        document_id=result.document_id,
        index_id=canonical_hash(
            {
                "source_index_id": result.index_id,
                "top_k": top_k,
                "rrf_constant": constant,
                "reconstruction": "ordered-branch-ids/v1",
            }
        ),
        method="hybrid",
        query=result.query,
        hits=hits,
        elapsed_ms=result.elapsed_ms,
        branches=result.branches,
    )


def _limit_search_result_to_context_groups(result: SearchResult, top_k: int) -> SearchResult:
    """Keep complete source mappings for selected retrieval chunks."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("context-group limit must be a positive integer")
    if not result.context_groups:
        return result.model_copy(update={"hits": result.hits[:top_k]})

    groups = result.context_groups[:top_k]
    ordered_ids = tuple(
        dict.fromkeys(block_id for group in groups for block_id in group.source_block_ids)
    )
    hits_by_id = {hit.block.block_id: hit for hit in result.hits}
    if len(hits_by_id) != len(result.hits) or set(ordered_ids) - hits_by_id.keys():
        raise ValueError("selected retrieval chunks have invalid or missing source blocks")
    hits = tuple(hits_by_id[block_id] for block_id in ordered_ids)
    return result.model_copy(
        update={
            "hits": tuple(
                hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(hits, 1)
            ),
            "context_groups": groups,
        }
    )


def _passage_ids_from_search(result: SearchResult, block_to_passage: dict[str, str]) -> list[str]:
    """Map the full candidate branches or admitted context blocks to QASPER passages."""
    block_ids = (
        [block_id for branch in result.branches for block_id in branch.block_ids]
        if result.branches
        else [hit.block.block_id for hit in result.hits]
    )
    return list(
        dict.fromkeys(
            block_to_passage[block_id] for block_id in block_ids if block_id in block_to_passage
        )
    )


def _search_snapshot(result: SearchResult, block_to_passage: dict[str, str]) -> dict:
    candidate_block_ids = (
        list(dict.fromkeys(block_id for branch in result.branches for block_id in branch.block_ids))
        if result.branches
        else [hit.block.block_id for hit in result.hits]
    )
    context_block_ids = [hit.block.block_id for hit in result.hits]
    return {
        "candidate_block_ids": candidate_block_ids,
        "candidate_ranked_block_ids": [hit.block.block_id for hit in result.hits],
        "candidate_passage_ids": _passage_ids_from_search(result, block_to_passage),
        "candidate_ids_sha256": canonical_hash(candidate_block_ids),
        "retrieval_elapsed_ms": result.elapsed_ms,
        "context_block_ids": context_block_ids,
        "context_passage_ids": list(
            dict.fromkeys(
                block_to_passage[hit.block.block_id]
                for hit in result.hits
                if hit.block.block_id in block_to_passage
            )
        ),
        "context_ids_sha256": canonical_hash(context_block_ids),
    }


def _set_context_snapshot(
    snapshot: dict, context_result: SearchResult, block_to_passage: dict[str, str]
) -> None:
    block_ids = [hit.block.block_id for hit in context_result.hits]
    snapshot["context_block_ids"] = block_ids
    snapshot["context_passage_ids"] = list(
        dict.fromkeys(
            block_to_passage[block_id] for block_id in block_ids if block_id in block_to_passage
        )
    )
    snapshot["context_ids_sha256"] = canonical_hash(block_ids)


def _rewrite_initial_cases(inputs: list[dict]) -> list[dict]:
    """Inspect the first case in each available question/document language direction."""
    selected = {}
    for case in sorted(inputs, key=lambda value: value["case_id"]):
        direction = case.get("direction")
        if isinstance(direction, str):
            selected.setdefault(direction, case)
    return [selected[direction] for direction in sorted(selected)]


def _validate_rewrite_record(case: dict, rewrite: dict, prior_reason: str | None) -> None:
    """Stop systematic request failures before dispatching further questions."""
    if rewrite.get("original_query") != case["question"]:
        raise ValueError("rewrite original question does not match the frozen input")
    lexical = rewrite.get("lexical_query")
    status = rewrite.get("status")
    if not isinstance(lexical, str) or not lexical.strip():
        raise ValueError("rewrite lexical query is missing")
    if rewrite.get("mode", "keyword_expansion") == "full_sentence":
        if status == "rewritten" and lexical == case["question"].strip():
            raise ValueError("rewritten full sentence did not change the original query")
        if rewrite.get("version") != "full-sentence-document-language/v1":
            raise ValueError("full-sentence rewrite version differs from the frozen protocol")
    elif status == "rewritten" and not lexical.startswith(case["question"] + " "):
        raise ValueError("keyword expansion removed or altered the original question")
    if status != "rewritten" and lexical != case["question"]:
        if not (rewrite.get("mode") == "full_sentence" and lexical == case["question"].strip()):
            raise ValueError("degraded rewrite did not preserve the original query")
    reason = rewrite.get("reason")
    if reason in {"generation_budget_mismatch", "model_identity_or_budget_invalid"}:
        raise ValueError(f"rewrite request or budget contract failed: {reason}")
    if reason == "generation_failure" and prior_reason == reason:
        raise ValueError("two consecutive rewrite model failures; stop before further dispatch")


def _restore_search_branch(store, document_id: str, query: str, trace, method: str) -> SearchResult:
    """Rebuild rank-only branch results from a persisted hybrid branch trace."""
    expected_name = "lexical" if method == "bm25" else "dense"
    if trace.name != expected_name or not trace.index_id:
        raise ValueError("prepared baseline has an invalid retrieval branch trace")
    hits = []
    seen = set()
    for rank, block_id in enumerate(trace.block_ids, 1):
        if block_id in seen:
            raise ValueError("prepared retrieval branch contains a duplicate block")
        seen.add(block_id)
        block = store.get_block(document_id, block_id)
        if block is None:
            raise ValueError("prepared retrieval branch references a missing source block")
        # RRF depends on rank, not the original BM25/cosine score.
        hits.append(SearchHit(rank=rank, score=0.0, block=block))
    return SearchResult(
        document_id=document_id,
        index_id=trace.index_id,
        method=method,
        query=query,
        hits=tuple(hits),
        elapsed_ms=trace.elapsed_ms,
    )


def _full_sentence_routed_variants(
    query: str,
    original_lexical: SearchResult,
    rewritten_lexical: SearchResult,
    original_dense: SearchResult,
    rewritten_dense: SearchResult,
    *,
    candidates: int = 20,
    constant: int = 60,
) -> dict[str, SearchResult]:
    """Build the three preregistered rewrite routes from shared branch results."""
    from page_cite_rag.hybrid import fuse_routed_hybrid_results

    return {
        "rewrite_bm25_only": fuse_routed_hybrid_results(
            query,
            [original_lexical, rewritten_lexical],
            original_dense,
            lexical_top_k=candidates,
            top_k=2 * candidates,
            constant=constant,
        ),
        "rewrite_dense_only": fuse_routed_hybrid_results(
            query,
            [original_lexical],
            rewritten_dense,
            lexical_top_k=candidates,
            top_k=2 * candidates,
            constant=constant,
        ),
        "rewrite_both": fuse_routed_hybrid_results(
            query,
            [original_lexical, rewritten_lexical],
            rewritten_dense,
            lexical_top_k=candidates,
            top_k=2 * candidates,
            constant=constant,
        ),
    }


def _run_full_sentence_retrieval_grid(
    run_root: Path,
    static: dict,
    inputs: list[dict],
    *,
    rewrite_initial_only: bool = False,
    allow_recorded_failures: bool = False,
) -> dict:
    """Run the frozen full-sentence rewrite routing grid without answer calls."""
    protocol = static["retrieval"].get("retrieval_grid", {}).get("protocol")
    if protocol != "full-sentence-rewrite-routing/v1":
        raise ValueError("full-sentence grid requires its frozen retrieval protocol")
    grid_dir = run_root / "retrieval-grid"
    grid_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    static_hash = canonical_hash(static)
    progress_path = grid_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("static_configuration_sha256") != static_hash:
            raise ValueError("full-sentence grid progress belongs to another configuration")
    else:
        progress = {
            "state": "in_progress",
            "protocol": protocol,
            "static_configuration_sha256": static_hash,
            "case_count": len(inputs),
            "rewrite_completed": 0,
            "rewrite_call_attempts": 0,
            "retrieval_completed": 0,
        }
        write_state_json(progress_path, progress)
    if len(inputs) > 240:
        raise ValueError("full-sentence rewrite calls exceed the frozen 240-case limit")
    final_path = grid_dir / "retrieval-grid.jsonl"
    if final_path.is_file():
        rows = read_jsonl(final_path)
        if [row.get("case_id") for row in rows] != [case["case_id"] for case in inputs]:
            raise ValueError("completed full-sentence grid differs from its frozen inputs")
        return {
            "state": "complete",
            "case_count": len(rows),
            "static_configuration_sha256": static_hash,
            "retrieval_grid_sha256": sha256(final_path),
            "resumed": True,
        }

    # A previous explicit continuation may process only never-started inputs;
    # interrupted requests remain unrecoverable and are rejected below.
    _ = allow_recorded_failures
    store, baseline_rows = _load_prepared(run_root, static, include_reranked=False)
    input_ids = [case["case_id"] for case in inputs]
    if set(input_ids) != set(baseline_rows):
        raise ValueError("prepared base retrieval does not match frozen grid input IDs")

    rewrite_path = grid_dir / "rewrites.jsonl"
    attempts_path = grid_dir / "rewrite-attempts.jsonl"
    rewrites = load_unique_jsonl(rewrite_path)
    attempts = load_unique_jsonl(attempts_path)
    if rewrites.keys() - set(input_ids) or attempts.keys() - set(input_ids):
        raise ValueError("rewrite records contain IDs outside the frozen input set")
    reject_interrupted_attempts(attempts, rewrites, "full-sentence rewrite")
    input_by_id = {case["case_id"]: case for case in inputs}
    previous_reason = None
    rewrite_cache = {}
    for case_id, row in rewrites.items():
        if (
            row.get("question_sha256")
            != hashlib.sha256(input_by_id[case_id]["question"].encode("utf-8")).hexdigest()
        ):
            raise ValueError("saved rewrite is for a different frozen question")
        _validate_rewrite_record(input_by_id[case_id], row["rewrite"], previous_reason)
        previous_reason = row["rewrite"].get("reason")
        signature = (row["question_sha256"], row.get("source_language") or "unknown")
        rewrite_cache.setdefault(signature, row["rewrite"])

    initial_cases = _rewrite_initial_cases(inputs)
    initial_ids = {case["case_id"] for case in initial_cases}
    ordered_cases = initial_cases + [case for case in inputs if case["case_id"] not in initial_ids]
    missing_rewrites = [case for case in ordered_cases if case["case_id"] not in rewrites]
    if rewrite_initial_only:
        missing_rewrites = [case for case in missing_rewrites if case["case_id"] in initial_ids]
    rewrite_attempt_count = len(attempts)
    if rewrite_attempt_count + len(missing_rewrites) > 240:
        raise ValueError("rewrite attempts exceed the frozen 240-call ceiling")
    rewrite_session = None
    if missing_rewrites:
        from page_cite_rag.query import QueryRewriteConfig, QueryRewriter

        started = time.monotonic()
        call_count = input_tokens = output_tokens = 0
        with LlamaModelClient(
            _llama_config(static, num_predict=128),
            budget_policy="exact",
            recover_output_errors=False,
            sampling_profile=static["answer"]["sampling_profile"],
            sampling_seed=42,
        ) as model:
            rewriters = {}
            for case in missing_rewrites:
                case_id = case["case_id"]
                question = case["question"]
                source_language = case.get("source_language")
                signature = (
                    hashlib.sha256(question.encode("utf-8")).hexdigest(),
                    source_language or "unknown",
                )
                if signature in rewrite_cache:
                    rewrite = {**rewrite_cache[signature], "stats": None, "cache_hit": True}
                    attempted = False
                    shared_from = next(
                        (
                            prior_id
                            for prior_id, prior in rewrites.items()
                            if (
                                prior.get("question_sha256"),
                                prior.get("source_language") or "unknown",
                            )
                            == signature
                        ),
                        None,
                    )
                else:
                    config = QueryRewriteConfig.for_document_language(
                        source_language, mode="full_sentence"
                    )
                    rewriter = rewriters.get(config.source_language)
                    if rewriter is None:
                        rewriter = QueryRewriter(model, model.config, config)
                        rewriters[config.source_language] = rewriter
                    if len(question) <= config.max_question_chars:
                        if rewrite_attempt_count >= 240:
                            raise ValueError("rewrite attempts exceed 240")
                        append_jsonl_record(
                            attempts_path,
                            {
                                "case_id": case_id,
                                "question_sha256": signature[0],
                                "source_language": signature[1],
                                "state": "started",
                            },
                        )
                        attempts[case_id] = {"case_id": case_id, "state": "started"}
                        rewrite_attempt_count += 1
                        attempted = True
                    else:
                        attempted = False
                    result = rewriter.rewrite(question)
                    rewrite = result.model_dump(mode="json")
                    shared_from = None
                    if result.stats is not None:
                        call_count += 1
                        input_tokens += result.stats.prompt_eval_count
                        output_tokens += result.stats.eval_count
                    rewrite_cache[signature] = rewrite
                row = {
                    "case_id": case_id,
                    "question_sha256": signature[0],
                    "source_language": signature[1],
                    "rewrite": rewrite,
                    "llm_call_attempted": attempted,
                    "shared_from_case_id": shared_from,
                }
                append_jsonl_record(rewrite_path, row)
                rewrites[case_id] = row
                _validate_rewrite_record(case, rewrite, previous_reason)
                previous_reason = rewrite.get("reason")
                progress.update(
                    {
                        "rewrite_completed": len(rewrites),
                        "rewrite_call_attempts": rewrite_attempt_count,
                    }
                )
                write_state_json(progress_path, progress)
            rewrite_session = {
                "confirmed_model_calls": call_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "wall_seconds": time.monotonic() - started,
                "gpu": _safe_model_diagnostics(model),
            }

    rewrite_rows = list(rewrites.values())
    rewrite_usage = {
        "confirmed_model_calls": sum(
            row["rewrite"].get("stats") is not None for row in rewrite_rows
        ),
        "call_attempts": len(attempts),
        "input_tokens": sum(
            int((row["rewrite"].get("stats") or {}).get("prompt_eval_count") or 0)
            for row in rewrite_rows
        ),
        "output_tokens": sum(
            int((row["rewrite"].get("stats") or {}).get("eval_count") or 0) for row in rewrite_rows
        ),
        "cache_shared_cases": sum(
            row.get("shared_from_case_id") is not None for row in rewrite_rows
        ),
        "last_invocation": rewrite_session,
    }
    write_state_json(
        grid_dir / "rewrite-usage.json",
        {"static_configuration_sha256": static_hash, "usage": rewrite_usage},
    )

    retrieval_path = grid_dir / "rewritten-retrieval.jsonl"
    retrieval_attempts_path = grid_dir / "retrieval-attempts.jsonl"
    retrievals = load_unique_jsonl(retrieval_path)
    retrieval_attempts = load_unique_jsonl(retrieval_attempts_path)
    if retrievals.keys() - set(input_ids) or retrieval_attempts.keys() - set(input_ids):
        raise ValueError("retrieval records contain IDs outside the frozen input set")
    reject_interrupted_attempts(retrieval_attempts, retrievals, "rewritten retrieval")
    missing_retrieval = [case for case in inputs if case["case_id"] not in retrievals]
    if rewrite_initial_only:
        missing_retrieval = [case for case in missing_retrieval if case["case_id"] in initial_ids]
    if len(retrieval_attempts) + len(missing_retrieval) > 240:
        raise ValueError("rewritten dense-query attempts exceed the frozen 240-case ceiling")
    query_progress = {"stage": "full_sentence_routed_query_embedding", "case_id": None}
    query_session = None
    if missing_retrieval:
        from page_cite_rag.application import document_retriever_parts
        from page_cite_rag.hybrid import fuse_routed_hybrid_results

        started = time.monotonic()
        with _preparation_embedding_session(run_root, static, query_progress) as (
            server,
            counting_transport,
        ):
            encoder = NativeEmbeddingEncoder(
                counting_transport,
                DEFAULT_MODEL_DIR,
                max_tokens=static["embedding_input"]["document_max_tokens"],
                query_max_tokens=static["embedding_input"]["query_max_tokens"],
                batch_size=4,
            )
            query_prefix = encoder.identity["query_prompt"]
            counting_transport.query_prefix = query_prefix
            candidates = static["retrieval"]["candidates_per_branch"]
            rrf_constant = static["retrieval"]["rrf_constant"]
            for case in missing_retrieval:
                case_id = case["case_id"]
                query_progress["case_id"] = case_id
                base = baseline_rows[case_id]
                baseline_candidate = _search_result_from_json(base["candidate_result"])
                baseline_context = _search_result_from_json(base["result"])
                trace_by_name = {trace.name: trace for trace in baseline_candidate.branches}
                if set(trace_by_name) != {"lexical", "dense"}:
                    raise ValueError(
                        "prepared hybrid result lacks exact BM25 and dense branch traces"
                    )
                original_lexical = _restore_search_branch(
                    store, base["document_id"], case["question"], trace_by_name["lexical"], "bm25"
                )
                original_dense = _restore_search_branch(
                    store, base["document_id"], case["question"], trace_by_name["dense"], "dense"
                )
                check_original = fuse_routed_hybrid_results(
                    case["question"],
                    [original_lexical],
                    original_dense,
                    lexical_top_k=candidates,
                    top_k=2 * candidates,
                    constant=rrf_constant,
                )
                if [hit.block.block_id for hit in check_original.hits] != [
                    hit.block.block_id for hit in baseline_candidate.hits
                ]:
                    raise ValueError(
                        "persisted baseline branch traces do not reproduce its RRF ranking"
                    )

                rewrite = rewrites[case_id]["rewrite"]
                rewritten_query = rewrite["lexical_query"]
                if not isinstance(rewritten_query, str) or not rewritten_query.strip():
                    raise ValueError("full-sentence rewrite query is missing")
                append_jsonl_record(
                    retrieval_attempts_path,
                    {
                        "case_id": case_id,
                        "question_sha256": hashlib.sha256(
                            case["question"].encode("utf-8")
                        ).hexdigest(),
                        "rewritten_query_sha256": hashlib.sha256(
                            rewritten_query.encode("utf-8")
                        ).hexdigest(),
                        "dense_route_status": "started",
                        "state": "started_before_backend_dispatch",
                    },
                )
                retrieval_attempts[case_id] = {"case_id": case_id, "state": "started"}
                if rewritten_query == case["question"]:
                    rewritten_lexical = original_lexical
                    rewritten_dense = original_dense
                    rewritten_query_tokens = 0
                    embedding_tokens = 0
                    dense_route_status = "same_query_reused_original_branch"
                else:
                    from page_cite_rag.retrieval import BM25Retriever

                    lexical = BM25Retriever(store, base["document_id"])
                    rewritten_lexical = lexical.search(rewritten_query, top_k=candidates)
                    rewritten_query_tokens = len(server.token_ids(query_prefix + rewritten_query))
                    if rewritten_query_tokens > encoder.identity["query_max_tokens"]:
                        embedding_tokens = 0
                        rewritten_dense = original_dense
                        dense_route_status = "query_token_limit_fallback_to_original"
                    else:
                        _, dense = document_retriever_parts(
                            store,
                            base["document_id"],
                            "hybrid",
                            encoder=encoder,
                            versioned_dense=True,
                        )
                        if dense is None:
                            raise ValueError("hybrid retrieval requires the frozen dense branch")
                        rewritten_dense = dense.search(rewritten_query, top_k=candidates)
                        embedding_tokens = rewritten_query_tokens
                        dense_route_status = "rewritten_dense_query"

                variants = {
                    "original_hybrid": baseline_candidate,
                    **_full_sentence_routed_variants(
                        case["question"],
                        original_lexical,
                        rewritten_lexical,
                        original_dense,
                        rewritten_dense,
                        candidates=candidates,
                        constant=rrf_constant,
                    ),
                }
                results = {}
                for name, candidate_result in variants.items():
                    context_result = (
                        baseline_context
                        if name == "original_hybrid"
                        else _limit_search_result_to_context_groups(candidate_result, TOP_K)
                    )
                    results[name] = {
                        "candidate_result": candidate_result.model_dump(mode="json"),
                        "context_result": context_result.model_dump(mode="json"),
                        "snapshot": _search_snapshot(candidate_result, base["block_to_passage_id"]),
                    }
                    _set_context_snapshot(
                        results[name]["snapshot"],
                        context_result,
                        base["block_to_passage_id"],
                    )
                row = {
                    "case_id": case_id,
                    "document_cluster": case["document_cluster"],
                    "direction": case["direction"],
                    "dataset": case["dataset"],
                    "question_language": case["question_language"],
                    "source_language": case.get("source_language"),
                    "question_sha256": hashlib.sha256(case["question"].encode("utf-8")).hexdigest(),
                    "rewrite": rewrite,
                    "rewrite_llm_call_attempted": rewrites[case_id]["llm_call_attempted"],
                    "rewrite_shared_from_case_id": rewrites[case_id]["shared_from_case_id"],
                    "rewrite_elapsed_ms": rewrite.get("elapsed_ms"),
                    "rewritten_query_tokens": rewritten_query_tokens,
                    "query_embedding_input_tokens": embedding_tokens,
                    "dense_route_status": dense_route_status,
                    "lexical_query": rewritten_query,
                    "branch_inputs": {
                        "original_bm25_query": case["question"],
                        "rewritten_bm25_query": rewritten_query,
                        "original_bm25_block_ids": list(
                            hit.block.block_id for hit in original_lexical.hits
                        ),
                        "rewritten_bm25_block_ids": list(
                            hit.block.block_id for hit in rewritten_lexical.hits
                        ),
                        "original_dense_block_ids": list(
                            hit.block.block_id for hit in original_dense.hits
                        ),
                        "rewritten_dense_block_ids": list(
                            hit.block.block_id for hit in rewritten_dense.hits
                        ),
                    },
                    "variants": results,
                }
                append_jsonl_record(retrieval_path, row)
                retrievals[case_id] = row
                progress.update(
                    {
                        "retrieval_completed": len(retrievals),
                        "query_embedding_tokens_this_invocation": counting_transport.query_tokens,
                        "query_embedding_inputs_this_invocation": counting_transport.embed_inputs,
                    }
                )
                write_state_json(progress_path, progress)
            query_session = {
                "query_embedding_tokens": counting_transport.query_tokens,
                "query_embedding_inputs": counting_transport.embed_inputs,
                "embedding_requests": _embedding_request_accounting(counting_transport),
                "gpu": server.diagnostics(),
                "wall_seconds": time.monotonic() - started,
            }

    retrieval_rows = list(retrievals.values())
    query_usage = {
        "query_embedding_tokens": sum(
            row["query_embedding_input_tokens"] for row in retrieval_rows
        ),
        "query_embedding_cases": sum(
            row["dense_route_status"] == "rewritten_dense_query" for row in retrieval_rows
        ),
        "dense_fallback_cases": sum(
            row["dense_route_status"] == "query_token_limit_fallback_to_original"
            for row in retrieval_rows
        ),
        "last_invocation": query_session,
    }
    write_state_json(
        grid_dir / "query-embedding-usage.json",
        {"static_configuration_sha256": static_hash, "usage": query_usage},
    )
    if (
        query_usage["query_embedding_tokens"]
        > static["preparation_budget"]["global_query_embedding_input_tokens_max"]
    ):
        raise ValueError("rewritten query embedding usage exceeded the frozen global limit")

    if rewrite_initial_only:
        if not initial_ids.issubset(rewrites) or not initial_ids.issubset(retrievals):
            raise ValueError("four-direction rewrite and route inspection did not complete")
        return {
            "state": "four_direction_rewrite_route_inspection_complete",
            "case_ids": [case["case_id"] for case in initial_cases],
            "rewrites_completed": len(rewrites),
            "retrieval_routes_completed": len(retrievals),
            "rewrite_usage": rewrite_usage,
            "query_embedding_usage": query_usage,
        }
    if len(retrievals) != len(inputs):
        raise ValueError("full-sentence retrieval grid is incomplete; preserve its checkpoint")

    grid_rows = []
    for case in inputs:
        result = retrievals[case["case_id"]]
        grid_rows.append(
            {
                "case_id": case["case_id"],
                "document_cluster": case["document_cluster"],
                "direction": case["direction"],
                "dataset": case["dataset"],
                "rewrite_status": result["rewrite"]["status"],
                "rewrite_reason": result["rewrite"]["reason"],
                "rewrite_elapsed_ms": result["rewrite_elapsed_ms"],
                "rewrite_llm_call_attempted": result["rewrite_llm_call_attempted"],
                "rewrite_shared_from_case_id": result["rewrite_shared_from_case_id"],
                "rewrite_stats": result["rewrite"].get("stats"),
                "rewritten_query_tokens": result["rewritten_query_tokens"],
                "query_embedding_input_tokens": result["query_embedding_input_tokens"],
                "dense_route_status": result["dense_route_status"],
                "branch_inputs": result["branch_inputs"],
                "variants": {name: value["snapshot"] for name, value in result["variants"].items()},
                "candidate_results": {
                    name: value["candidate_result"] for name, value in result["variants"].items()
                },
                "context_results": {
                    name: value["context_result"] for name, value in result["variants"].items()
                },
            }
        )
    atomic_jsonl_once(final_path, grid_rows)
    final = {
        "state": "complete",
        "case_count": len(grid_rows),
        "static_configuration_sha256": static_hash,
        "retrieval_grid_sha256": sha256(final_path),
        "rewrite_usage": rewrite_usage,
        "query_embedding_usage": query_usage,
        "variant_names": [
            "original_hybrid",
            "rewrite_bm25_only",
            "rewrite_dense_only",
            "rewrite_both",
        ],
    }
    progress.update(
        {
            "state": "complete",
            "rewrite_completed": len(rewrites),
            "rewrite_call_attempts": len(attempts),
            "retrieval_completed": len(retrievals),
            "retrieval_grid_sha256": final["retrieval_grid_sha256"],
        }
    )
    write_state_json(progress_path, progress)
    atomic_json_once(grid_dir / "retrieval-grid-result.json", final)
    return final


def _run_retrieval_grid(
    run_root: Path,
    static: dict,
    inputs: list[dict],
    *,
    rewrite_initial_only: bool = False,
    allow_recorded_failures: bool = False,
) -> dict:
    """Run the frozen rewrite × reranker retrieval-only comparison, resumably."""
    if not static["retrieval"].get("retrieval_grid"):
        raise ValueError("prepare this run with --retrieval-grid before running its grid")
    if static["retrieval"]["retrieval_grid"].get("protocol") == "full-sentence-rewrite-routing/v1":
        return _run_full_sentence_retrieval_grid(
            run_root,
            static,
            inputs,
            rewrite_initial_only=rewrite_initial_only,
            allow_recorded_failures=allow_recorded_failures,
        )
    grid_dir = run_root / "retrieval-grid"
    grid_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    static_hash = canonical_hash(static)
    progress_path = grid_dir / "progress.json"
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("static_configuration_sha256") != static_hash:
            raise ValueError("retrieval-grid progress belongs to another configuration")
    else:
        write_state_json(
            progress_path,
            {
                "state": "in_progress",
                "static_configuration_sha256": static_hash,
                "case_count": len(inputs),
                "rewrite_completed": 0,
                "rewrite_call_attempts": 0,
                "rewritten_retrieval_completed": 0,
                "rewritten_reranking_completed": 0,
            },
        )
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
    final_path = grid_dir / "retrieval-grid.jsonl"
    if final_path.exists():
        rows = read_jsonl(final_path)
        if len(rows) != len(inputs) or [row.get("case_id") for row in rows] != [
            case["case_id"] for case in inputs
        ]:
            raise ValueError("completed retrieval-grid artifact differs from frozen inputs")
        return {
            "state": "complete",
            "case_count": len(rows),
            "static_configuration_sha256": static_hash,
            "retrieval_grid_sha256": sha256(final_path),
            "resumed": True,
        }

    store, baseline_rows = _load_prepared(run_root, static, include_reranked=False)
    input_ids = [case["case_id"] for case in inputs]
    if set(input_ids) != set(baseline_rows):
        raise ValueError("prepared base retrieval does not match frozen grid input IDs")

    from page_cite_rag.hybrid import rerank_search_result
    from page_cite_rag.reranker_server import (
        DEFAULT_MODEL_PATH,
        MAX_PAIR_TOKENS,
        NativeRerankerServer,
    )

    base_reranked_path = grid_dir / "base-reranked.jsonl"
    base_attempts_path = grid_dir / "base-reranker-attempts.jsonl"
    base_failures_path = grid_dir / "base-reranker-failures.jsonl"
    base_reranked = load_unique_jsonl(base_reranked_path)
    base_attempts = load_unique_jsonl(base_attempts_path)
    if base_reranked.keys() - set(input_ids) or base_attempts.keys() - set(input_ids):
        raise ValueError("base reranker records contain unknown case IDs")
    reject_interrupted_attempts(base_attempts, base_reranked, "base reranker")
    missing_base_reranks = [case for case in inputs if case["case_id"] not in base_reranked]
    pair_token_limit = 200_000_000
    known_prior_pair_tokens = 353
    failed_attempt_pair_token_upper_bound = 40 * MAX_PAIR_TOKENS
    maximum_base_pair_tokens = 0
    for case in missing_base_reranks:
        candidate = _search_result_from_json(baseline_rows[case["case_id"]]["candidate_result"])
        pair_count = (
            len(candidate.context_groups) if candidate.context_groups else len(candidate.hits)
        )
        maximum_base_pair_tokens += min(40, pair_count) * MAX_PAIR_TOKENS
    maximum_rewritten_pair_tokens = len(inputs) * 40 * MAX_PAIR_TOKENS
    base_completed_pair_tokens = sum(sum(row["pair_tokens"]) for row in base_reranked.values())
    projected_pair_tokens = (
        known_prior_pair_tokens
        + failed_attempt_pair_token_upper_bound
        + base_completed_pair_tokens
        + maximum_base_pair_tokens
        + maximum_rewritten_pair_tokens
    )
    if projected_pair_tokens > pair_token_limit:
        raise ValueError("retrieval-grid reranker preflight exceeds the frozen pair-token ceiling")
    if missing_base_reranks:
        rerank_started = time.monotonic()
        with NativeRerankerServer(
            DEFAULT_MODEL_PATH, GPU_UUID, max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C
        ) as ranker:
            for case in missing_base_reranks:
                case_id = case["case_id"]
                candidate = _search_result_from_json(baseline_rows[case_id]["candidate_result"])
                append_jsonl_record(
                    base_attempts_path,
                    {
                        "case_id": case_id,
                        "state": "started",
                        "candidate_count": (
                            len(candidate.context_groups)
                            if candidate.context_groups
                            else len(candidate.hits)
                        ),
                        "candidate_sha256": canonical_hash(candidate.model_dump(mode="json")),
                    },
                )
                saved["base_reranking_started"] = case_id
                write_state_json(progress_path, saved)
                case_started = time.monotonic()
                try:
                    result = rerank_search_result(candidate, ranker, top_k=TOP_K, store=store)
                except Exception as exc:
                    try:
                        gpu = ranker.diagnostics()
                    except Exception:
                        gpu = {"state": "diagnostics_unavailable"}
                    append_jsonl_record(
                        base_failures_path,
                        {
                            "case_id": case_id,
                            "state": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                            "pair_tokens": list(ranker.last_pair_tokens),
                            "batch_sizes": list(ranker.last_batch_sizes),
                            "attempted_batch_count": ranker.last_attempted_batch_count,
                            "completed_batch_count": ranker.last_completed_batch_count,
                            "tokenization_ms": ranker.last_tokenization_ms,
                            "forward_ms": ranker.last_forward_ms,
                            "wall_ms": (time.monotonic() - case_started) * 1000,
                            "gpu": gpu,
                        },
                    )
                    saved.update(
                        {"state": "stopped_on_base_reranker_failure", "failed_case": case_id}
                    )
                    write_state_json(progress_path, saved)
                    raise
                pair_tokens = list(ranker.last_pair_tokens)
                base_completed_pair_tokens += sum(pair_tokens)
                if (
                    known_prior_pair_tokens
                    + failed_attempt_pair_token_upper_bound
                    + base_completed_pair_tokens
                    + maximum_rewritten_pair_tokens
                    > pair_token_limit
                ):
                    raise ValueError(
                        "retrieval-grid reranker exceeded the frozen pair-token ceiling"
                    )
                record = {
                    "case_id": case_id,
                    "result": result.model_dump(mode="json"),
                    "pair_tokens": pair_tokens,
                    "tokenization_ms": ranker.last_tokenization_ms,
                    "forward_ms": ranker.last_forward_ms,
                    "total_ms": ranker.last_total_ms,
                    "wall_ms": (time.monotonic() - case_started) * 1000,
                    "batch_sizes": list(ranker.last_batch_sizes),
                }
                append_jsonl_record(base_reranked_path, record)
                base_reranked[case_id] = record
                saved.update(
                    {
                        "base_reranking_completed": len(base_reranked),
                        "base_reranker_pair_tokens": base_completed_pair_tokens,
                    }
                )
                write_state_json(progress_path, saved)
            base_reranker_usage = {
                "known_prior_pair_tokens": known_prior_pair_tokens,
                "failed_setup_attempt_pair_tokens": "unknown",
                "failed_setup_attempt_pair_tokens_upper_bound": (
                    failed_attempt_pair_token_upper_bound
                ),
                "base_pair_tokens": base_completed_pair_tokens,
                "wall_seconds": time.monotonic() - rerank_started,
                "gpu": ranker.diagnostics(),
            }
        write_state_json(
            grid_dir / "base-reranker-usage.json",
            {"static_configuration_sha256": static_hash, "usage": base_reranker_usage},
        )
    else:
        base_reranker_usage = (
            json.loads((grid_dir / "base-reranker-usage.json").read_text(encoding="utf-8"))["usage"]
            if (grid_dir / "base-reranker-usage.json").is_file()
            else {
                "known_prior_pair_tokens": known_prior_pair_tokens,
                "failed_setup_attempt_pair_tokens": "unknown",
                "failed_setup_attempt_pair_tokens_upper_bound": (
                    failed_attempt_pair_token_upper_bound
                ),
                "base_pair_tokens": base_completed_pair_tokens,
                "state": "reused_base_reranking",
            }
        )

    rewrite_path = grid_dir / "rewrites.jsonl"
    rewrites = load_unique_jsonl(rewrite_path)
    attempts_path = grid_dir / "rewrite-attempts.jsonl"
    rewrite_attempts = load_unique_jsonl(attempts_path)
    if rewrites.keys() - set(input_ids):
        raise ValueError("rewrite records contain cases outside the frozen inputs")
    if rewrite_attempts.keys() - set(input_ids):
        raise ValueError("rewrite attempts contain cases outside the frozen inputs")
    if rewrite_attempts.keys() - rewrites.keys():
        raise ValueError(
            "a rewrite call was interrupted; preserve its attempt and do not retry the same case"
        )
    rewrite_cache = {}
    input_by_id = {case["case_id"]: case for case in inputs}
    previous_reason = None
    for case_id, row in rewrites.items():
        _validate_rewrite_record(input_by_id[case_id], row["rewrite"], previous_reason)
        previous_reason = row["rewrite"].get("reason")
        signature = (row["question_sha256"], row["source_language"])
        rewrite_cache.setdefault(signature, row["rewrite"])

    initial_cases = _rewrite_initial_cases(inputs)
    initial_ids = {case["case_id"] for case in initial_cases}
    ordered_cases = initial_cases + [case for case in inputs if case["case_id"] not in initial_ids]
    missing_rewrites = [case for case in ordered_cases if case["case_id"] not in rewrites]
    if rewrite_initial_only:
        missing_rewrites = [case for case in missing_rewrites if case["case_id"] in initial_ids]
    attempted_calls = len(rewrite_attempts)
    if attempted_calls + len(missing_rewrites) > len(inputs) or len(inputs) > 240:
        raise ValueError("runtime rewrite attempts exceed the frozen stage-3 allowance")
    if missing_rewrites:
        from page_cite_rag.query import QueryRewriteConfig, QueryRewriter

        rewrite_started = time.monotonic()
        rewrite_input_tokens = rewrite_output_tokens = confirmed_calls = 0
        with LlamaModelClient(
            _llama_config(static, num_predict=128),
            budget_policy="exact",
            recover_output_errors=False,
            sampling_profile=static["answer"]["sampling_profile"],
            sampling_seed=42,
        ) as model:
            rewriters = {}
            for case in missing_rewrites:
                case_id = case["case_id"]
                source_language = case.get("source_language")
                language = source_language if isinstance(source_language, str) else None
                signature = (
                    hashlib.sha256(case["question"].encode("utf-8")).hexdigest(),
                    language or "unknown",
                )
                if signature in rewrite_cache:
                    rewrite = {**rewrite_cache[signature], "stats": None, "cache_hit": True}
                    llm_call_attempted = False
                    shared_from_case_id = next(
                        (
                            existing_id
                            for existing_id, existing in rewrites.items()
                            if (existing["question_sha256"], existing["source_language"])
                            == signature
                        ),
                        None,
                    )
                else:
                    config = QueryRewriteConfig.for_document_language(language)
                    rewriter = rewriters.get(config.source_language)
                    if rewriter is None:
                        rewriter = QueryRewriter(model, model.config, config)
                        rewriters[config.source_language] = rewriter
                    if len(case["question"]) <= config.max_question_chars:
                        if attempted_calls >= 240:
                            raise ValueError("runtime rewrite attempts exceed 240")
                        append_jsonl_record(
                            attempts_path,
                            {
                                "case_id": case_id,
                                "question_sha256": signature[0],
                                "source_language": signature[1],
                                "state": "started",
                            },
                        )
                        rewrite_attempts[case_id] = {"case_id": case_id, "state": "started"}
                        attempted_calls += 1
                        llm_call_attempted = True
                    else:
                        llm_call_attempted = False
                    rewrite_result = rewriter.rewrite(case["question"])
                    rewrite = rewrite_result.model_dump(mode="json")
                    shared_from_case_id = None
                    if rewrite_result.stats is not None:
                        confirmed_calls += 1
                        rewrite_input_tokens += rewrite_result.stats.prompt_eval_count
                        rewrite_output_tokens += rewrite_result.stats.eval_count
                    rewrite_cache[signature] = rewrite
                record = {
                    "case_id": case_id,
                    "question_sha256": signature[0],
                    "source_language": signature[1],
                    "rewrite": rewrite,
                    "llm_call_attempted": llm_call_attempted,
                    "shared_from_case_id": shared_from_case_id,
                }
                append_jsonl_record(rewrite_path, record)
                rewrites[case_id] = record
                previous = list(rewrites.values())[-2] if len(rewrites) > 1 else None
                _validate_rewrite_record(
                    case, rewrite, previous["rewrite"].get("reason") if previous else None
                )
                if rewrite_initial_only:
                    lexical = BM25Retriever(store, baseline_rows[case_id]["document_id"])
                    observed = lexical.search(rewrite["lexical_query"], top_k=20)
                    if observed.query != rewrite["lexical_query"]:
                        raise ValueError("BM25 did not receive the frozen lexical query")
                if attempted_calls > 240:
                    raise ValueError("runtime rewrite attempts exceed 240")
                saved.update(
                    {
                        "rewrite_completed": len(rewrites),
                        "rewrite_call_attempts": attempted_calls,
                    }
                )
                write_state_json(progress_path, saved)
            rewrite_gpu = _safe_model_diagnostics(model)
        rewrite_session = {
            "confirmed_model_calls": confirmed_calls,
            "input_tokens": rewrite_input_tokens,
            "output_tokens": rewrite_output_tokens,
            "wall_seconds": time.monotonic() - rewrite_started,
            "gpu": rewrite_gpu,
        }
    else:
        rewrite_session = None

    if rewrite_initial_only:
        if not initial_ids.issubset(rewrites):
            raise ValueError("initial rewrite inspection did not complete every language direction")
        for case in initial_cases:
            _validate_rewrite_record(case, rewrites[case["case_id"]]["rewrite"], None)
        return {
            "state": "initial_rewrite_inspection_complete",
            "case_ids": [case["case_id"] for case in initial_cases],
            "rewrite_call_attempts": len(rewrite_attempts),
            "rewrite_completed": len(rewrites),
            "gpu": rewrite_session["gpu"] if rewrite_session else None,
        }

    rewrite_call_rows = [row for row in rewrites.values() if row.get("llm_call_attempted")]
    rewrite_usage = {
        "confirmed_model_calls": sum(
            row["rewrite"].get("stats") is not None for row in rewrite_call_rows
        ),
        "call_attempts": len(rewrite_attempts),
        "input_tokens": sum(
            int((row["rewrite"].get("stats") or {}).get("prompt_eval_count") or 0)
            for row in rewrite_call_rows
        ),
        "output_tokens": sum(
            int((row["rewrite"].get("stats") or {}).get("eval_count") or 0)
            for row in rewrite_call_rows
        ),
        "wall_seconds": sum(
            float(row["rewrite"].get("elapsed_ms") or 0) / 1000 for row in rewrite_call_rows
        ),
        "last_invocation": rewrite_session,
    }
    write_state_json(
        grid_dir / "rewrite-usage.json",
        {"static_configuration_sha256": static_hash, "usage": rewrite_usage},
    )

    rewritten_path = grid_dir / "rewritten-retrieval.jsonl"
    rewritten = load_unique_jsonl(rewritten_path)
    if rewritten.keys() - set(input_ids):
        raise ValueError("rewritten retrieval records contain unknown case IDs")
    missing_retrieval = [case for case in inputs if case["case_id"] not in rewritten]
    if missing_retrieval:
        from page_cite_rag.application import document_retriever_parts
        from page_cite_rag.hybrid import HybridRetriever

        query_progress = {"stage": "rewrite_grid_query_embedding", "case_id": None}
        with _preparation_embedding_session(run_root, static, query_progress) as (
            server,
            counting_transport,
        ):
            encoder = NativeEmbeddingEncoder(
                counting_transport,
                DEFAULT_MODEL_DIR,
                max_tokens=static["embedding_input"]["document_max_tokens"],
                query_max_tokens=static["embedding_input"]["query_max_tokens"],
                batch_size=4,
            )
            counting_transport.query_prefix = encoder.identity["query_prompt"]
            query_started = time.monotonic()
            for case in missing_retrieval:
                case_id = case["case_id"]
                query_progress["case_id"] = case_id
                base = baseline_rows[case_id]
                rewrite_record = rewrites[case_id]
                rewrite = rewrite_record["rewrite"]
                lexical_query = rewrite["lexical_query"]
                if not isinstance(lexical_query, str) or not lexical_query.strip():
                    raise ValueError("rewrite record has an invalid lexical query")
                if lexical_query == case["question"]:
                    query_embedding_tokens = 0
                    candidate = _search_result_from_json(base["candidate_result"])
                    context_result = _search_result_from_json(base["result"])
                else:
                    query_embedding_tokens = len(
                        server.token_ids(encoder.identity["query_prompt"] + case["question"])
                    )
                    lexical, dense = document_retriever_parts(
                        store,
                        base["document_id"],
                        "hybrid",
                        encoder=encoder,
                        versioned_dense=True,
                    )
                    if lexical is None or dense is None:
                        raise ValueError("hybrid retrieval requires both frozen branches")
                    hybrid = HybridRetriever(
                        lexical,
                        dense,
                        candidates=static["retrieval"]["candidates_per_branch"],
                        constant=static["retrieval"]["rrf_constant"],
                    )
                    candidate = hybrid.search_with_lexical_query(
                        case["question"],
                        lexical_query,
                        top_k=2 * static["retrieval"]["candidates_per_branch"],
                    )
                    context_result = _limit_search_result_to_context_groups(candidate, TOP_K)
                block_to_passage = base["block_to_passage_id"]
                record = {
                    "case_id": case_id,
                    "question_sha256": case["question_sha256"],
                    "rewrite_status": rewrite["status"],
                    "rewrite_reason": rewrite["reason"],
                    "lexical_query": lexical_query,
                    "rewrite_model_call_attempted": rewrite_record["llm_call_attempted"],
                    "rewrite_shared_from_case_id": rewrite_record["shared_from_case_id"],
                    "query_embedding_tokens": query_embedding_tokens,
                    "candidate_passage_ids": _passage_ids_from_search(candidate, block_to_passage),
                    "context_passage_ids": _passage_ids_from_search(
                        context_result, block_to_passage
                    )
                    if not context_result.branches
                    else list(
                        dict.fromkeys(
                            block_to_passage[hit.block.block_id]
                            for hit in context_result.hits
                            if hit.block.block_id in block_to_passage
                        )
                    ),
                    "candidate_result": candidate.model_dump(mode="json"),
                    "context_result": context_result.model_dump(mode="json"),
                }
                append_jsonl_record(rewritten_path, record)
                rewritten[case_id] = record
                saved.update(
                    {
                        "rewritten_retrieval_completed": len(rewritten),
                        "query_embedding_tokens_this_invocation": counting_transport.query_tokens,
                        "query_embedding_inputs_this_invocation": counting_transport.embed_inputs,
                    }
                )
                write_state_json(progress_path, saved)
            invocation_query_tokens = counting_transport.query_tokens
            invocation_query_inputs = counting_transport.embed_inputs
            expected_query_tokens = sum(
                rewritten[case["case_id"]]["query_embedding_tokens"] for case in missing_retrieval
            )
            if invocation_query_tokens != expected_query_tokens:
                raise ValueError("rewritten query embedding tokens differ from exact preflight")
            query_session = {
                "query_embedding_tokens": invocation_query_tokens,
                "query_embedding_inputs": invocation_query_inputs,
                "wall_seconds": time.monotonic() - query_started,
                "gpu": server.diagnostics(),
            }
        write_state_json(
            grid_dir / "query-embedding-usage.json",
            {
                "static_configuration_sha256": static_hash,
                "usage": {"last_invocation": query_session},
            },
        )
    else:
        query_session = None

    query_usage = {
        "query_embedding_tokens": sum(
            int(row.get("query_embedding_tokens") or 0) for row in rewritten.values()
        ),
        "query_embedding_inputs": sum(
            bool(row.get("query_embedding_tokens")) for row in rewritten.values()
        ),
        "last_invocation": query_session,
    }
    write_state_json(
        grid_dir / "query-embedding-usage.json",
        {"static_configuration_sha256": static_hash, "usage": query_usage},
    )

    reranked_path = grid_dir / "rewritten-reranked.jsonl"
    reranked_attempts_path = grid_dir / "rewritten-reranker-attempts.jsonl"
    reranked_failures_path = grid_dir / "rewritten-reranker-failures.jsonl"
    reranked = load_unique_jsonl(reranked_path)
    reranked_attempts = load_unique_jsonl(reranked_attempts_path)
    reranked_failures = load_unique_jsonl(reranked_failures_path)
    if reranked.keys() - set(input_ids):
        raise ValueError("reranked records contain unknown case IDs")
    if reranked_attempts.keys() - set(input_ids):
        raise ValueError("rewritten reranker attempts contain unknown case IDs")
    if allow_recorded_failures:
        missing_reranks = _pending_grid_cases(
            inputs, reranked_attempts, reranked, reranked_failures
        )
    else:
        reject_interrupted_attempts(reranked_attempts, reranked, "rewritten reranker")
        missing_reranks = [case for case in inputs if case["case_id"] not in reranked]
    if missing_reranks:
        prep = json.loads((run_root / "prepared" / "prepared.json").read_text(encoding="utf-8"))
        base_pair_tokens = sum(sum(row["pair_tokens"]) for row in base_reranked.values())
        completed_pair_tokens = sum(sum(row.get("pair_tokens", [])) for row in reranked.values())
        worst_remaining_pairs = 0
        for case in missing_reranks:
            candidate = _search_result_from_json(rewritten[case["case_id"]]["candidate_result"])
            pair_count = (
                len(candidate.context_groups) if candidate.context_groups else len(candidate.hits)
            )
            worst_remaining_pairs += min(40, pair_count) * 4096
        if (
            known_prior_pair_tokens
            + failed_attempt_pair_token_upper_bound
            + base_pair_tokens
            + completed_pair_tokens
            + len(reranked_failures) * 40 * MAX_PAIR_TOKENS
            + worst_remaining_pairs
            > pair_token_limit
        ):
            raise ValueError(
                "retrieval-grid reranker preflight exceeds the frozen pair-token ceiling"
            )
        rerank_started = time.monotonic()
        with NativeRerankerServer(
            DEFAULT_MODEL_PATH, GPU_UUID, max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C
        ) as ranker:
            for case in missing_reranks:
                case_id = case["case_id"]
                row = rewritten[case_id]
                candidate = _search_result_from_json(row["candidate_result"])
                append_jsonl_record(
                    reranked_attempts_path,
                    {
                        "case_id": case_id,
                        "state": "started",
                        "candidate_count": (
                            len(candidate.context_groups)
                            if candidate.context_groups
                            else len(candidate.hits)
                        ),
                        "candidate_sha256": canonical_hash(candidate.model_dump(mode="json")),
                    },
                )
                started = time.monotonic()
                try:
                    result = rerank_search_result(candidate, ranker, top_k=TOP_K, store=store)
                except Exception as exc:
                    try:
                        gpu = ranker.diagnostics()
                    except Exception:
                        gpu = {"state": "diagnostics_unavailable"}
                    append_jsonl_record(
                        reranked_failures_path,
                        {
                            "case_id": case_id,
                            "state": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                            "pair_tokens": list(ranker.last_pair_tokens),
                            "batch_sizes": list(ranker.last_batch_sizes),
                            "attempted_batch_count": ranker.last_attempted_batch_count,
                            "completed_batch_count": ranker.last_completed_batch_count,
                            "tokenization_ms": ranker.last_tokenization_ms,
                            "forward_ms": ranker.last_forward_ms,
                            "wall_ms": (time.monotonic() - started) * 1000,
                            "gpu": gpu,
                        },
                    )
                    saved.update(
                        {"state": "stopped_on_rewritten_reranker_failure", "failed_case": case_id}
                    )
                    write_state_json(progress_path, saved)
                    raise
                pair_tokens = list(ranker.last_pair_tokens)
                completed_pair_tokens += sum(pair_tokens)
                if (
                    known_prior_pair_tokens
                    + failed_attempt_pair_token_upper_bound
                    + base_pair_tokens
                    + completed_pair_tokens
                    + len(reranked_failures) * 40 * MAX_PAIR_TOKENS
                    > pair_token_limit
                ):
                    raise ValueError(
                        "retrieval-grid reranker exceeds the frozen pair-token ceiling"
                    )
                record = {
                    "case_id": case_id,
                    "result": result.model_dump(mode="json"),
                    "pair_tokens": pair_tokens,
                    "tokenization_ms": ranker.last_tokenization_ms,
                    "forward_ms": ranker.last_forward_ms,
                    "total_ms": ranker.last_total_ms,
                    "wall_ms": (time.monotonic() - started) * 1000,
                    "batch_sizes": list(ranker.last_batch_sizes),
                }
                append_jsonl_record(reranked_path, record)
                reranked[case_id] = record
                saved.update(
                    {
                        "rewritten_reranking_completed": len(reranked),
                        "reranker_pair_tokens": base_pair_tokens + completed_pair_tokens,
                    }
                )
                write_state_json(progress_path, saved)
            rerank_usage = {
                "base_reranker_pair_tokens": base_pair_tokens,
                "rewritten_reranker_pair_tokens": completed_pair_tokens,
                "total_reranker_pair_tokens": base_pair_tokens + completed_pair_tokens,
                "prior_confirmed_pair_tokens": known_prior_pair_tokens,
                "failed_setup_pair_tokens": "unknown",
                "failed_setup_pair_tokens_upper_bound": failed_attempt_pair_token_upper_bound,
                "wall_seconds": time.monotonic() - rerank_started,
                "gpu": ranker.diagnostics(),
                "prepared_artifact_sha256": prep["retrieval_sha256"],
            }
        write_state_json(
            grid_dir / "reranker-usage.json",
            {"static_configuration_sha256": static_hash, "usage": rerank_usage},
        )
    else:
        rerank_usage = (
            json.loads((grid_dir / "reranker-usage.json").read_text(encoding="utf-8"))["usage"]
            if (grid_dir / "reranker-usage.json").is_file()
            else {"state": "reused_rewritten_reranking"}
        )

    grid_rows = []
    for case in inputs:
        case_id = case["case_id"]
        base = baseline_rows[case_id]
        rewritten_row = rewritten[case_id]
        base_candidate = _search_result_from_json(base["candidate_result"])
        base_context = _search_result_from_json(base["result"])
        base_reranked_result = _search_result_from_json(base_reranked[case_id]["result"])
        rewritten_candidate = _search_result_from_json(rewritten_row["candidate_result"])
        rewritten_context = _search_result_from_json(rewritten_row["context_result"])
        rewritten_reranked = (
            _search_result_from_json(reranked[case_id]["result"]) if case_id in reranked else None
        )
        mapping = base["block_to_passage_id"]
        variants = {
            "original_hybrid": _search_snapshot(base_candidate, mapping),
            "keyword_expansion_hybrid": _search_snapshot(rewritten_candidate, mapping),
            "original_hybrid_reranked": _search_snapshot(base_reranked_result, mapping),
            "keyword_expansion_hybrid_reranked": (
                _search_snapshot(rewritten_reranked, mapping) if rewritten_reranked else None
            ),
        }
        # Use the separately admitted top-10 context for the non-reranked variants.
        _set_context_snapshot(variants["original_hybrid"], base_context, mapping)
        _set_context_snapshot(variants["keyword_expansion_hybrid"], rewritten_context, mapping)
        rewrite = rewrites[case_id]["rewrite"]
        grid_rows.append(
            {
                "case_id": case_id,
                "document_cluster": case["document_cluster"],
                "direction": case["direction"],
                "dataset": case["dataset"],
                "question_language": case["question_language"],
                "source_language": case["source_language"],
                "rewrite_status": rewrite["status"],
                "rewrite_reason": rewrite["reason"],
                "rewrite_elapsed_ms": rewrite.get("elapsed_ms"),
                "rewrite_shared_from_case_id": rewrites[case_id]["shared_from_case_id"],
                "rewrite_llm_call_attempted": rewrites[case_id]["llm_call_attempted"],
                "rewrite_stats": rewrite.get("stats"),
                "base_reranker_pair_tokens": base_reranked[case_id]["pair_tokens"],
                "base_reranker_timing_ms": {
                    "tokenization": base_reranked[case_id]["tokenization_ms"],
                    "forward": base_reranked[case_id]["forward_ms"],
                    "total": base_reranked[case_id]["total_ms"],
                },
                "reranker_pair_tokens": reranked[case_id]["pair_tokens"]
                if rewritten_reranked
                else [],
                "reranker_timing_ms": {
                    "tokenization": reranked[case_id]["tokenization_ms"],
                    "forward": reranked[case_id]["forward_ms"],
                    "total": reranked[case_id]["total_ms"],
                }
                if rewritten_reranked
                else None,
                "rewritten_reranker_state": "completed"
                if rewritten_reranked
                else "failed_no_retry",
                "variants": variants,
            }
        )
    atomic_jsonl_once(final_path, grid_rows)
    result = {
        "state": "complete_with_recorded_failures" if reranked_failures else "complete",
        "case_count": len(grid_rows),
        "rewritten_reranker_failed_case_count": len(reranked_failures),
        "static_configuration_sha256": static_hash,
        "retrieval_grid_sha256": sha256(final_path),
        "rewrite_usage": rewrite_usage,
        "query_embedding_usage": query_usage,
        "reranker_usage": rerank_usage,
    }
    saved.update(
        {
            "state": result["state"],
            "rewrite_completed": len(rewrites),
            "rewritten_retrieval_completed": len(rewritten),
            "rewritten_reranking_completed": len(reranked),
            "retrieval_grid_sha256": result["retrieval_grid_sha256"],
        }
    )
    write_state_json(progress_path, saved)
    atomic_json_once(grid_dir / "retrieval-grid-result.json", result)
    return result


def _retrieval_evidence_metrics(
    gold_row: dict, retrieved_ids: list[str], valid_ids: set[str]
) -> dict | None:
    if gold_row.get("unanswerable") is not False:
        return None
    raw_sets = gold_row.get("evidence_reference_passage_ids")
    if not isinstance(raw_sets, list):
        return None
    references = [
        set(reference) & valid_ids
        for reference in raw_sets
        if isinstance(reference, list)
        and reference
        and all(isinstance(value, str) for value in reference)
        and set(reference) <= valid_ids
    ]
    if not references:
        return None
    retrieved = set(retrieved_ids)
    union = set.union(*references)
    recall_by_reference = [len(reference & retrieved) / len(reference) for reference in references]
    return {
        "reference_aware_recall": max(recall_by_reference),
        "union_recall": len(union & retrieved) / len(union),
        "complete_evidence_set_coverage": float(
            any(reference <= retrieved for reference in references)
        ),
        "reference_count": len(references),
        "gold_evidence_passage_count": len(union),
    }


def _alignment_issue_counts(gold_by_id: dict[str, dict], case_ids: list[str]) -> Counter:
    """Count each alignment issue once per case, independent of retrieval variants."""
    counts = Counter()
    for case_id in dict.fromkeys(case_ids):
        issues = gold_by_id[case_id].get("evidence_alignment_issues") or []
        counts.update({str(issue) for issue in issues})
    return counts


def _cluster_bootstrap_mean_ci(rows: list[dict], *, seed: int, iterations: int = 5000) -> dict:
    if not rows:
        return {"estimate": None, "ci95": None, "case_count": 0, "cluster_count": 0}
    clusters: dict[str, list[float]] = {}
    for row in rows:
        clusters.setdefault(row["document_cluster"], []).append(float(row["value"]))
    estimate = sum(float(row["value"]) for row in rows) / len(rows)
    if len(clusters) < 2:
        return {
            "estimate": estimate,
            "ci95": None,
            "case_count": len(rows),
            "cluster_count": len(clusters),
        }
    rng = random.Random(seed)
    names = sorted(clusters)
    estimates = []
    for _ in range(iterations):
        values = []
        for _ in names:
            values.extend(clusters[rng.choice(names)])
        estimates.append(sum(values) / len(values))
    estimates.sort()
    low_index = max(0, math.floor(0.025 * iterations))
    high_index = min(iterations - 1, math.ceil(0.975 * iterations) - 1)
    return {
        "estimate": estimate,
        "ci95": [estimates[low_index], estimates[high_index]],
        "case_count": len(rows),
        "cluster_count": len(clusters),
        "bootstrap_iterations": iterations,
        "seed": seed,
    }


def _score_retrieval_grid(
    run_root: Path, manifest: dict, inputs: list[dict], gold: list[dict]
) -> dict:
    grid_dir = run_root / "retrieval-grid"
    score_path = grid_dir / "retrieval-score.json"
    if score_path.exists():
        raise ValueError("retrieval score already exists; never overwrite a frozen score")
    grid_path = grid_dir / "retrieval-grid.jsonl"
    if not grid_path.is_file():
        raise ValueError("run --phase retrieval-grid before scoring")
    grid_rows = read_jsonl(grid_path)
    case_ids = [case["case_id"] for case in inputs]
    if [row.get("case_id") for row in grid_rows] != case_ids:
        raise ValueError("retrieval grid does not match the frozen input order")
    gold_by_id = {row["case_id"]: row for row in gold}
    if set(gold_by_id) != set(case_ids):
        raise ValueError("retrieval gold does not match frozen input IDs")
    input_by_id = {case["case_id"]: case for case in inputs}
    variant_names = (
        "original_hybrid",
        "keyword_expansion_hybrid",
        "original_hybrid_reranked",
        "keyword_expansion_hybrid_reranked",
    )
    details = []
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    alignment_issues = Counter()
    eligibility_reasons = Counter()
    for row in grid_rows:
        case_id = row["case_id"]
        source = input_by_id[case_id]["source"]
        valid_passages = {
            item["passage_id"]
            for item in source.get("paragraphs", [])
            if isinstance(item, dict) and isinstance(item.get("passage_id"), str)
        }
        gold_row = gold_by_id[case_id]
        for issue in gold_row.get("evidence_alignment_issues") or []:
            alignment_issues[str(issue)] += 1
        row_metrics = {}
        for variant in variant_names:
            if row["variants"][variant] is None:
                row_metrics[variant] = None
                continue
            metrics = _retrieval_evidence_metrics(
                gold_row,
                row["variants"][variant]["candidate_passage_ids"],
                valid_passages,
            )
            context_metrics = _retrieval_evidence_metrics(
                gold_row,
                row["variants"][variant]["context_passage_ids"],
                valid_passages,
            )
            if metrics is None or context_metrics is None:
                row_metrics[variant] = None
                continue
            values = {
                "candidate_reference_aware_recall": metrics["reference_aware_recall"],
                "candidate_union_recall": metrics["union_recall"],
                "context_reference_aware_recall": context_metrics["reference_aware_recall"],
                "context_union_recall": context_metrics["union_recall"],
                "complete_evidence_set_coverage": context_metrics["complete_evidence_set_coverage"],
            }
            row_metrics[variant] = values
            group_key = (row["direction"], variant, "")
            grouped.setdefault(group_key, []).append(
                {
                    "case_id": case_id,
                    "document_cluster": row["document_cluster"],
                    **values,
                }
            )
        if row_metrics["original_hybrid"] is None:
            gold_row = gold_by_id[case_id]
            if gold_row.get("unanswerable") is True:
                reason = "unanswerable"
            elif gold_row.get("unanswerable") is None:
                reason = "no_gold_for_source_track"
            elif not valid_passages:
                reason = "passage_mapping_unavailable"
            else:
                reason = "no_complete_mappable_reference"
            eligibility_reasons[reason] += 1
        details.append(
            {
                "case_id": case_id,
                "document_cluster": row["document_cluster"],
                "direction": row["direction"],
                "eligible": row_metrics["original_hybrid"] is not None,
                "metrics_by_variant": row_metrics,
                "alignment_issues": gold_row.get("evidence_alignment_issues"),
            }
        )

    directions = sorted({row["direction"] for row in grid_rows})
    summaries = {}
    paired_differences = {}
    for direction in directions:
        summaries[direction] = {}
        for variant in variant_names:
            values = grouped.get((direction, variant, ""), [])
            summaries[direction][variant] = {
                "eligible_case_count": len(values),
                "eligible_cluster_count": len({row["document_cluster"] for row in values}),
                **{
                    metric: (sum(row[metric] for row in values) / len(values) if values else None)
                    for metric in (
                        "candidate_reference_aware_recall",
                        "candidate_union_recall",
                        "context_reference_aware_recall",
                        "context_union_recall",
                        "complete_evidence_set_coverage",
                    )
                },
            }
        paired_differences[direction] = {}
        base_by_id = {
            row["case_id"]: row for row in grouped.get((direction, "original_hybrid", ""), [])
        }
        for variant in variant_names[1:]:
            variant_by_id = {
                row["case_id"]: row for row in grouped.get((direction, variant, ""), [])
            }
            common = sorted(base_by_id.keys() & variant_by_id.keys())
            paired_differences[direction][variant] = {}
            for metric in (
                "candidate_reference_aware_recall",
                "context_reference_aware_recall",
                "context_union_recall",
                "complete_evidence_set_coverage",
            ):
                paired = [
                    {
                        "document_cluster": base_by_id[case_id]["document_cluster"],
                        "value": variant_by_id[case_id][metric] - base_by_id[case_id][metric],
                    }
                    for case_id in common
                ]
                paired_differences[direction][variant][metric] = _cluster_bootstrap_mean_ci(
                    paired, seed=20261027
                )

    # The evidence selection rule applies only to QASPER English-source tracks;
    # the Chinese MultiFieldQA cases remain diagnostics because they lack passage gold.
    eligible_directions = [
        direction
        for direction in ("en->en", "zh->en")
        if any(case["direction"] == direction and case["dataset"] == "qasper" for case in inputs)
    ]
    variant_order = list(variant_names)
    mean_context = {}
    mean_coverage = {}
    for variant in variant_order:
        directional = [
            summaries[direction][variant]["context_reference_aware_recall"]
            for direction in eligible_directions
            if summaries[direction][variant]["context_reference_aware_recall"] is not None
        ]
        mean_context[variant] = sum(directional) / len(directional) if directional else None
        coverages = [
            summaries[direction][variant]["complete_evidence_set_coverage"]
            for direction in eligible_directions
            if summaries[direction][variant]["complete_evidence_set_coverage"] is not None
        ]
        mean_coverage[variant] = sum(coverages) / len(coverages) if coverages else None
    prepared_rows = read_jsonl(run_root / "prepared" / "retrieval.jsonl")
    prepared_by_id = {row["case_id"]: row for row in prepared_rows}
    timing_values = {variant: [] for variant in variant_names}
    rewrite_call_count = rewrite_input_tokens = rewrite_output_tokens = 0
    reranker_pair_tokens = {variant: 0 for variant in variant_names}
    for row in grid_rows:
        case_id = row["case_id"]
        base = prepared_by_id[case_id]
        rewrite_stats = row.get("rewrite_stats") or {}
        rewrite_call = bool(row.get("rewrite_llm_call_attempted"))
        if rewrite_call:
            rewrite_call_count += 1
            rewrite_input_tokens += int(rewrite_stats.get("prompt_eval_count") or 0)
            rewrite_output_tokens += int(rewrite_stats.get("eval_count") or 0)
        base_retrieval_ms = float(base["candidate_result"].get("elapsed_ms") or 0)
        rewrite_elapsed_ms = float(row.get("rewrite_elapsed_ms") or 0) if rewrite_call else 0
        rewritten_retrieval_ms = float(
            row["variants"]["keyword_expansion_hybrid"].get("retrieval_elapsed_ms") or 0
        )
        base_reranker_ms = float((row.get("base_reranker_timing_ms") or {}).get("total") or 0)
        rewritten_ranker_ms = float((row.get("reranker_timing_ms") or {}).get("total") or 0)
        timing_values["original_hybrid"].append(base_retrieval_ms)
        timing_values["keyword_expansion_hybrid"].append(
            rewrite_elapsed_ms + rewritten_retrieval_ms
        )
        timing_values["original_hybrid_reranked"].append(base_retrieval_ms + base_reranker_ms)
        if row["variants"]["keyword_expansion_hybrid_reranked"] is not None:
            timing_values["keyword_expansion_hybrid_reranked"].append(
                rewrite_elapsed_ms + rewritten_retrieval_ms + rewritten_ranker_ms
            )
        base_pairs = sum(row.get("base_reranker_pair_tokens", []))
        expanded_pairs = sum(row.get("reranker_pair_tokens", []))
        reranker_pair_tokens["original_hybrid_reranked"] += base_pairs
        reranker_pair_tokens["keyword_expansion_hybrid_reranked"] += expanded_pairs
    query_embedding_usage = (
        json.loads((grid_dir / "query-embedding-usage.json").read_text(encoding="utf-8"))["usage"]
        if (grid_dir / "query-embedding-usage.json").is_file()
        else {}
    )
    retrieval_costs = {
        variant: {
            "rewrite_model_calls": rewrite_call_count if "keyword_expansion" in variant else 0,
            "rewrite_input_tokens": rewrite_input_tokens if "keyword_expansion" in variant else 0,
            "rewrite_output_tokens": rewrite_output_tokens if "keyword_expansion" in variant else 0,
            "reranker_pair_tokens": reranker_pair_tokens[variant],
            "additional_query_embedding_tokens_for_grid_measurement": (
                query_embedding_usage.get("query_embedding_tokens", 0)
                if "keyword_expansion" in variant
                else 0
            ),
            "per_case_component_ms": {
                "p50": _percentile(timing_values[variant], 0.50),
                "p95": _percentile(timing_values[variant], 0.95),
            },
            "timing_scope": (
                "recorded retrieval, rewrite and reranker components; "
                "not complete end-to-end time or shared service startup"
            ),
        }
        for variant in variant_names
    }
    max_context = max((value for value in mean_context.values() if value is not None), default=None)
    tied = [
        variant
        for variant in variant_order
        if mean_context[variant] is not None and max_context - mean_context[variant] < 0.01
    ]
    max_coverage = max((mean_coverage[variant] for variant in tied), default=None)
    coverage_tied = [
        variant
        for variant in tied
        if mean_coverage[variant] is not None
        and max_coverage is not None
        and abs(max_coverage - mean_coverage[variant]) < 1e-12
    ]

    def measured_cost(variant: str) -> tuple[int, int, int, float]:
        cost = retrieval_costs[variant]
        latency = cost["per_case_component_ms"]["p50"]
        return (
            cost["rewrite_model_calls"],
            cost["rewrite_input_tokens"] + cost["rewrite_output_tokens"],
            cost["reranker_pair_tokens"],
            float("inf") if latency is None else float(latency),
        )

    initial_winner = min(coverage_tied or tied or ["original_hybrid"], key=measured_cost)
    winner = initial_winner
    regression = any(
        (
            summaries[direction][winner]["context_reference_aware_recall"] is not None
            and summaries[direction]["original_hybrid"]["context_reference_aware_recall"]
            - summaries[direction][winner]["context_reference_aware_recall"]
            > 0.05
        )
        or (
            summaries[direction][winner]["complete_evidence_set_coverage"] is not None
            and summaries[direction]["original_hybrid"]["complete_evidence_set_coverage"]
            - summaries[direction][winner]["complete_evidence_set_coverage"]
            > 0.02
        )
        for direction in eligible_directions
    )
    if regression or max_context is None:
        winner = "original_hybrid"
    missing_rewritten_reranks = sum(
        row["variants"]["keyword_expansion_hybrid_reranked"] is None for row in grid_rows
    )
    if missing_rewritten_reranks:
        winner = None
    score = {
        "state": "complete",
        "source_manifest_sha256": manifest["files"]["gold.jsonl"]["sha256"],
        "retrieval_grid_sha256": sha256(grid_path),
        "sample_count": len(inputs),
        "rewritten_reranker_missing_count": missing_rewritten_reranks,
        "eligible_evidence_case_count": sum(detail["eligible"] for detail in details),
        "eligible_cluster_count": len(
            {detail["document_cluster"] for detail in details if detail["eligible"]}
        ),
        "eligibility_reasons": dict(eligibility_reasons),
        "alignment_issue_counts": dict(alignment_issues),
        "metric_contract": {
            "candidate_recall": "max recall over complete mappable alternative evidence sets",
            "context_recall": (
                "max recall over complete mappable alternative evidence sets in top-10 "
                "selected blocks before exact model-token admission"
            ),
            "complete_evidence_set_coverage": (
                "whether context contains every passage in at least one complete "
                "mappable evidence set"
            ),
            "bootstrap": "paired source-cluster bootstrap, 5000 replicates, seed 20261027",
            "unanswerable_and_unmapped": "N/A; never imputed as zero",
        },
        "by_direction": summaries,
        "paired_delta_vs_original_hybrid": paired_differences,
        "measured_cost_by_variant": retrieval_costs,
        "selection": {
            "eligible_directions": eligible_directions,
            "primary_selection_score_macro_direction_context_recall": mean_context,
            "complete_evidence_set_coverage_macro_direction": mean_coverage,
            "initial_winner": initial_winner,
            "selected_variant": winner,
            "selection_state": (
                "incomplete_grid_requires_explicit_decision"
                if missing_rewritten_reranks
                else "selected_from_mappable_qasper_evidence"
                if max_context is not None
                else "not_evaluable_keep_original_hybrid_control"
            ),
            "regression_gate_reverted_to_original": regression,
            "rule": (
                "highest macro-direction context recall; within 0.01 prefer complete "
                "evidence-set coverage, then lower cost; reject >0.05 recall or >0.02 "
                "coverage regression in any eligible direction"
            ),
            "evidence_scope": (
                "QASPER passage text only; no claim about Chinese-source MultiFieldQA "
                "evidence recall"
            ),
        },
        "case_details": details,
    }
    atomic_json_once(score_path, score)
    return score


def _exact_context_admission(store, search: SearchResult, case: dict, model) -> dict:
    """Measure the exact production prompt budget without generating an answer."""
    from page_cite_rag.application import FrozenSearchRetriever, document_workflow

    workflow = document_workflow(
        store,
        search.document_id,
        FrozenSearchRetriever(search, top_k=TOP_K),
        model,
        model.config,
        top_k=TOP_K,
        answer_format=_answer_format_for(case),
    )
    blocks = workflow._context_blocks(search, workflow._read_hits(search, case["question"]))
    selected, omitted, _, budget = workflow._context(
        case["question"],
        case["language"],
        "local",
        blocks,
        None,
        context_groups=search.context_groups,
    )
    if budget.exact_input_tokens is None or budget.exact_input_tokens > EXACT_INPUT_LIMIT:
        raise ContextBudgetError("reranker context failed exact generation-input admission")
    return {
        "admitted_block_ids": [block.block_id for block in selected],
        "omitted_block_ids": list(omitted),
        "exact_input_tokens": budget.exact_input_tokens,
        "context_group_count": len(search.context_groups),
    }


def _rank_full_sentence_retrieval_grid(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    """Rerank frozen original/selected-route pools and exact-admit their contexts."""
    if static["retrieval"].get("retrieval_grid", {}).get("protocol") != (
        "full-sentence-rewrite-routing/v1"
    ):
        raise ValueError("same-pool reranking requires the frozen full-sentence retrieval grid")
    if not static["retrieval"].get("reranker"):
        raise ValueError("prepare the full-sentence grid with the reranked arm enabled")

    from page_cite_rag.hybrid import rerank_search_result
    from page_cite_rag.reranker_server import (
        DEFAULT_MODEL_PATH,
        MAX_PAIR_TOKENS,
        NativeRerankerServer,
    )

    grid_dir = run_root / "retrieval-grid"
    grid_path = grid_dir / "retrieval-grid.jsonl"
    route_score_path = grid_dir / "retrieval-score.json"
    if not grid_path.is_file() or not route_score_path.is_file():
        raise ValueError("complete and score the rewrite grid before same-pool reranking")
    grid_rows = read_jsonl(grid_path)
    case_ids = [case["case_id"] for case in inputs]
    if [row.get("case_id") for row in grid_rows] != case_ids:
        raise ValueError("reranker grid input order differs from the frozen cases")
    route_score = json.loads(route_score_path.read_text(encoding="utf-8"))
    selected_route = route_score.get("selection", {}).get("selected_variant")
    if not isinstance(selected_route, str) or not grid_rows:
        raise ValueError("retrieval grid has no selected route")
    available_routes = set(grid_rows[0].get("candidate_results", {}))
    if selected_route not in available_routes:
        raise ValueError("selected retrieval route is not present in the frozen grid")
    pool_variants = list(dict.fromkeys(("original_hybrid", selected_route)))

    pools = []
    candidate_counts = {}
    for row in grid_rows:
        for variant in pool_variants:
            candidate = _search_result_from_json(row["candidate_results"][variant])
            count = len(candidate.context_groups) or len(candidate.hits)
            if count > 40:
                raise ValueError("frozen candidate pool exceeds the 40-item reranker limit")
            pool_id = f"{row['case_id']}::{variant}"
            pools.append((pool_id, row, variant, candidate))
            candidate_counts[pool_id] = count
    if len(pools) > 480:
        raise ValueError("same-pool reranker exceeds its frozen 480-pool ceiling")
    pair_token_upper_bound = sum(candidate_counts.values()) * MAX_PAIR_TOKENS
    if pair_token_upper_bound > MAX_RERANKER_PAIR_TOKENS:
        raise ValueError("same-pool reranker upper bound exceeds its frozen pair-token ceiling")

    output_dir = grid_dir / "same-pool-reranker"
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = {
        "protocol": "same-candidate-pool-reranker/v1",
        "static_configuration_sha256": canonical_hash(static),
        "retrieval_grid_sha256": sha256(grid_path),
        "route_score_sha256": sha256(route_score_path),
        "selected_route": selected_route,
        "pool_variants": pool_variants,
        "candidate_pool_count": len(pools),
        "pair_token_upper_bound": pair_token_upper_bound,
        "pair_token_limit": MAX_RERANKER_PAIR_TOKENS,
        "pair_max_tokens": MAX_PAIR_TOKENS,
        "rerank_top_k": TOP_K,
        "generation_calls": 0,
    }
    config_path = output_dir / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("same-pool reranker configuration changed after it started")
    else:
        atomic_json_once(config_path, config)

    results_path = output_dir / "results.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    failures_path = output_dir / "failures.jsonl"
    admissions_path = output_dir / "admissions.jsonl"
    admission_attempts_path = output_dir / "admission-attempts.jsonl"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "complete.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            saved.get("configuration_sha256") != canonical_hash(config)
            or saved.get("results_sha256") != sha256(results_path)
            or saved.get("admissions_sha256") != sha256(admissions_path)
        ):
            raise ValueError("completed same-pool reranker artifacts changed")
        return saved

    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {
            "state": "in_progress",
            "configuration_sha256": canonical_hash(config),
            "candidate_pool_count": len(pools),
            "pair_token_upper_bound": pair_token_upper_bound,
            "completed_pools": 0,
            "confirmed_pair_tokens": 0,
            "unknown_pair_tokens_upper_bound": 0,
            "reranker_gpu_wall_seconds": 0.0,
            "admission_gpu_wall_seconds": 0.0,
        }
    )
    if progress.get("configuration_sha256") != canonical_hash(config):
        raise ValueError("same-pool reranker progress belongs to another configuration")

    result_rows = load_unique_jsonl(results_path, key="pool_id")
    attempt_rows = load_unique_jsonl(attempts_path, key="pool_id")
    failure_rows = load_unique_jsonl(failures_path, key="pool_id")
    valid_pool_ids = {pool[0] for pool in pools}
    if (result_rows.keys() | attempt_rows.keys() | failure_rows.keys()) - valid_pool_ids:
        raise ValueError("same-pool reranker contains unknown candidate pools")
    if failure_rows:
        raise ValueError("a reranker or admission request failed; preserve it and do not retry")
    reject_interrupted_attempts(attempt_rows, result_rows, "same-pool reranker")
    confirmed_pair_tokens = sum(sum(row.get("pair_tokens", [])) for row in result_rows.values())
    unknown_pair_tokens = int(progress.get("unknown_pair_tokens_upper_bound", 0))
    if confirmed_pair_tokens + unknown_pair_tokens > MAX_RERANKER_PAIR_TOKENS:
        raise ValueError("same-pool reranker cumulative pair-token accounting is invalid")
    missing_pools = [pool for pool in pools if pool[0] not in result_rows]
    remaining_upper_bound = (
        sum(candidate_counts[pool[0]] for pool in missing_pools) * MAX_PAIR_TOKENS
    )
    if (
        confirmed_pair_tokens + unknown_pair_tokens + remaining_upper_bound
        > MAX_RERANKER_PAIR_TOKENS
    ):
        raise ValueError("same-pool reranker remaining pair-token bound exceeds its ceiling")

    store, prepared_rows = _load_prepared(run_root, static, include_reranked=False)
    input_by_id = {case["case_id"]: case for case in inputs}
    reranker_started = time.monotonic()
    reranker_base_wall = float(progress.get("reranker_gpu_wall_seconds", 0))
    reranker_gpu = None
    if missing_pools:
        with NativeRerankerServer(
            DEFAULT_MODEL_PATH,
            GPU_UUID,
            max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C,
        ) as ranker:
            for pool_id, row, variant, candidate in missing_pools:
                current_wall = reranker_base_wall + time.monotonic() - reranker_started
                if current_wall >= MAX_RERANKER_GRID_WALL_SECONDS:
                    progress.update(
                        {
                            "state": "pending_unstarted_pools",
                            "completed_pools": len(result_rows),
                            "reranker_gpu_wall_seconds": current_wall,
                        }
                    )
                    write_state_json(progress_path, progress)
                    return progress
                attempt = {
                    "pool_id": pool_id,
                    "case_id": row["case_id"],
                    "variant": variant,
                    "state": "started",
                    "candidate_count": candidate_counts[pool_id],
                    "candidate_sha256": canonical_hash(candidate.model_dump(mode="json")),
                }
                append_jsonl_record(attempts_path, attempt)
                attempt_started = time.monotonic()
                try:
                    ranked = rerank_search_result(candidate, ranker, top_k=TOP_K, store=store)
                except Exception as exc:
                    known_tokens = list(ranker.last_pair_tokens)
                    unknown_upper = (
                        max(0, candidate_counts[pool_id] - len(known_tokens)) * MAX_PAIR_TOKENS
                    )
                    failure = {
                        **attempt,
                        "state": "failed_no_retry",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "pair_tokens_known": known_tokens,
                        "pair_tokens_unknown_upper_bound": unknown_upper,
                        "batch_sizes": list(ranker.last_batch_sizes),
                        "attempted_batch_count": ranker.last_attempted_batch_count,
                        "completed_batch_count": ranker.last_completed_batch_count,
                        "wall_ms": (time.monotonic() - attempt_started) * 1000,
                        "gpu": ranker.diagnostics(),
                    }
                    append_jsonl_record(failures_path, failure)
                    progress.update(
                        {
                            "state": "failed_no_retry",
                            "completed_pools": len(result_rows),
                            "confirmed_pair_tokens": confirmed_pair_tokens,
                            "unknown_pair_tokens_upper_bound": unknown_pair_tokens + unknown_upper,
                            "reranker_gpu_wall_seconds": reranker_base_wall
                            + time.monotonic()
                            - reranker_started,
                        }
                    )
                    write_state_json(progress_path, progress)
                    raise
                record = {
                    **attempt,
                    "state": "complete",
                    "result": ranked.model_dump(mode="json"),
                    "pair_tokens": list(ranker.last_pair_tokens),
                    "batch_sizes": list(ranker.last_batch_sizes),
                    "tokenization_ms": ranker.last_tokenization_ms,
                    "forward_ms": ranker.last_forward_ms,
                    "total_ms": ranker.last_total_ms,
                    "wall_ms": (time.monotonic() - attempt_started) * 1000,
                }
                append_jsonl_record(results_path, record)
                result_rows[pool_id] = record
                confirmed_pair_tokens += sum(record["pair_tokens"])
                remaining_upper_bound -= candidate_counts[pool_id] * MAX_PAIR_TOKENS
                if confirmed_pair_tokens + unknown_pair_tokens + remaining_upper_bound > (
                    MAX_RERANKER_PAIR_TOKENS
                ):
                    raise ValueError("actual pair-token usage exceeded the frozen cumulative bound")
                progress.update(
                    {
                        "state": "reranking_in_progress",
                        "completed_pools": len(result_rows),
                        "confirmed_pair_tokens": confirmed_pair_tokens,
                        "reranker_gpu_wall_seconds": reranker_base_wall
                        + time.monotonic()
                        - reranker_started,
                    }
                )
                write_state_json(progress_path, progress)
            reranker_gpu = ranker.diagnostics()
    progress["reranker_gpu_wall_seconds"] = reranker_base_wall + time.monotonic() - reranker_started
    progress["confirmed_pair_tokens"] = confirmed_pair_tokens
    progress["completed_pools"] = len(result_rows)
    progress["state"] = "reranking_complete_context_admission_pending"
    write_state_json(progress_path, progress)

    admission_rows = load_unique_jsonl(admissions_path, key="pool_id")
    admission_attempts = load_unique_jsonl(admission_attempts_path, key="pool_id")
    if (admission_rows.keys() | admission_attempts.keys()) - valid_pool_ids:
        raise ValueError("exact context admissions contain unknown candidate pools")
    reject_interrupted_attempts(admission_attempts, admission_rows, "exact context admission")
    missing_admissions = [pool for pool in pools if pool[0] not in admission_rows]
    admission_started = time.monotonic()
    admission_base_wall = float(progress.get("admission_gpu_wall_seconds", 0))
    admission_gpu = None
    if missing_admissions:
        from page_cite_rag.llama import LlamaModelClient

        with LlamaModelClient(
            _llama_config(static),
            budget_policy="exact",
            recover_output_errors=False,
            sampling_profile=static["answer"]["sampling_profile"],
            sampling_seed=42,
        ) as model:
            for pool_id, row, variant, candidate in missing_admissions:
                current_wall = (
                    progress["reranker_gpu_wall_seconds"]
                    + admission_base_wall
                    + time.monotonic()
                    - admission_started
                )
                if current_wall >= MAX_RERANKER_GRID_WALL_SECONDS:
                    progress.update(
                        {
                            "state": "pending_unstarted_admissions",
                            "completed_admissions": len(admission_rows),
                            "admission_gpu_wall_seconds": admission_base_wall
                            + time.monotonic()
                            - admission_started,
                        }
                    )
                    write_state_json(progress_path, progress)
                    return progress
                ranked_row = result_rows[pool_id]
                candidate_sha = canonical_hash(candidate.model_dump(mode="json"))
                if ranked_row.get("candidate_sha256") != candidate_sha:
                    raise ValueError("reranked result does not match its frozen candidate pool")
                append_jsonl_record(
                    admission_attempts_path,
                    {
                        "pool_id": pool_id,
                        "case_id": row["case_id"],
                        "variant": variant,
                        "state": "started",
                    },
                )
                case = input_by_id[row["case_id"]]
                before = _search_result_from_json(row["context_results"][variant])
                after = _search_result_from_json(ranked_row["result"])
                try:
                    before_admission = _exact_context_admission(store, before, case, model)
                    after_admission = _exact_context_admission(store, after, case, model)
                except Exception as exc:
                    append_jsonl_record(
                        failures_path,
                        {
                            "pool_id": pool_id,
                            "case_id": row["case_id"],
                            "variant": variant,
                            "state": "admission_failed_no_retry",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        },
                    )
                    progress.update(
                        {
                            "state": "admission_failed_no_retry",
                            "admission_gpu_wall_seconds": admission_base_wall
                            + time.monotonic()
                            - admission_started,
                        }
                    )
                    write_state_json(progress_path, progress)
                    raise
                admission = {
                    "pool_id": pool_id,
                    "case_id": row["case_id"],
                    "variant": variant,
                    "candidate_sha256": candidate_sha,
                    "before": before_admission,
                    "after": after_admission,
                }
                append_jsonl_record(admissions_path, admission)
                admission_rows[pool_id] = admission
                progress.update(
                    {
                        "state": "context_admission_in_progress",
                        "completed_admissions": len(admission_rows),
                        "admission_gpu_wall_seconds": admission_base_wall
                        + time.monotonic()
                        - admission_started,
                    }
                )
                write_state_json(progress_path, progress)
            admission_gpu = _safe_model_diagnostics(model)
    progress["admission_gpu_wall_seconds"] = (
        admission_base_wall + time.monotonic() - admission_started
    )

    if len(result_rows) != len(pools) or len(admission_rows) != len(pools):
        raise ValueError("same-pool reranker comparison is incomplete")
    result = {
        "state": "complete",
        "protocol": config["protocol"],
        "configuration_sha256": canonical_hash(config),
        "retrieval_grid_sha256": config["retrieval_grid_sha256"],
        "selected_route": selected_route,
        "pool_variants": pool_variants,
        "candidate_pool_count": len(pools),
        "completed_pools": len(result_rows),
        "completed_context_admissions": len(admission_rows),
        "candidate_pair_token_upper_bound": pair_token_upper_bound,
        "confirmed_pair_tokens": confirmed_pair_tokens,
        "unknown_pair_tokens_upper_bound": unknown_pair_tokens,
        "reranker_gpu_wall_seconds": progress["reranker_gpu_wall_seconds"],
        "admission_gpu_wall_seconds": progress["admission_gpu_wall_seconds"],
        "reranker_gpu": reranker_gpu,
        "admission_gpu": admission_gpu,
        "generation_calls": 0,
        "results_sha256": sha256(results_path),
        "admissions_sha256": sha256(admissions_path),
        "attempts_sha256": sha256(attempts_path),
    }
    atomic_json_once(manifest_path, result)
    progress.update({"state": "complete", "completed_admissions": len(admission_rows)})
    write_state_json(progress_path, progress)
    return result


def _score_same_pool_reranker_grid(
    run_root: Path, manifest: dict, inputs: list[dict], gold: list[dict]
) -> dict:
    """Compare reranking on/off over identical candidate pools and exact contexts."""
    grid_dir = run_root / "retrieval-grid"
    output_dir = grid_dir / "same-pool-reranker"
    score_path = output_dir / "score.json"
    if score_path.exists():
        raise ValueError("same-pool reranker score already exists; never overwrite it")
    config = json.loads((output_dir / "configuration.json").read_text(encoding="utf-8"))
    completed = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    grid_path = grid_dir / "retrieval-grid.jsonl"
    route_score_path = grid_dir / "retrieval-score.json"
    results_path = output_dir / "results.jsonl"
    admissions_path = output_dir / "admissions.jsonl"
    if (
        completed.get("state") != "complete"
        or completed.get("configuration_sha256") != canonical_hash(config)
        or completed.get("results_sha256") != sha256(results_path)
        or completed.get("admissions_sha256") != sha256(admissions_path)
        or config.get("retrieval_grid_sha256") != sha256(grid_path)
        or config.get("route_score_sha256") != sha256(route_score_path)
        or config.get("generation_calls") != 0
    ):
        raise ValueError("same-pool reranker artifacts do not match their frozen inputs")
    grid_rows = read_jsonl(grid_path)
    case_ids = [case["case_id"] for case in inputs]
    if [row.get("case_id") for row in grid_rows] != case_ids:
        raise ValueError("same-pool reranker cases differ from frozen input order")
    pool_variants = config["pool_variants"]
    expected_pool_ids = {
        f"{row['case_id']}::{variant}" for row in grid_rows for variant in pool_variants
    }
    results = load_unique_jsonl(results_path, key="pool_id")
    admissions = load_unique_jsonl(admissions_path, key="pool_id")
    if set(results) != expected_pool_ids or set(admissions) != expected_pool_ids:
        raise ValueError("same-pool reranker output is missing or has extra result rows")
    if completed.get("candidate_pool_count") != len(expected_pool_ids):
        raise ValueError("same-pool reranker manifest has an inconsistent pool count")

    static = json.loads((run_root / "configuration.json").read_text(encoding="utf-8"))["static"]
    if canonical_hash(static) != config["static_configuration_sha256"]:
        raise ValueError("same-pool reranker run configuration changed")
    _, prepared_rows = _load_prepared(run_root, static, include_reranked=False)
    if set(prepared_rows) != set(case_ids):
        raise ValueError("prepared source mapping does not match reranker inputs")
    input_by_id = {case["case_id"]: case for case in inputs}
    gold_by_id = {row["case_id"]: row for row in gold}
    if set(gold_by_id) != set(case_ids):
        raise ValueError("same-pool reranker gold does not match frozen input IDs")

    metric_names = (
        "reference_aware_recall",
        "union_recall",
        "complete_evidence_set_coverage",
    )
    grouped = {}
    details = []
    alignment_issues = _alignment_issue_counts(gold_by_id, case_ids)
    eligibility_reasons = Counter()
    for row in grid_rows:
        case_id = row["case_id"]
        case = input_by_id[case_id]
        gold_row = gold_by_id[case_id]
        source = case["source"]
        valid_passages = {
            passage["passage_id"]
            for passage in source.get("paragraphs", [])
            if isinstance(passage, dict) and isinstance(passage.get("passage_id"), str)
        }
        mapping = prepared_rows[case_id].get("block_to_passage_id", {})
        for variant in pool_variants:
            pool_id = f"{case_id}::{variant}"
            record = results[pool_id]
            admission = admissions[pool_id]
            candidate_result = _search_result_from_json(row["candidate_results"][variant])
            candidate_sha = canonical_hash(candidate_result.model_dump(mode="json"))
            if (
                record.get("candidate_sha256") != candidate_sha
                or admission.get("candidate_sha256") != candidate_sha
            ):
                raise ValueError("reranker changed the frozen candidate pool")
            candidate_ids = row["variants"][variant]["candidate_passage_ids"]
            candidate_metrics = _retrieval_evidence_metrics(gold_row, candidate_ids, valid_passages)
            before_ids = admission["before"]["admitted_block_ids"]
            after_ids = admission["after"]["admitted_block_ids"]
            before_passages = list(
                dict.fromkeys(mapping[block_id] for block_id in before_ids if block_id in mapping)
            )
            after_passages = list(
                dict.fromkeys(mapping[block_id] for block_id in after_ids if block_id in mapping)
            )
            before_metrics = _retrieval_evidence_metrics(gold_row, before_passages, valid_passages)
            after_metrics = _retrieval_evidence_metrics(gold_row, after_passages, valid_passages)
            values = {
                "candidate": candidate_metrics,
                "before_rerank": before_metrics,
                "after_rerank": after_metrics,
            }
            if candidate_metrics is not None:
                grouped.setdefault((row["direction"], variant, "candidate"), []).append(
                    {
                        "case_id": case_id,
                        "document_cluster": row["document_cluster"],
                        **candidate_metrics,
                    }
                )
            eligible = before_metrics is not None and after_metrics is not None
            if eligible:
                grouped.setdefault((row["direction"], variant, "before_rerank"), []).append(
                    {
                        "case_id": case_id,
                        "document_cluster": row["document_cluster"],
                        **before_metrics,
                    }
                )
                grouped.setdefault((row["direction"], variant, "after_rerank"), []).append(
                    {
                        "case_id": case_id,
                        "document_cluster": row["document_cluster"],
                        **after_metrics,
                    }
                )
                grouped.setdefault((row["direction"], variant, "paired"), []).append(
                    {
                        "case_id": case_id,
                        "document_cluster": row["document_cluster"],
                        "before": before_metrics,
                        "after": after_metrics,
                    }
                )
            elif gold_row.get("unanswerable") is True:
                eligibility_reasons["unanswerable"] += 1
            elif gold_row.get("unanswerable") is None:
                eligibility_reasons["no_gold_for_source_track"] += 1
            elif not valid_passages:
                eligibility_reasons["passage_mapping_unavailable"] += 1
            else:
                eligibility_reasons["no_complete_mappable_reference"] += 1
            details.append(
                {
                    "case_id": case_id,
                    "document_cluster": row["document_cluster"],
                    "direction": row["direction"],
                    "variant": variant,
                    "candidate_sha256": candidate_sha,
                    "candidate_count": record["candidate_count"],
                    "reranked_context_block_ids": [
                        hit["block"]["block_id"] for hit in record["result"]["hits"]
                    ],
                    "before_admission": admission["before"],
                    "after_admission": admission["after"],
                    "metrics": values,
                    "eligible": eligible,
                    "pair_tokens": sum(record.get("pair_tokens", [])),
                    "rerank_wall_ms": record.get("wall_ms"),
                }
            )

    directions = sorted({row["direction"] for row in grid_rows})
    by_direction = {}
    paired_deltas = {}
    for direction in directions:
        by_direction[direction] = {}
        paired_deltas[direction] = {}
        for variant in pool_variants:
            candidate_rows = grouped.get((direction, variant, "candidate"), [])
            before_rows = grouped.get((direction, variant, "before_rerank"), [])
            after_rows = grouped.get((direction, variant, "after_rerank"), [])
            by_direction[direction][variant] = {
                "eligible_case_count": len(before_rows),
                "eligible_cluster_count": len({row["document_cluster"] for row in before_rows}),
                "candidate_pool": {
                    metric: (
                        sum(row[metric] for row in candidate_rows) / len(candidate_rows)
                        if candidate_rows
                        else None
                    )
                    for metric in metric_names
                },
                "before_rerank": {
                    metric: (
                        sum(row[metric] for row in before_rows) / len(before_rows)
                        if before_rows
                        else None
                    )
                    for metric in metric_names
                },
                "after_rerank": {
                    metric: (
                        sum(row[metric] for row in after_rows) / len(after_rows)
                        if after_rows
                        else None
                    )
                    for metric in metric_names
                },
            }
            paired_rows = grouped.get((direction, variant, "paired"), [])
            paired_deltas[direction][variant] = {
                output_metric: _cluster_bootstrap_mean_ci(
                    [
                        {
                            "document_cluster": row["document_cluster"],
                            "value": (row["after"][source_metric] - row["before"][source_metric]),
                        }
                        for row in paired_rows
                    ],
                    seed=20261028,
                )
                for output_metric, source_metric in (
                    ("context_reference_aware_recall", "reference_aware_recall"),
                    ("complete_evidence_set_coverage", "complete_evidence_set_coverage"),
                )
            }

    score = {
        "state": "complete",
        "protocol": "same-candidate-pool-reranker/v1",
        "source_manifest_sha256": manifest["files"]["gold.jsonl"]["sha256"],
        "retrieval_grid_sha256": sha256(grid_path),
        "reranker_results_sha256": sha256(results_path),
        "exact_admissions_sha256": sha256(admissions_path),
        "sample_count": len(inputs),
        "candidate_pool_count": completed["candidate_pool_count"],
        "pool_variants": pool_variants,
        "eligible_case_count": sum(
            item["eligible"] and item["variant"] == pool_variants[0] for item in details
        ),
        "alignment_issue_counts": dict(alignment_issues),
        "eligibility_reasons": dict(eligibility_reasons),
        "metric_contract": {
            "candidate_recall": (
                "measured over the identical frozen top-20-per-branch candidate pool"
            ),
            "context_recall": (
                "measured after exact production prompt token admission at the same "
                "12000-token input limit"
            ),
            "complete_evidence_set_coverage": (
                "one complete mappable QASPER evidence alternative fits in the "
                "exact admitted context"
            ),
            "paired_comparison": "same case and source cluster, before versus after reranking",
            "bootstrap": "paired source-cluster bootstrap, 5000 replicates, seed 20261028",
            "answer_quality": "not measured in this retrieval-only stage",
        },
        "by_direction": by_direction,
        "paired_delta_after_minus_before": paired_deltas,
        "cost": {
            "confirmed_pair_tokens": completed["confirmed_pair_tokens"],
            "pair_token_upper_bound": completed["candidate_pair_token_upper_bound"],
            "reranker_gpu_wall_seconds": completed["reranker_gpu_wall_seconds"],
            "exact_context_admission_gpu_wall_seconds": completed["admission_gpu_wall_seconds"],
            "generation_calls": 0,
            "rerank_wall_ms_p50": _percentile(
                [
                    float(row["rerank_wall_ms"])
                    for row in details
                    if row["rerank_wall_ms"] is not None
                ],
                0.50,
            ),
            "rerank_wall_ms_p95": _percentile(
                [
                    float(row["rerank_wall_ms"])
                    for row in details
                    if row["rerank_wall_ms"] is not None
                ],
                0.95,
            ),
        },
        "by_direction_and_pool": by_direction,
        "paired_delta_vs_rerank_off": paired_deltas,
        "case_details": details,
    }
    atomic_json_once(score_path, score)
    return score


def _score_full_sentence_retrieval_grid(
    run_root: Path, manifest: dict, inputs: list[dict], gold: list[dict]
) -> dict:
    """Score four rewrite routes on frozen evidence, without answer judging."""
    grid_dir = run_root / "retrieval-grid"
    score_path = grid_dir / "retrieval-score.json"
    if score_path.exists():
        raise ValueError("retrieval score already exists; never overwrite a frozen score")
    grid_path = grid_dir / "retrieval-grid.jsonl"
    if not grid_path.is_file():
        raise ValueError("run the complete full-sentence retrieval grid before scoring")
    rows = read_jsonl(grid_path)
    case_ids = [case["case_id"] for case in inputs]
    if [row.get("case_id") for row in rows] != case_ids:
        raise ValueError("full-sentence retrieval grid differs from frozen input order")
    gold_by_id = {row["case_id"]: row for row in gold}
    input_by_id = {case["case_id"]: case for case in inputs}
    if set(gold_by_id) != set(case_ids):
        raise ValueError("frozen retrieval gold does not match input IDs")

    variants = ("original_hybrid", "rewrite_bm25_only", "rewrite_dense_only", "rewrite_both")
    grouped: dict[tuple[str, str], list[dict]] = {}
    details = []
    alignment_issues = Counter()
    eligibility_reasons = Counter()
    rewrite_calls = rewrite_input_tokens = rewrite_output_tokens = 0
    timing: dict[str, list[float]] = {name: [] for name in variants}
    for row in rows:
        case_id = row["case_id"]
        source = input_by_id[case_id]["source"]
        valid_passages = {
            passage["passage_id"]
            for passage in source.get("paragraphs", [])
            if isinstance(passage, dict) and isinstance(passage.get("passage_id"), str)
        }
        gold_row = gold_by_id[case_id]
        for issue in gold_row.get("evidence_alignment_issues") or []:
            alignment_issues[str(issue)] += 1
        stats = row.get("rewrite_stats") or {}
        if row.get("rewrite_llm_call_attempted"):
            rewrite_calls += 1
            rewrite_input_tokens += int(stats.get("prompt_eval_count") or 0)
            rewrite_output_tokens += int(stats.get("eval_count") or 0)
        metrics_by_variant = {}
        for variant in variants:
            snapshot = row["variants"].get(variant)
            if not isinstance(snapshot, dict):
                metrics_by_variant[variant] = None
                continue
            candidate = _retrieval_evidence_metrics(
                gold_row, snapshot["candidate_passage_ids"], valid_passages
            )
            context = _retrieval_evidence_metrics(
                gold_row, snapshot["context_passage_ids"], valid_passages
            )
            if candidate is None or context is None:
                metrics_by_variant[variant] = None
                continue
            values = {
                "candidate_reference_aware_recall": candidate["reference_aware_recall"],
                "candidate_union_recall": candidate["union_recall"],
                "context_reference_aware_recall": context["reference_aware_recall"],
                "context_union_recall": context["union_recall"],
                "complete_evidence_set_coverage": context["complete_evidence_set_coverage"],
            }
            metrics_by_variant[variant] = values
            grouped.setdefault((row["direction"], variant), []).append(
                {
                    "case_id": case_id,
                    "document_cluster": row["document_cluster"],
                    **values,
                }
            )
            component_ms = float(snapshot.get("retrieval_elapsed_ms") or 0)
            if variant != "original_hybrid" and row.get("rewrite_llm_call_attempted"):
                component_ms += float(row.get("rewrite_elapsed_ms") or 0)
            timing[variant].append(component_ms)
        eligible = metrics_by_variant["original_hybrid"] is not None
        if not eligible:
            if gold_row.get("unanswerable") is True:
                reason = "unanswerable"
            elif gold_row.get("unanswerable") is None:
                reason = "no_gold_for_source_track"
            elif not valid_passages:
                reason = "passage_mapping_unavailable"
            else:
                reason = "no_complete_mappable_reference"
            eligibility_reasons[reason] += 1
        details.append(
            {
                "case_id": case_id,
                "document_cluster": row["document_cluster"],
                "direction": row["direction"],
                "eligible": eligible,
                "rewrite_status": row["rewrite_status"],
                "dense_route_status": row["dense_route_status"],
                "metrics_by_variant": metrics_by_variant,
                "alignment_issues": gold_row.get("evidence_alignment_issues"),
            }
        )

    directions = sorted({row["direction"] for row in rows})
    by_direction = {}
    paired_deltas = {}
    metric_names = (
        "candidate_reference_aware_recall",
        "candidate_union_recall",
        "context_reference_aware_recall",
        "context_union_recall",
        "complete_evidence_set_coverage",
    )
    for direction in directions:
        by_direction[direction] = {}
        for variant in variants:
            values = grouped.get((direction, variant), [])
            by_direction[direction][variant] = {
                "eligible_case_count": len(values),
                "eligible_cluster_count": len({item["document_cluster"] for item in values}),
                **{
                    metric: sum(item[metric] for item in values) / len(values) if values else None
                    for metric in metric_names
                },
            }
        baseline = {
            item["case_id"]: item for item in grouped.get((direction, "original_hybrid"), [])
        }
        paired_deltas[direction] = {}
        for variant in variants[1:]:
            alternate = {item["case_id"]: item for item in grouped.get((direction, variant), [])}
            common = sorted(baseline.keys() & alternate.keys())
            paired_deltas[direction][variant] = {}
            for metric in (
                "candidate_reference_aware_recall",
                "context_reference_aware_recall",
                "complete_evidence_set_coverage",
            ):
                paired = [
                    {
                        "document_cluster": baseline[case_id]["document_cluster"],
                        "value": alternate[case_id][metric] - baseline[case_id][metric],
                    }
                    for case_id in common
                ]
                paired_deltas[direction][variant][metric] = _cluster_bootstrap_mean_ci(
                    paired, seed=20261028
                )

    eligible_directions = [
        direction
        for direction in ("en->en", "zh->en")
        if any(case["direction"] == direction and case["dataset"] == "qasper" for case in inputs)
    ]
    macro_context = {}
    macro_coverage = {}
    for variant in variants:
        context_values = [
            by_direction[direction][variant]["context_reference_aware_recall"]
            for direction in eligible_directions
            if by_direction[direction][variant]["context_reference_aware_recall"] is not None
        ]
        coverage_values = [
            by_direction[direction][variant]["complete_evidence_set_coverage"]
            for direction in eligible_directions
            if by_direction[direction][variant]["complete_evidence_set_coverage"] is not None
        ]
        macro_context[variant] = (
            sum(context_values) / len(context_values) if context_values else None
        )
        macro_coverage[variant] = (
            sum(coverage_values) / len(coverage_values) if coverage_values else None
        )

    query_usage_path = grid_dir / "query-embedding-usage.json"
    query_usage = json.loads(query_usage_path.read_text(encoding="utf-8"))["usage"]
    measured_cost = {}
    for variant in variants:
        rewritten = variant != "original_hybrid"
        dense_rewrite = variant in ("rewrite_dense_only", "rewrite_both")
        measured_cost[variant] = {
            "shared_rewrite_model_calls": rewrite_calls if rewritten else 0,
            "shared_rewrite_input_tokens": rewrite_input_tokens if rewritten else 0,
            "shared_rewrite_output_tokens": rewrite_output_tokens if rewritten else 0,
            "rewritten_query_embedding_tokens": (
                query_usage["query_embedding_tokens"] if dense_rewrite else 0
            ),
            "component_latency_ms": {
                "p50": _percentile(timing[variant], 0.50),
                "p95": _percentile(timing[variant], 0.95),
            },
            "latency_scope": (
                "recorded retrieval and shared rewrite components; excludes server startup "
                "and answer generation"
            ),
        }
    best_context = max(
        (value for value in macro_context.values() if value is not None), default=None
    )
    recall_tied = [
        variant
        for variant in variants
        if macro_context[variant] is not None and best_context - macro_context[variant] < 0.01
    ]
    best_coverage = max((macro_coverage[name] for name in recall_tied), default=None)
    finalists = [
        name
        for name in recall_tied
        if best_coverage is not None and macro_coverage[name] == best_coverage
    ]
    selected = min(
        finalists or recall_tied or ["original_hybrid"],
        key=lambda name: (
            measured_cost[name]["shared_rewrite_model_calls"],
            measured_cost[name]["shared_rewrite_input_tokens"]
            + measured_cost[name]["shared_rewrite_output_tokens"],
            measured_cost[name]["rewritten_query_embedding_tokens"],
            float("inf")
            if measured_cost[name]["component_latency_ms"]["p50"] is None
            else measured_cost[name]["component_latency_ms"]["p50"],
        ),
    )
    regression = any(
        (
            by_direction[direction][selected]["context_reference_aware_recall"] is not None
            and by_direction[direction]["original_hybrid"]["context_reference_aware_recall"]
            - by_direction[direction][selected]["context_reference_aware_recall"]
            > 0.05
        )
        or (
            by_direction[direction][selected]["complete_evidence_set_coverage"] is not None
            and by_direction[direction]["original_hybrid"]["complete_evidence_set_coverage"]
            - by_direction[direction][selected]["complete_evidence_set_coverage"]
            > 0.02
        )
        for direction in eligible_directions
    )
    if regression or best_context is None:
        selected = "original_hybrid"
    score = {
        "state": "complete",
        "protocol": "full-sentence-rewrite-routing/v1",
        "source_manifest_sha256": manifest["files"]["gold.jsonl"]["sha256"],
        "retrieval_grid_sha256": sha256(grid_path),
        "sample_count": len(inputs),
        "eligible_evidence_case_count": sum(item["eligible"] for item in details),
        "eligible_cluster_count": len(
            {item["document_cluster"] for item in details if item["eligible"]}
        ),
        "eligibility_reasons": dict(eligibility_reasons),
        "alignment_issue_counts": dict(alignment_issues),
        "metric_contract": {
            "candidate_recall": (
                "reference-aware max recall over complete mappable QASPER evidence "
                "alternatives in logical top-20 branch union"
            ),
            "context_recall": (
                "reference-aware max recall over complete mappable evidence alternatives "
                "in top-10 retrieved chunks before generation token admission"
            ),
            "complete_evidence_set_coverage": (
                "whether top-10 context contains one entire mappable gold evidence set"
            ),
            "bootstrap": "paired source-cluster bootstrap, 5000 replicates, seed 20261028",
            "unanswerable_and_unmapped": "N/A; never imputed as zero",
            "answer_quality": "not measured in this retrieval-only stage",
        },
        "by_direction": by_direction,
        "paired_delta_vs_original_hybrid": paired_deltas,
        "measured_cost_by_variant": measured_cost,
        "selection": {
            "eligible_directions": eligible_directions,
            "macro_context_reference_aware_recall": macro_context,
            "macro_complete_evidence_set_coverage": macro_coverage,
            "selected_variant": selected,
            "selection_state": "selected_from_mappable_qasper_evidence"
            if best_context is not None
            else "not_evaluable_keep_original_hybrid_control",
            "regression_gate_reverted_to_original": regression,
            "rule": (
                "maximize context recall; variants within 0.01 are tied, then maximize "
                "complete evidence coverage and minimize measured rewrite, embedding, and "
                "latency cost; revert to original if any eligible direction loses >0.05 "
                "recall or >0.02 coverage"
            ),
            "evidence_scope": (
                "QASPER supports en->en and zh->en metrics; 12 MultiFieldQA cases are "
                "descriptive zh->zh/en->zh route diagnostics only"
            ),
        },
        "case_details": details,
    }
    atomic_json_once(score_path, score)
    return score


def _llama_config(static: dict, *, num_predict: int = OUTPUT_TOKENS) -> LlamaConfig:
    model_size = static["models"]["generator"]["size"]
    spec = model_spec(model_size)
    bundle = (
        DEFAULT_BUNDLE
        if model_size == "9b"
        else ROOT / "models" / "llama-qwen35-4b" / f"{LLAMA_COMMIT[:8]}-{spec['revision'][:8]}"
    )
    return LlamaConfig(
        bundle_dir=bundle,
        model_size=model_size,
        device="cuda",
        gpu_uuid=GPU_UUID,
        num_ctx=CONTEXT_TOKENS,
        num_predict=num_predict,
        template_reserve=TEMPLATE_RESERVE,
        timeout_seconds=600,
        max_gpu_memory_mib=MAX_GPU_MEMORY_MIB,
        max_gpu_temperature_c=GPU_TEMPERATURE_LIMIT_C,
    )


def _longbench_v2_admission(model, search: SearchResult, case: dict) -> dict:
    from page_cite_rag.generation import ChatMessage

    ranked = list(enumerate(search.hits[:TOP_K], start=1))
    selected = list(ranked)
    omitted = []
    while True:
        source_order = sorted(selected, key=lambda pair: (pair[1].block.ordinal, pair[0]))
        retrieved_context = [
            {
                "retrieval_rank": rank,
                "c_idx": hit.block.ordinal,
                "block_id": hit.block.block_id,
                "content": hit.block.text,
            }
            for rank, hit in source_order
        ]
        context = "\n\n".join(
            f"Retrieved chunk {index}: {item['content']}"
            for index, item in enumerate(retrieved_context, start=1)
        )
        prompt = _longbench_v2_official_prompt(
            context,
            case["benchmark_question"],
            case["choices"],
        )
        messages = (ChatMessage(role="user", content=prompt),)
        try:
            budget = model.prepare_budget(messages, None)
            break
        except ContextBudgetError:
            if not selected:
                raise
            lowest_rank = max(selected, key=lambda pair: pair[0])
            selected.remove(lowest_rank)
            omitted.append(
                {
                    "retrieval_rank": lowest_rank[0],
                    "block_id": lowest_rank[1].block.block_id,
                    "reason": "exact_input_budget",
                }
            )
    return {
        "messages": messages,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "exact_input_tokens": budget.exact_input_tokens,
        "retrieved_block_ids": [hit.block.block_id for _, hit in ranked],
        "admitted_block_ids": [item["block_id"] for item in retrieved_context],
        "omitted_block_ids": [item["block_id"] for item in omitted],
        "retrieved_context": retrieved_context,
        "omitted_context": omitted,
    }


def _preflight_longbench_official(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    from page_cite_rag.llama import LlamaModelClient

    if static["run_selection"]["arms"] != [LONG_BENCH_OFFICIAL_ARM]:
        raise ValueError("LongBench official-adapter preflight must be isolated")
    store, retrieval = _load_prepared(run_root, static)
    preflight_path = run_root / "preflight.json"
    if preflight_path.exists():
        saved = json.loads(preflight_path.read_text(encoding="utf-8"))
        if saved.get("static_configuration_sha256") != canonical_hash(static):
            raise ValueError("LongBench preflight was made with different settings")
        return saved

    report = {}
    started = time.monotonic()
    with LlamaModelClient(
        _llama_config(static, num_predict=128),
        budget_policy="exact",
        recover_output_errors=False,
        sampling_profile="longbench-official",
        sampling_seed=42,
    ) as model:
        identity = model.identity
        for index, case in enumerate(inputs, start=1):
            row = retrieval[case["case_id"]]
            search = _search_result_from_json(row["result"])
            admission = _longbench_v2_admission(model, search, case)
            report[case["case_id"]] = {
                "question_sha256": hashlib.sha256(
                    case["benchmark_question"].encode("utf-8")
                ).hexdigest(),
                "retrieval_sha256": canonical_hash(row["result"]),
                "prompt_sha256": admission["prompt_sha256"],
                "exact_input_tokens": admission["exact_input_tokens"],
                "retrieved_block_ids": admission["retrieved_block_ids"],
                "admitted_block_ids": admission["admitted_block_ids"],
                "omitted_block_ids": admission["omitted_block_ids"],
                "retrieved_context": [
                    {key: item[key] for key in ("retrieval_rank", "c_idx", "block_id")}
                    for item in admission["retrieved_context"]
                ],
                "omitted_context": admission["omitted_context"],
            }
            if index % 25 == 0:
                print(f"LongBench exact preflight {index}/{len(inputs)}", flush=True)
        gpu = _safe_model_diagnostics(model)
    result = {
        "state": "preflight_complete_no_generation",
        "static_configuration_sha256": canonical_hash(static),
        "generator_identity": identity,
        "cases": report,
        "gpu": gpu,
        "wall_seconds": time.monotonic() - started,
        "completion_calls": 0,
    }
    atomic_json_once(preflight_path, result)
    return result


def _preflight_reading_plan(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    preflight_path = run_root / "preflight.json"
    if preflight_path.exists():
        saved = json.loads(preflight_path.read_text(encoding="utf-8"))
        if saved.get("static_configuration_sha256") != canonical_hash(static):
            raise ValueError("reading-plan preflight was made with different settings")
        return saved
    store, _base_retrieval = _load_prepared(run_root, static)
    plans = _load_reading_plans(run_root, static, inputs)
    planned = _load_planned_retrieval(run_root, static, inputs)
    shared_reranks = (
        _load_planned_initial_reranks(run_root, static, inputs)
        if static["retrieval"].get("shared_initial_reranker")
        else None
    )
    from page_cite_rag.application import FrozenSearchRetriever, document_workflow

    report = {}
    started = time.monotonic()
    with LlamaModelClient(
        _llama_config(static), budget_policy="exact", recover_output_errors=False
    ) as model:
        identity = model.identity
        for case in inputs:
            case_id = case["case_id"]
            by_arm = {}
            plan = (
                ReadingPlan.model_validate_json(
                    json.dumps(plans[case_id]["plan"], ensure_ascii=False), strict=True
                )
                if plans[case_id]["status"] == "complete"
                else None
            )
            for arm in QUALITY_V1_ARMS:
                if arm != "fixed-rag" and planned[case_id]["status"] not in (
                    "complete",
                    "degraded",
                ):
                    by_arm[arm] = {"status": "planned_retrieval_failed"}
                    continue
                row = planned[case_id]
                pool_name = "original_query" if arm == "fixed-rag" else "decomposed"
                result_value = (
                    shared_reranks[f"{case_id}::{pool_name}"]["result"]
                    if shared_reranks is not None
                    else row["original_query_result"]
                    if arm == "fixed-rag"
                    else row["result"]
                )
                search = _search_result_from_json(result_value)
                top_k = _context_group_limit(static)
                workflow = document_workflow(
                    store,
                    row["document_id"],
                    FrozenSearchRetriever(search, top_k=top_k),
                    model,
                    model.config,
                    top_k=top_k,
                    answer_format=_answer_format_for(case),
                )
                if arm != "fixed-rag" and plan is not None:
                    workflow = workflow.with_reading_plan(plan)
                blocks = workflow._context_blocks(
                    search, workflow._read_hits(search, case["question"])
                )
                admitted, omitted, messages, budget = workflow._context(
                    case["question"],
                    case["language"],
                    "local",
                    blocks,
                    None,
                    context_groups=search.context_groups,
                )
                if (
                    budget.exact_input_tokens is None
                    or budget.exact_input_tokens > EXACT_INPUT_LIMIT
                ):
                    raise ContextBudgetError("reading-plan answer failed exact input preflight")
                by_arm[arm] = {
                    "status": "ready",
                    "reading_plan_status": (
                        plans[case_id]["status"] if arm != "fixed-rag" else "not_used"
                    ),
                    "document_id": row["document_id"],
                    "retrieved_block_ids": [hit.block.block_id for hit in search.hits],
                    "admitted_block_ids": [block.block_id for block in admitted],
                    "omitted_block_ids": list(omitted),
                    "exact_final_input_tokens": budget.exact_input_tokens,
                    "prompt_sha256": canonical_hash([message.model_dump() for message in messages]),
                    "search_sha256": canonical_hash(result_value),
                    "initial_reranker_pool": pool_name if shared_reranks is not None else None,
                }
            if (
                all(
                    by_arm[arm].get("status") == "ready"
                    for arm in (
                        "fixed-rag-followup-once",
                        "decomposition-fixed-rag",
                        "decomposition-agent",
                    )
                )
                and len(
                    {
                        tuple(by_arm[arm]["admitted_block_ids"])
                        for arm in (
                            "fixed-rag-followup-once",
                            "decomposition-fixed-rag",
                            "decomposition-agent",
                        )
                    }
                )
                != 1
            ):
                raise ValueError("decomposition/follow-up arms do not share initial evidence")
            report[case_id] = {"by_arm": by_arm}
        diagnostics = _safe_model_diagnostics(model)
    result = {
        "state": "preflight_complete_no_generation",
        "protocol": "question-needs-and-bilingual-query-plan/v2",
        "static_configuration_sha256": canonical_hash(static),
        "generator_identity": identity,
        "planning_results_sha256": sha256(run_root / "reading-plans/results.jsonl"),
        "planned_retrieval_results_sha256": sha256(run_root / "planned-retrieval/results.jsonl"),
        "cases": report,
        "gpu": diagnostics,
        "wall_seconds": time.monotonic() - started,
        "completion_calls": 0,
    }
    atomic_json_once(preflight_path, result)
    return result


def _preflight(run_root: Path, static: dict, inputs: list[dict]) -> dict:
    if static.get("reading_plan") is not None:
        return _preflight_reading_plan(run_root, static, inputs)
    if static["run_selection"]["arms"] == [LONG_BENCH_OFFICIAL_ARM]:
        return _preflight_longbench_official(run_root, static, inputs)
    store, retrieval = _load_prepared(run_root, static)
    preflight_path = run_root / "preflight.json"
    if preflight_path.exists():
        saved = json.loads(preflight_path.read_text(encoding="utf-8"))
        if saved.get("static_configuration_sha256") != canonical_hash(static):
            raise ValueError("preflight was made with different settings")
        return saved

    if static["retrieval"].get("reranker"):
        return _preflight_reranked(run_root, static, inputs, store, retrieval)

    from page_cite_rag.application import FrozenSearchRetriever, document_workflow

    report = {}
    started = time.monotonic()
    with LlamaModelClient(
        _llama_config(static),
        budget_policy="exact",
        recover_output_errors=False,
        sampling_profile=static["answer"]["sampling_profile"],
    ) as model:
        generator_identity = model.identity
        for case in inputs:
            row = retrieval[case["case_id"]]
            search = _search_result_from_json(row["result"])
            prompt_preflights = {}
            workflows = {}
            context_blocks = None
            revisions = list(dict.fromkeys(static["answer"]["prompt_revisions_by_arm"].values()))
            for revision in revisions:
                workflow = document_workflow(
                    store,
                    row["document_id"],
                    FrozenSearchRetriever(search, top_k=TOP_K),
                    model,
                    model.config,
                    top_k=TOP_K,
                    answer_format=_answer_format_for(case),
                    prompt_revision=revision,
                )
                blocks = context_blocks
                if blocks is None:
                    blocks = workflow._context_blocks(
                        search, workflow._read_hits(search, case["question"])
                    )
                    context_blocks = blocks
                workflows[revision] = workflow
                admitted, omitted, _messages, budget = workflow._context(
                    case["question"], case["language"], "local", blocks, None
                )
                if (
                    budget.exact_input_tokens is None
                    or budget.exact_input_tokens > EXACT_INPUT_LIMIT
                ):
                    raise ContextBudgetError("fixed final-answer prompt failed exact preflight")
                prompt_preflights[revision] = {
                    "admitted_block_ids": [block.block_id for block in admitted],
                    "prompt_budget_omitted_block_ids": list(omitted),
                }
            retrieved_block_ids = [hit.block.block_id for hit in search.hits]
            common_ids = _common_admitted_block_ids(prompt_preflights, retrieved_block_ids)
            common_set = set(common_ids)
            common_blocks = tuple(block for block in context_blocks if block.block_id in common_set)
            if [block.block_id for block in common_blocks] != common_ids:
                raise ValueError("common evidence order differs from the retrieved context")
            for revision, workflow in workflows.items():
                budget_admitted_ids = prompt_preflights[revision]["admitted_block_ids"]
                admitted, omitted, _messages, budget = workflow._context(
                    case["question"],
                    case["language"],
                    "local",
                    common_blocks,
                    None,
                )
                admitted_ids = [block.block_id for block in admitted]
                if admitted_ids != common_ids or omitted:
                    raise ValueError("a prompt revision cannot admit the frozen common evidence")
                prompt_preflights[revision].update(
                    {
                        "exact_final_input_tokens": budget.exact_input_tokens,
                        "admitted_block_ids": common_ids,
                        "prompt_budget_admitted_block_ids": budget_admitted_ids,
                        "retrieved_block_ids": retrieved_block_ids,
                        "omitted_block_ids": [
                            block_id
                            for block_id in retrieved_block_ids
                            if block_id not in common_set
                        ],
                    }
                )
            report[case["case_id"]] = {
                "document_id": row["document_id"],
                "common_admitted_block_ids": common_ids,
                "by_prompt_revision": prompt_preflights,
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


def _row_for_arm(row: dict, arm: str, *, shared_reranker: bool = False) -> dict:
    if arm != "fixed-rag-reranked" and not shared_reranker:
        return row
    result = row["reranked_result"]
    mapping = row.get("block_to_passage_id", {})
    return {
        **row,
        "result": result,
        "retrieved_passage_ids": list(
            dict.fromkeys(
                mapping[hit["block"]["block_id"]]
                for hit in result["hits"]
                if hit["block"]["block_id"] in mapping
            )
        ),
    }


def _fixed_followup_search(
    initial: SearchResult,
    candidate_pool: SearchResult,
    *,
    initial_admitted_block_ids: list[str],
    limit: int = FIXED_FOLLOWUP_BLOCKS,
) -> SearchResult | None:
    """Select source groups outside the context that the first writer actually received."""
    if type(limit) is not int or limit < 1:
        raise ValueError("fixed follow-up candidate limit must be positive")
    if (
        initial.document_id != candidate_pool.document_id
        or initial.query != candidate_pool.query
        or not candidate_pool.index_id
    ):
        raise ValueError("fixed follow-up candidate pool differs from the initial search")
    initial_hit_ids = [hit.block.block_id for hit in initial.hits]
    candidate_ids = [hit.block.block_id for hit in candidate_pool.hits]
    if (
        not isinstance(initial_admitted_block_ids, list)
        or any(
            not isinstance(block_id, str) or not block_id for block_id in initial_admitted_block_ids
        )
        or len(set(initial_admitted_block_ids)) != len(initial_admitted_block_ids)
        or len(set(initial_hit_ids)) != len(initial_hit_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
        or set(initial_hit_ids) - set(candidate_ids)
        or [hit.rank for hit in candidate_pool.hits] != list(range(1, len(candidate_ids) + 1))
        or any(hit.block.document_id != initial.document_id for hit in candidate_pool.hits)
    ):
        raise ValueError("fixed follow-up candidate pool has invalid scope or ranking")
    groups_by_block: dict[str, list[SearchContextGroup]] = {}
    group_ids = set()
    candidate_universe = set(candidate_ids)
    for group in candidate_pool.context_groups:
        if (
            group.chunk_index in group_ids
            or not group.source_block_ids
            or len(set(group.source_block_ids)) != len(group.source_block_ids)
        ):
            raise ValueError("fixed follow-up context group is outside the frozen candidate pool")
        group_ids.add(group.chunk_index)
        candidate_universe.update(group.source_block_ids)
        for block_id in group.source_block_ids:
            groups_by_block.setdefault(block_id, []).append(group)
    if not set(initial_admitted_block_ids).issubset(candidate_universe):
        raise ValueError("initial writer context is absent from the fixed candidate pool")

    excluded = set(initial_admitted_block_ids)
    selected_hits: list[SearchHit] = []
    selected_groups: list[SearchContextGroup] = []
    selected_group_ids: set[int] = set()
    selected_ids: set[str] = set()
    for hit in candidate_pool.hits:
        block_id = hit.block.block_id
        if block_id in excluded or block_id in selected_ids:
            continue
        group = next(
            (
                item
                for item in groups_by_block.get(block_id, ())
                if item.chunk_index not in selected_group_ids
                and set(item.source_block_ids) - excluded
            ),
            None,
        )
        if group is None:
            selected_hits.append(hit)
            selected_ids.add(block_id)
        else:
            selected_groups.append(group)
            selected_group_ids.add(group.chunk_index)
            selected_hits.append(hit)
            selected_ids.update(group.source_block_ids)
        if (
            len(selected_groups) + (0 if candidate_pool.context_groups else len(selected_hits))
            >= limit
        ):
            break
    if not selected_hits:
        return None
    return candidate_pool.model_copy(
        update={
            "hits": tuple(
                hit.model_copy(update={"rank": rank})
                for rank, hit in enumerate(selected_hits, start=1)
            ),
            "context_groups": tuple(selected_groups),
        }
    )


def _fixed_followup_admission(
    workflow,
    question: str,
    language: str,
    initial_context,
    followup_search: SearchResult,
    followup_blocks: tuple,
):
    """Fit a follow-up while dropping only the lowest-ranked initial context units."""
    from page_cite_rag.agent import _admit

    initial_blocks = tuple(initial_context.blocks)
    initial_ids = {block.block_id for block in initial_blocks}
    new_ids = {
        hit.block.block_id for hit in followup_search.hits if hit.block.block_id not in initial_ids
    }
    if followup_search.context_groups:
        new_ids.update(
            block_id
            for group in followup_search.context_groups
            for block_id in group.source_block_ids
            if block_id not in initial_ids
        )
    if not new_ids:
        return None, {"state": "no_new_source_ids"}

    initial_groups = tuple(initial_context.context_groups)
    followup_groups = tuple(followup_search.context_groups)
    if bool(initial_groups) != bool(followup_groups):
        raise ValueError("fixed follow-up source context type changed within one document")
    initial_units = (
        tuple((group.source_block_ids, group) for group in initial_groups)
        if initial_groups
        else tuple(((block.block_id,), None) for block in initial_blocks)
    )
    followup_units = (
        tuple((group.source_block_ids, group) for group in followup_groups)
        if followup_groups
        else tuple(((hit.block.block_id,), None) for hit in followup_search.hits)
    )
    blocks_by_id = {block.block_id: block for block in (*initial_blocks, *followup_blocks)}
    if len(blocks_by_id) != len(initial_blocks) + len(followup_blocks):
        blocks_by_id = {block.block_id: block for block in (*initial_blocks, *followup_blocks)}
    if any(
        block_id not in blocks_by_id
        for ids, _ in (*initial_units, *followup_units)
        for block_id in ids
    ):
        raise ValueError("fixed follow-up group is missing its verified original blocks")

    for keep_count in range(len(initial_units), -1, -1):
        kept_initial = initial_units[:keep_count]
        combined_units = (*kept_initial, *followup_units)
        ordered_ids = tuple(
            dict.fromkeys(block_id for ids, _ in combined_units for block_id in ids)
        )
        groups_by_index = {
            group.chunk_index: group for _, group in combined_units if group is not None
        }
        context_groups = tuple(groups_by_index.values())
        blocks = tuple(blocks_by_id[block_id] for block_id in ordered_ids)
        admitted = _admit(workflow, question, language, blocks, context_groups)
        admitted_ids = {block.block_id for block in admitted.blocks}
        kept_ids = {block_id for ids, _ in kept_initial for block_id in ids}
        added_ids = new_ids & admitted_ids
        if kept_ids.issubset(admitted_ids) and added_ids:
            return admitted, {
                "state": "completed",
                "kept_initial_unit_count": keep_count,
                "dropped_initial_unit_count": len(initial_units) - keep_count,
                "initial_block_ids_preserved": sorted(kept_ids),
                "new_block_ids_admitted": sorted(added_ids),
            }
    return None, {
        "state": "no_new_blocks_fit_exact_budget",
        "initial_unit_count": len(initial_units),
        "candidate_new_block_ids": sorted(new_ids),
    }


def _preflight_for_arm(preflight: dict, case_id: str, arm: str, revision: str) -> dict:
    row = preflight["cases"][case_id]
    return row["by_arm"][arm] if "by_arm" in row else row["by_prompt_revision"][revision]


def _prompt_budget_admitted_block_ids(preflight_case: dict) -> list[str]:
    """Read either preflight schema while preserving the exact admitted evidence."""
    value = preflight_case.get(
        "prompt_budget_admitted_block_ids", preflight_case["admitted_block_ids"]
    )
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("preflight prompt-admitted block IDs are malformed")
    return value


def _preflight_reranked(run_root, static, inputs, store, retrieval):
    from page_cite_rag.application import FrozenSearchRetriever, document_workflow

    report = {}
    started = time.monotonic()
    with LlamaModelClient(_llama_config(static), budget_policy="exact") as model:
        identity = model.identity
        for case in inputs:
            by_arm = {}
            for arm in static["run_selection"]["arms"]:
                row = _row_for_arm(
                    retrieval[case["case_id"]],
                    arm,
                    shared_reranker=static.get("retrieval", {}).get("reranker", False),
                )
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
                blocks = workflow._context_blocks(
                    search, workflow._read_hits(search, case["question"])
                )
                admitted, omitted, _, budget = workflow._context(
                    case["question"], case["language"], "local", blocks, None
                )
                ids = [block.block_id for block in admitted]
                if (
                    not ids
                    or budget.exact_input_tokens is None
                    or budget.exact_input_tokens > EXACT_INPUT_LIMIT
                ):
                    raise ContextBudgetError("reranker comparison context failed exact preflight")
                by_arm[arm] = {
                    "admitted_block_ids": ids,
                    "prompt_budget_admitted_block_ids": ids,
                    "omitted_block_ids": list(omitted),
                    "retrieved_block_ids": [hit.block.block_id for hit in search.hits],
                    "exact_final_input_tokens": budget.exact_input_tokens,
                }
            if len({tuple(value["admitted_block_ids"]) for value in by_arm.values()}) > 1:
                raise ValueError("comparison arms do not share identical admitted evidence")
            report[case["case_id"]] = {"by_arm": by_arm}
        gpu = _safe_model_diagnostics(model)
    result = {
        "state": "preflight_complete_no_generation",
        "static_configuration_sha256": canonical_hash(static),
        "generator_identity": identity,
        "cases": report,
        "gpu": gpu,
        "wall_seconds": time.monotonic() - started,
        "completion_calls": 0,
    }
    atomic_json_once(run_root / "preflight.json", result)
    return result


@contextmanager
def _owned_case_deadline(model, retriever, remaining_seconds: float):
    """Cancel this case's owned retrieval and generator if its deadline expires."""
    expired = threading.Event()
    set_deadline = getattr(model, "set_request_deadline", None)
    deadline = time.monotonic() + max(0.0, remaining_seconds)
    if callable(set_deadline):
        set_deadline(deadline)

    def cancel_owned_work() -> None:
        expired.set()
        cancel_retrieval = getattr(retriever, "cancel", None)
        try:
            if callable(cancel_retrieval):
                cancel_retrieval()
        finally:
            model.close()

    timer = threading.Timer(max(0.0, remaining_seconds), cancel_owned_work)
    timer.daemon = True
    timer.start()
    try:
        yield expired
    finally:
        timer.cancel()
        timer.join(timeout=5)
        clear_deadline = getattr(model, "clear_request_deadline", None)
        if callable(clear_deadline):
            clear_deadline()


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
        "last_gpu_check": value.get("last_gpu_check"),
        "last_dispatch_wait": value.get("last_dispatch_wait"),
        "dispatch_wait_seconds_total": value.get("dispatch_wait_seconds_total", 0.0),
        "server_exit_code": value.get("exit_code_before_termination"),
    }


def _gpu_guard_detected(*diagnostics: object) -> bool:
    return any(isinstance(value, dict) and bool(value.get("guard_error")) for value in diagnostics)


def _run_longbench_official_arm(
    run_root: Path,
    static: dict,
    inputs: list[dict],
    case_id: str | None,
) -> dict:
    from page_cite_rag.generation import ChatMessage

    if static["run_selection"]["arms"] != [LONG_BENCH_OFFICIAL_ARM]:
        raise ValueError("LongBench official-adapter run must be isolated")
    preflight_path = run_root / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError("run exact LongBench prompt preflight before generation")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("static_configuration_sha256") != canonical_hash(static):
        raise ValueError("LongBench preflight differs from the frozen configuration")
    store, retrieval = _load_prepared(run_root, static)
    selected = [case for case in inputs if case_id is None or case["case_id"] == case_id]
    if not selected or (case_id is not None and selected[0]["case_id"] != case_id):
        raise ValueError("requested case ID is not in the frozen input manifest")

    arm_dir = run_root / LONG_BENCH_OFFICIAL_ARM
    attempts_dir = arm_dir / ".attempts"
    arm_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempts_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = time.monotonic()
    attempted = completed = 0
    stopped = None
    with LlamaModelClient(
        _llama_config(static, num_predict=128),
        budget_policy="exact",
        recover_output_errors=True,
        sampling_profile="longbench-official",
        sampling_seed=42,
    ) as model:
        if model.identity != preflight["generator_identity"]:
            raise ValueError("LongBench generator differs from exact preflight")
        for case in selected:
            target = arm_dir / f"{case['case_id']}.json"
            attempt_marker = attempts_dir / f"{case['case_id']}.json"
            if target.exists() or attempt_marker.exists():
                continue
            if time.monotonic() - started >= LONG_BENCH_MAX_INVOCATION_WALL_SECONDS:
                stopped = "invocation_wall_limit"
                break

            case_started = time.monotonic()
            case_preflight = preflight["cases"][case["case_id"]]
            row = retrieval[case["case_id"]]
            search = _search_result_from_json(row["result"])
            retrieval_by_id = {
                hit.block.block_id: (rank, hit) for rank, hit in enumerate(search.hits[:TOP_K], 1)
            }
            admitted_ids = case_preflight["admitted_block_ids"]
            if any(block_id not in retrieval_by_id for block_id in admitted_ids):
                raise ValueError("LongBench preflight selected a block outside top-10 retrieval")
            ordered = sorted(
                (retrieval_by_id[block_id] for block_id in admitted_ids),
                key=lambda pair: (pair[1].block.ordinal, pair[0]),
            )
            retrieved_context = [
                {
                    "retrieval_rank": rank,
                    "c_idx": hit.block.ordinal,
                    "block_id": hit.block.block_id,
                    "content": hit.block.text,
                }
                for rank, hit in ordered
            ]
            context = "\n\n".join(
                f"Retrieved chunk {index}: {item['content']}"
                for index, item in enumerate(retrieved_context, start=1)
            )
            prompt = _longbench_v2_official_prompt(
                context,
                case["benchmark_question"],
                case["choices"],
            )
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            messages = (ChatMessage(role="user", content=prompt),)
            if (
                prompt_hash != case_preflight["prompt_sha256"]
                or [item["block_id"] for item in retrieved_context] != admitted_ids
            ):
                raise ValueError("LongBench runtime prompt differs from frozen preflight")

            atomic_json_once(
                attempt_marker,
                {
                    "state": "attempt_started_no_retry",
                    "arm": LONG_BENCH_OFFICIAL_ARM,
                    "case_id": case["case_id"],
                    "static_configuration_sha256": canonical_hash(static),
                    "retrieval_sha256": case_preflight["retrieval_sha256"],
                    "prompt_sha256": prompt_hash,
                    "started_monotonic_seconds": time.monotonic(),
                },
            )
            attempted += 1
            generation = None
            failure = None
            accounting = None
            raw_output = None
            call_started = time.perf_counter()
            try:
                generation = model.generate(messages, None)
                raw_output = generation.raw_model_output
                completed += 1
            except Exception as exc:
                failure = {
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:300],
                    "reason": getattr(exc, "reason", None),
                }
                raw_output = getattr(exc, "raw_model_output", None)
                error_accounting = getattr(exc, "accounting", None)
                if error_accounting is not None:
                    accounting = error_accounting.model_dump(mode="json")
                if not isinstance(exc, OutputValidationError):
                    stopped = "model_client_failed"
            call_elapsed_ms = (time.perf_counter() - call_started) * 1000
            diagnostics = _safe_model_diagnostics(model)
            stats = generation.stats.model_dump(mode="json") if generation is not None else None
            observed_input = (
                stats.get("prompt_eval_count")
                if stats is not None
                else accounting.get("prompt_eval_count")
                if isinstance(accounting, dict)
                else None
            )
            observed_output = (
                stats.get("eval_count")
                if stats is not None
                else accounting.get("eval_count")
                if isinstance(accounting, dict)
                else None
            )
            record = {
                "arm": LONG_BENCH_OFFICIAL_ARM,
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "execution": "ok" if generation is not None else "failed",
                "failure": failure,
                "result": None,
                "raw_model_output": raw_output,
                "official_extracted_choice": (
                    _longbench_v2_official_choice(raw_output)
                    if isinstance(raw_output, str)
                    else None
                ),
                "generation_stats": stats,
                "failure_accounting": accounting,
                "completion_calls_attempted": 1,
                "input_tokens_lower_bound": observed_input or 0,
                "input_tokens_exact_or_observed": observed_input,
                "input_usage_unknown": observed_input is None,
                "output_tokens_lower_bound": observed_output or 0,
                "output_usage_unknown": observed_output is None,
                "model_call_elapsed_ms": [call_elapsed_ms],
                "elapsed_ms": (time.monotonic() - case_started) * 1000,
                "retrieval_elapsed_ms": search.elapsed_ms,
                "initial_search_sha256": canonical_hash(row["result"]),
                "context_trace": {
                    "retrieved_block_ids": case_preflight["retrieved_block_ids"],
                    "admitted_block_ids": admitted_ids,
                    "omitted_block_ids": case_preflight["omitted_block_ids"],
                    "exact_input_tokens": case_preflight["exact_input_tokens"],
                    "prompt_sha256": prompt_hash,
                    "retrieved_context": retrieved_context,
                },
                "gpu_diagnostics": diagnostics,
                "no_retry": True,
                "gold_not_loaded": True,
            }
            atomic_json_once(target, record)
            print(
                f"{LONG_BENCH_OFFICIAL_ARM} {case['case_id']}: {record['execution']}, "
                "1 generation attempt",
                flush=True,
            )
            if _gpu_guard_detected(diagnostics):
                stopped = "gpu_guard_stopped"
                break
            if stopped is not None:
                break
            if time.monotonic() - started >= LONG_BENCH_MAX_INVOCATION_WALL_SECONDS:
                stopped = "invocation_wall_limit"
                break

    return {
        "arm": LONG_BENCH_OFFICIAL_ARM,
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
            and (attempts_dir / f"{case['case_id']}.json").is_file()
        ],
    }


class _CountingModelClient:
    """Record every structured stage without altering prompts, schemas or responses."""

    def __init__(
        self,
        model,
        *,
        global_call_ledger: Path | None = None,
        global_call_limit: int | None = None,
        global_call_phase: str | None = None,
    ) -> None:
        if (global_call_ledger is None) != (global_call_limit is None):
            raise ValueError("global model-call ledger and limit must be supplied together")
        if global_call_limit is not None and (
            type(global_call_limit) is not int or global_call_limit < 1
        ):
            raise ValueError("global model-call limit must be a positive integer")
        self._model = model
        self._global_call_ledger = global_call_ledger
        self._global_call_limit = global_call_limit
        self._global_call_phase = global_call_phase
        self.generate_calls = 0
        self.records: list[dict] = []
        self.last_result = None
        self._case_start_calls = 0
        self._case_call_limit: int | None = None
        self._case_required_remaining_calls: int | Callable[[], int] = 0
        self._call_context: dict[str, str] = {}
        self._pre_reserved_global_call: tuple[int, str] | None = None

    def begin_case(
        self,
        *,
        structured_call_limit: int | None = None,
        required_remaining_calls: int | Callable[[], int] = 0,
        call_context: dict[str, str] | None = None,
    ) -> None:
        if self._pre_reserved_global_call is not None:
            raise ModelError("a global model reservation was not consumed by its request")
        self.last_result = None
        self.records = []
        self._case_start_calls = self.generate_calls
        self._case_call_limit = structured_call_limit
        if not callable(required_remaining_calls) and (
            type(required_remaining_calls) is not int or required_remaining_calls < 0
        ):
            raise ValueError("remaining model-call reservation must be non-negative")
        self._case_required_remaining_calls = required_remaining_calls
        self._call_context = dict(call_context or {})

    def _required_remaining_calls(self) -> int:
        value = self._case_required_remaining_calls
        if callable(value):
            value = value()
        if type(value) is not int or value < 0:
            raise ModelError("remaining model-call reservation is invalid")
        return value

    def reserve_before_dispatch(
        self,
        response_model: type,
        *,
        required_remaining_calls: int = 0,
        dynamic_schema: bool = False,
    ) -> tuple[bool, int | None]:
        """Reserve a writer slot before local ledgers mark it as dispatched."""
        if self._pre_reserved_global_call is not None:
            raise ModelError("a global model reservation is already pending")
        if self._global_call_ledger is None:
            return True, None
        try:
            index = self._reserve_global_call(
                response_model, required_remaining_calls=required_remaining_calls
            )
        except ModelError as exc:
            if "ceiling reached" in str(exc):
                return False, None
            raise
        if index is None:
            raise ModelError("global model ledger did not return its reserved dispatch index")
        self._pre_reserved_global_call = (
            index,
            "*" if dynamic_schema else response_model.__name__,
        )
        return True, index

    def _reserve_global_call(
        self, response_model: type, *, required_remaining_calls: int = 0
    ) -> int | None:
        if self._global_call_ledger is None:
            return None
        import fcntl

        path = self._global_call_ledger
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.is_symlink():
            raise ModelError("refusing to use a symlinked model-call ledger")
        lock_path = path.with_name(f".{path.name}.lock")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            rows = read_jsonl(path) if path.is_file() else []
            reservations = {
                row["call_index"]: row
                for row in rows
                if row.get("state") == "reserved_before_dispatch"
                and type(row.get("call_index")) is int
            }
            voided = []
            for row in rows:
                if row.get("state") != "voided_before_dispatch":
                    continue
                indices = row.get("voids_call_indices")
                if indices is None:
                    indices = [row.get("voids_call_index")]
                if not isinstance(indices, list):
                    raise ModelError("global model-call ledger has invalid dispatch corrections")
                voided.extend(indices)
            if (
                any(
                    row.get("state")
                    not in {
                        "reserved_before_dispatch",
                        "voided_before_dispatch",
                        "usage_recorded",
                        "duplicate_observer_correction",
                    }
                    for row in rows
                )
                or any(type(index) is not int for index in voided)
                or len(voided) != len(set(voided))
                or not set(voided).issubset(reservations)
            ):
                raise ModelError("global model-call ledger has invalid dispatch corrections")
            usage_rows = [row for row in rows if row.get("state") == "usage_recorded"]
            usage_indices = [row.get("call_index") for row in usage_rows]
            if (
                any(type(index) is not int or index not in reservations for index in usage_indices)
                or len(usage_indices) != len(set(usage_indices))
                or set(usage_indices) & set(voided)
            ):
                raise ModelError("global model-call ledger has invalid usage records")
            usage_by_index = {row["call_index"]: row for row in usage_rows}
            duplicate_indices = _duplicate_observer_call_indices(rows, reservations, usage_by_index)
            active_indices = set(reservations) - set(voided) - duplicate_indices
            if set(voided) & duplicate_indices:
                raise ModelError("global model-call ledger both voids and duplicates a dispatch")
            if (
                type(required_remaining_calls) is not int
                or required_remaining_calls < 0
                or len(active_indices) + 1 + required_remaining_calls > self._global_call_limit
            ):
                raise ModelError("frozen quality-comparison model-call ceiling reached")
            progress_path = path.parent / "progress.json"
            if progress_path.is_file() and not progress_path.is_symlink():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                revision = progress.get("revision", progress.get("plan_revision"))
                if revision in {
                    "agent-quality-closure-v3",
                    "agent-production-alignment-v4",
                    "agent-control-followup-v1",
                }:
                    limits = (
                        progress.get("generation_dispatch_limits")
                        if revision
                        in {"agent-production-alignment-v4", "agent-control-followup-v1"}
                        else progress.get("scope", {}).get("phase_generation_dispatch_limits")
                    )
                    if (
                        not isinstance(limits, dict)
                        or self._global_call_phase not in limits
                        or type(limits.get(self._global_call_phase)) is not int
                    ):
                        raise ModelError("generation phase or its frozen ceiling is invalid")
                    if progress.get("state") != "in_progress":
                        raise ModelError("generation batch is not active")
                    phase_used = sum(
                        reservations.get(index, {}).get("budget_phase") == self._global_call_phase
                        for index in active_indices
                    )
                    if phase_used + 1 + required_remaining_calls > limits[self._global_call_phase]:
                        raise ModelError("frozen phase generation ceiling reached")
            index = max(reservations, default=0) + 1
            append_jsonl_record(
                path,
                {
                    "call_index": index,
                    "state": "reserved_before_dispatch",
                    **(
                        {"budget_phase": self._global_call_phase}
                        if self._global_call_phase is not None
                        else {}
                    ),
                    "response_model": response_model.__name__,
                    **self._call_context,
                },
            )
            self._refresh_v2_progress(path)
            return index
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def __getattr__(self, name):
        return getattr(self._model, name)

    @staticmethod
    def _refresh_v2_progress(path: Path) -> None:
        progress_path = path.parent / "progress.json"
        if not progress_path.is_file() or progress_path.is_symlink():
            return
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        revision = progress.get("revision", progress.get("plan_revision"))
        if revision not in {
            "quality-repair-closure-v2",
            "agent-quality-closure-v3",
            "agent-production-alignment-v4",
            "agent-control-followup-v1",
        }:
            return
        rows = read_jsonl(path)
        reservations = {
            row["call_index"]: row
            for row in rows
            if row.get("state") == "reserved_before_dispatch" and type(row.get("call_index")) is int
        }
        voided = {
            index
            for row in rows
            if row.get("state") == "voided_before_dispatch"
            for index in row.get("voids_call_indices", [row.get("voids_call_index")])
        }
        usage_rows = [row for row in rows if row.get("state") == "usage_recorded"]
        usage_by_index = {row["call_index"]: row for row in usage_rows}
        duplicate_indices = _duplicate_observer_call_indices(rows, reservations, usage_by_index)
        active_indices = set(reservations) - voided - duplicate_indices
        input_tokens = sum(
            usage_by_index[index].get("input_tokens") or 0
            for index in active_indices
            if index in usage_by_index
        )
        output_tokens = sum(
            usage_by_index[index].get("output_tokens") or 0
            for index in active_indices
            if index in usage_by_index
        )
        counters = progress.setdefault("counters", {})
        dispatches = len(active_indices)
        counters["generation_dispatches"] = dispatches
        input_key, output_key, unknown_key = (
            ("input_tokens", "output_tokens", "unknown_usage")
            if revision in {"agent-production-alignment-v4", "agent-control-followup-v1"}
            else ("generation_input_tokens", "generation_output_tokens", "unknown_generation_usage")
        )
        counters[input_key] = input_tokens
        counters[output_key] = output_tokens
        counters[unknown_key] = sum(
            1
            for index in active_indices
            if index not in usage_by_index or usage_by_index[index].get("usage_unknown") is True
        )
        if revision in {
            "agent-quality-closure-v3",
            "agent-production-alignment-v4",
            "agent-control-followup-v1",
        }:
            reservations_by_index = {index: row for index, row in reservations.items()}
            phase_limits = (
                progress["generation_dispatch_limits"]
                if revision in {"agent-production-alignment-v4", "agent-control-followup-v1"}
                else progress["scope"]["phase_generation_dispatch_limits"]
            )
            counters["generation_dispatches_by_phase"] = {
                phase: sum(
                    reservations_by_index[index].get("budget_phase") == phase
                    for index in active_indices
                )
                for phase in phase_limits
                if phase != "total"
            }
            if "total" in phase_limits:
                counters["generation_dispatches_by_phase"]["total"] = dispatches
        progress["new_model_calls"] = dispatches
        write_state_json(progress_path, progress)

    def _record_global_usage(self, call_index: int | None, record: dict) -> None:
        if call_index is None or self._global_call_ledger is None:
            return
        import fcntl

        accounting = record.get("stats") or record.get("accounting") or {}
        input_tokens = accounting.get("prompt_eval_count")
        output_tokens = accounting.get("eval_count")
        row = {
            "call_index": call_index,
            "state": "usage_recorded",
            "input_tokens": input_tokens if type(input_tokens) is int else None,
            "output_tokens": output_tokens if type(output_tokens) is int else None,
            "usage_unknown": type(input_tokens) is not int or type(output_tokens) is not int,
        }
        path = self._global_call_ledger
        lock_path = path.with_name(f".{path.name}.lock")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            rows = read_jsonl(path) if path.is_file() else []
            reservations = {
                item["call_index"]
                for item in rows
                if item.get("state") == "reserved_before_dispatch"
            }
            usage_rows = [item for item in rows if item.get("state") == "usage_recorded"]
            existing = [item for item in usage_rows if item.get("call_index") == call_index]
            if existing:
                if len(existing) != 1 or existing[0] != row:
                    raise ModelError("global model-call usage record conflicts with its dispatch")
            else:
                if call_index not in reservations:
                    raise ModelError("global model-call usage has no reserved dispatch")
                append_jsonl_record(path, row)
            self._refresh_v2_progress(path)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def generate(self, *args, **kwargs):
        if (
            self._case_call_limit is not None
            and self.generate_calls - self._case_start_calls >= self._case_call_limit
        ):
            raise ModelError("structured call limit reached before dispatch; no retry")
        messages = args[0] if args else kwargs["messages"]
        response_model = args[1] if len(args) > 1 else kwargs["response_model"]
        if self._pre_reserved_global_call is not None:
            global_call_index, reserved_model = self._pre_reserved_global_call
            if reserved_model != "*" and reserved_model != response_model.__name__:
                raise ModelError("reserved model schema differs from the dispatched request")
            self._pre_reserved_global_call = None
        else:
            global_call_index = self._reserve_global_call(
                response_model, required_remaining_calls=self._required_remaining_calls()
            )
        from page_cite_rag.review import SourceReview

        record = {
            "call_index": self.generate_calls - self._case_start_calls + 1,
            "role": "checker" if issubclass(response_model, SourceReview) else "answer",
            "response_model": response_model.__name__,
            "global_call_index": global_call_index,
            "messages": [message.model_dump(mode="json") for message in messages],
            "response_schema": response_model.model_json_schema(),
            "status": "entered_generate",
        }
        self.records.append(record)
        self.generate_calls += 1
        self.last_result = None
        started = time.perf_counter()
        try:
            result = self._model.generate(*args, **kwargs)
        except Exception as exc:
            record.update(
                status="failed_usage_unknown",
                error_type=type(exc).__name__,
                raw_model_output=getattr(exc, "raw_model_output", None),
            )
            accounting = getattr(exc, "accounting", None)
            if accounting is not None:
                record.update(
                    status="failed_accounted", accounting=accounting.model_dump(mode="json")
                )
            if errors := getattr(exc, "validation_errors", ()):
                record["validation_errors"] = list(errors)
            self._record_global_usage(global_call_index, record)
            raise
        finally:
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        self.last_result = result
        record.update(
            status="completed",
            stats=result.stats.model_dump(mode="json"),
            raw_model_output=result.raw_model_output,
            structured_value=result.value.model_dump(mode="json"),
        )
        self._record_global_usage(global_call_index, record)
        return result


def _remaining_quality_writer_reservation(
    run_root: Path,
    static: dict,
    *,
    current_arm: str,
    current_case_id: str,
    current_case_remaining_calls: int = 0,
) -> int:
    """Protect every not-yet-started fixed writer and the current/future Agent writers."""
    if type(current_case_remaining_calls) is not int or current_case_remaining_calls < 0:
        raise ModelError("current case writer reservation is invalid")
    selected_arms = tuple(static.get("run_selection", {}).get("arms", ()))
    total = current_case_remaining_calls
    if selected_arms == ("single-agent",):
        directory = run_root / "single-agent"
        for case_id in static["run_selection"]["case_ids"]:
            if case_id == current_case_id:
                continue
            if (directory / f"{case_id}.json").exists():
                continue
            if (directory / ".attempts" / f"{case_id}.json").exists():
                continue
            total += 1
        return total
    if selected_arms != QUALITY_V1_ARMS:
        raise ModelError("writer reservation requires a supported single-agent or four-arm run")
    for arm in QUALITY_V1_ARMS:
        directory = run_root / arm
        for case_id in static["run_selection"]["case_ids"]:
            if arm == current_arm and case_id == current_case_id:
                continue
            result_path = directory / f"{case_id}.json"
            attempt_path = directory / ".attempts" / f"{case_id}.json"
            if result_path.exists() or attempt_path.exists():
                continue
            total += {
                "fixed-rag": 1,
                "fixed-rag-followup-once": 2,
                "decomposition-fixed-rag": 1,
                "decomposition-agent": 2,
            }[arm]
    return total


def _structured_usage(records: list[dict]) -> dict:
    """Account writer/checker calls once; navigation has its own shared ledger."""
    inputs = outputs = 0
    unknown = False
    elapsed = []
    for record in records:
        accounting = record.get("stats") or record.get("accounting") or {}
        input_tokens = accounting.get("prompt_eval_count")
        output_tokens = accounting.get("eval_count")
        if type(input_tokens) is int:
            inputs += input_tokens
        if type(output_tokens) is int:
            outputs += output_tokens
        unknown |= type(input_tokens) is not int or type(output_tokens) is not int
        if type(record.get("elapsed_ms")) in (int, float):
            elapsed.append(record["elapsed_ms"])
    return {
        "input_tokens_lower_bound": inputs,
        "output_tokens_lower_bound": outputs,
        "usage_unknown": unknown,
        "elapsed_ms": elapsed,
    }


def _review_agent_trace(trace: dict | None) -> dict | None:
    if not isinstance(trace, dict):
        return None
    if isinstance(trace.get("agent"), dict):
        return trace["agent"]
    failed = trace.get("failed_stage_trace")
    if isinstance(failed, dict):
        if "model_call_records" in failed:
            return failed
        return _review_agent_trace(failed)
    return None


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
    prompt_revision = static["answer"]["prompt_revisions_by_arm"][arm]
    preflight_path = run_root / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError("run exact-token preflight before model generation")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("static_configuration_sha256") != canonical_hash(static):
        raise ValueError("preflight does not match the frozen run configuration")
    store, retrieval = _load_prepared(run_root, static)
    planned_rows = None
    plan_rows = None
    planned_reranks = None
    if static.get("reading_plan") is not None:
        planned_rows = _load_planned_retrieval(run_root, static, inputs)
        plan_rows = _load_reading_plans(run_root, static, inputs)
        if static["retrieval"].get("shared_initial_reranker"):
            planned_reranks = _load_planned_initial_reranks(run_root, static, inputs)
    selected = [case for case in inputs if case_id is None or case["case_id"] == case_id]
    if not selected or (case_id is not None and selected[0]["case_id"] != case_id):
        raise ValueError("requested case ID is not in the frozen input manifest")

    from page_cite_rag.agent import (
        AgentRunFailure,
        ReadingReport,
        _admit,
        local_chat_model,
    )
    from page_cite_rag.application import FrozenSearchRetriever, ask_with_agent, document_workflow
    from page_cite_rag.review import ReviewRunFailure, answer_with_review
    from page_cite_rag.workflows import RetrievalTrace

    arm_dir = run_root / arm
    arm_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt_dir = arm_dir / ".attempts"
    attempt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = time.monotonic()
    attempted = 0
    completed = 0
    stopped = None
    global_call_ledger, global_call_limit = _global_generation_budget(run_root, static)
    global_call_phase = _global_generation_phase(run_root, "run")
    with LlamaModelClient(
        _llama_config(static),
        budget_policy="exact",
        recover_output_errors=False,
        sampling_profile=static["answer"]["sampling_profile"],
        sampling_seed=42,
    ) as model:
        if model.identity != preflight["generator_identity"]:
            raise ValueError("generator identity differs from the frozen preflight")
        counted_model = _CountingModelClient(
            model,
            global_call_ledger=global_call_ledger,
            global_call_limit=global_call_limit,
            global_call_phase=global_call_phase,
        )
        for case in selected:
            target = arm_dir / f"{case['case_id']}.json"
            if target.exists():
                continue
            preflight_case = _preflight_for_arm(preflight, case["case_id"], arm, prompt_revision)
            if static.get("reading_plan") is not None and preflight_case.get("status") != "ready":
                atomic_json_once(
                    target,
                    {
                        "arm": arm,
                        "case_id": case["case_id"],
                        "dataset": case["dataset"],
                        "execution": "failed",
                        "failure": {
                            "error_type": preflight_case.get("status", "preflight_not_ready")
                        },
                        "result": None,
                        "completion_calls_attempted": 0,
                        "input_tokens_exact_or_observed": 0,
                        "output_tokens_lower_bound": 0,
                        "output_usage_unknown": False,
                        "final_answer_attempted": False,
                        "checker_is_quality_evaluator": False,
                    },
                )
                completed += 1
                continue
            attempt_marker = attempt_dir / f"{case['case_id']}.json"
            if attempt_marker.exists():
                continue
            if time.monotonic() - started >= MAX_INVOCATION_WALL_SECONDS:
                stopped = "invocation_wall_limit"
                break
            case_started = time.monotonic()

            def remaining_quality_writer_calls() -> int:
                if (
                    global_call_phase not in {"development", "answer", "regression_answers"}
                    or tuple(static.get("run_selection", {}).get("arms", ()))
                    not in {QUALITY_V1_ARMS, ("single-agent",)}
                    or (batch_root := _generation_batch_root(run_root)) is None
                    or batch_root.name
                    not in {"agent-production-alignment-v4", "agent-control-followup-v1"}
                ):
                    return 0
                current_case_remaining = int(
                    arm == "fixed-rag-followup-once"
                    and counted_model.generate_calls == counted_model._case_start_calls
                )
                return _remaining_quality_writer_reservation(
                    run_root,
                    static,
                    current_arm=arm,
                    current_case_id=case["case_id"],
                    current_case_remaining_calls=current_case_remaining,
                )

            counted_model.begin_case(
                structured_call_limit=(
                    1
                    if arm in ("fixed-rag", "decomposition-fixed-rag")
                    else 2
                    if arm == "fixed-rag-followup-once"
                    else 4
                    if arm in REVIEW_ARMS
                    else None
                ),
                required_remaining_calls=remaining_quality_writer_calls,
                call_context={
                    "phase": "answer",
                    "arm": arm,
                    "case_id": case["case_id"],
                },
            )
            base_row = retrieval[case["case_id"]]
            plan = None
            reading_plan_status = "not_used"
            if static.get("reading_plan") is not None and arm == "fixed-rag":
                retrieved_row = planned_rows[case["case_id"]]
                row = {
                    **base_row,
                    **retrieved_row,
                    "result": retrieved_row["original_query_result"],
                    "candidate_result": retrieved_row["original_query_candidate_result"],
                    "retrieved_passage_ids": retrieved_row.get(
                        "original_query_retrieved_passage_ids", []
                    ),
                    "candidate_passage_ids": retrieved_row.get(
                        "original_query_candidate_passage_ids", []
                    ),
                }
            elif static.get("reading_plan") is not None and arm != "fixed-rag":
                plan_row = plan_rows[case["case_id"]]
                retrieved_row = planned_rows[case["case_id"]]
                if plan_row["status"] not in ("complete", "degraded") or retrieved_row[
                    "status"
                ] not in ("complete", "degraded"):
                    raise ValueError("preflight admitted a failed reading plan or retrieval")
                reading_plan_status = plan_row["status"]
                if plan_row["status"] == "complete":
                    plan = ReadingPlan.model_validate_json(
                        json.dumps(plan_row["plan"], ensure_ascii=False), strict=True
                    )
                row = {
                    **base_row,
                    **retrieved_row,
                    "retrieved_passage_ids": retrieved_row.get("retrieved_passage_ids", []),
                    "candidate_passage_ids": retrieved_row.get("candidate_passage_ids", []),
                }
            else:
                row = _row_for_arm(
                    base_row,
                    arm,
                    shared_reranker=static.get("retrieval", {}).get("reranker", False),
                )
            if planned_reranks is not None:
                pool_name = "original_query" if arm == "fixed-rag" else "decomposed"
                ranked = planned_reranks[f"{case['case_id']}::{pool_name}"]
                row = {
                    **row,
                    "result": ranked["result"],
                    "retrieved_passage_ids": list(
                        dict.fromkeys(
                            row.get("block_to_passage_id", {})[hit["block"]["block_id"]]
                            for hit in ranked["result"].get("hits", [])
                            if hit["block"]["block_id"] in row.get("block_to_passage_id", {})
                        )
                    ),
                    "initial_reranker_pool": pool_name,
                    "initial_reranker_candidate_sha256": ranked["candidate_sha256"],
                }
            search = _search_result_from_json(row["result"])
            workflow_top_k = (
                _context_group_limit(static) if static.get("reading_plan") is not None else TOP_K
            )
            retriever = FrozenSearchRetriever(search, top_k=workflow_top_k)
            if _agent_uses_live_retrieval(arm) or arm in REVIEW_ARMS:
                reranker_model_path = None
                if static.get("retrieval", {}).get("reranker", False):
                    from page_cite_rag.reranker_server import DEFAULT_MODEL_PATH

                    reranker_model_path = DEFAULT_MODEL_PATH
                retriever = StagedGpuSearch(
                    store,
                    row["document_id"],
                    "hybrid",
                    DEFAULT_GGUF_PATH,
                    GPU_UUID,
                    tokenizer_directory=DEFAULT_MODEL_DIR,
                    gpu_temperature_limit_c=GPU_TEMPERATURE_LIMIT_C,
                    reranker_model_path=reranker_model_path,
                    candidates=(
                        static["reading_plan"]["retrieval_candidates_per_branch"]
                        if static.get("reading_plan") is not None and arm != "fixed-rag"
                        else None
                    ),
                )
            workflow = document_workflow(
                store,
                row["document_id"],
                retriever,
                counted_model,
                model.config,
                top_k=workflow_top_k,
                answer_format=_answer_format_for(case),
                prompt_revision=prompt_revision,
            )
            if plan is not None:
                workflow = workflow.with_reading_plan(plan)
            blocks = workflow._context_blocks(search, workflow._read_hits(search, case["question"]))
            exact_preflight = preflight_case
            common_ids = exact_preflight["admitted_block_ids"]
            common_set = set(common_ids)
            common_blocks = tuple(block for block in blocks if block.block_id in common_set)
            common_groups = tuple(
                group
                for group in search.context_groups
                if set(group.source_block_ids).issubset(common_set)
            )
            if [block.block_id for block in common_blocks] != common_ids:
                raise ValueError("runtime context differs from frozen common evidence")
            try:
                context = _admit(
                    workflow,
                    case["question"],
                    case["language"],
                    common_blocks,
                    common_groups,
                )
            except ModelError as exc:
                stopped = "context_preparation_model_error"
                atomic_json_once(
                    run_root / f"preparation-stop-{arm}-{time.time_ns()}.json",
                    {
                        "state": stopped,
                        "arm": arm,
                        "case_id": case["case_id"],
                        "case_generation_attempted": False,
                        "error_type": type(exc).__name__,
                        "gpu_diagnostics": _safe_model_diagnostics(model),
                    },
                )
                break
            if (
                context.exact_input_tokens != exact_preflight["exact_final_input_tokens"]
                or [block.block_id for block in context.blocks] != common_ids
                or context.omitted_block_ids
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
            navigation_answer = None
            reading_trace = None
            followup_trace = None
            first_answer_result = None
            final_attempted = False
            result = None
            failure = None
            final_calls_before = counted_model.generate_calls
            agent_dispatches: dict[int, int | None] = {}

            def record_agent_model_call(record: dict) -> bool | None:
                event = dict(record)
                local_index = event.get("call_index")
                status = event.get("status")
                is_writer = event.get("role") == "writer"
                # The common writer is dispatched through counted_model.generate,
                # which owns the authoritative global reservation and usage row.
                # Its application callback is trace-only; reserving here would
                # charge the same inference twice.
                if status == "ready_to_dispatch" and is_writer:
                    if type(local_index) is not int or local_index in agent_dispatches:
                        raise ModelError("Agent writer emitted an invalid dispatch record")
                    batch_root = _generation_batch_root(run_root)
                    reserve = (
                        _remaining_quality_writer_reservation(
                            run_root,
                            static,
                            current_arm=arm,
                            current_case_id=case["case_id"],
                            current_case_remaining_calls=0,
                        )
                        if batch_root is not None
                        and batch_root.name
                        in {"agent-production-alignment-v4", "agent-control-followup-v1"}
                        and global_call_phase in {"development", "answer", "regression_answers"}
                        else 0
                    )
                    allowed, global_index = counted_model.reserve_before_dispatch(
                        workflow.agent_response_model(),
                        required_remaining_calls=reserve,
                        dynamic_schema=True,
                    )
                    if not allowed:
                        return False
                    agent_dispatches[local_index] = global_index
                    if global_index is not None:
                        event["global_call_index"] = global_index
                    append_jsonl_record(
                        arm_dir / ".model-calls.jsonl",
                        {"case_id": case["case_id"], "arm": arm, **event},
                    )
                    return True
                if status == "ready_to_dispatch" and not is_writer:
                    if type(local_index) is not int or local_index in agent_dispatches:
                        raise ModelError("Agent emitted an invalid or duplicate dispatch record")
                    batch_root = _generation_batch_root(run_root)
                    reserve = (
                        _remaining_quality_writer_reservation(
                            run_root,
                            static,
                            current_arm=arm,
                            current_case_id=case["case_id"],
                            current_case_remaining_calls=1,
                        )
                        if batch_root is not None
                        and batch_root.name
                        in {"agent-production-alignment-v4", "agent-control-followup-v1"}
                        and global_call_phase in {"development", "answer", "regression_answers"}
                        else 0
                    )
                    try:
                        global_index = counted_model._reserve_global_call(
                            ReadingReport, required_remaining_calls=reserve
                        )
                    except ModelError as exc:
                        if "ceiling reached" in str(exc):
                            return False
                        raise
                    agent_dispatches[local_index] = global_index
                    if global_index is not None:
                        event["global_call_index"] = global_index
                    append_jsonl_record(
                        arm_dir / ".model-calls.jsonl",
                        {"case_id": case["case_id"], "arm": arm, **event},
                    )
                    return True
                if status == "dispatched" and not is_writer:
                    if type(local_index) is not int or local_index not in agent_dispatches:
                        raise ModelError("Agent dispatched without a global reservation")
                    global_index = agent_dispatches[local_index]
                    if global_index is not None:
                        event["global_call_index"] = global_index
                elif status == "dispatched" and is_writer:
                    if type(local_index) is not int or local_index not in agent_dispatches:
                        raise ModelError("Agent writer dispatched without a global reservation")
                    global_index = agent_dispatches[local_index]
                    if global_index is not None:
                        event["global_call_index"] = global_index
                elif status in {"completed", "failed_usage_unknown"} and not is_writer:
                    if type(local_index) is not int or local_index not in agent_dispatches:
                        raise ModelError("Agent completion has no reserved global dispatch")
                    global_index = agent_dispatches.pop(local_index)
                    if global_index is not None:
                        counted_model._record_global_usage(
                            global_index,
                            {
                                "stats": {
                                    "prompt_eval_count": event.get("input_tokens_exact"),
                                    "eval_count": event.get("output_tokens_observed"),
                                }
                            },
                        )
                        event["global_call_index"] = global_index
                append_jsonl_record(
                    arm_dir / ".model-calls.jsonl",
                    {"case_id": case["case_id"], "arm": arm, **event},
                )
                return None

            def capture_agent_wire_request(record: dict) -> None:
                append_jsonl_record(
                    arm_dir / "wire-requests.jsonl",
                    {"case_id": case["case_id"], "arm": arm, **record},
                )

            attempted += 1
            shared_retrieval_seconds = max(0.0, search.elapsed_ms / 1000)
            remaining_case_seconds = (
                MAX_CASE_WALL_SECONDS - shared_retrieval_seconds - (time.monotonic() - case_started)
            )
            with _owned_case_deadline(model, retriever, remaining_case_seconds) as deadline_expired:
                try:
                    if deadline_expired.is_set():
                        raise TimeoutError("case deadline expired before generation")
                    if arm in REVIEW_ARMS:
                        review_search = search.model_copy(
                            update={"context_groups": context.context_groups}
                        )
                        with retriever.session():
                            if _is_agent_arm(arm):
                                with local_chat_model(counted_model) as chat_model:
                                    reviewed = answer_with_review(
                                        workflow,
                                        case["question"],
                                        language=case["language"],
                                        strategy=REVIEW_ARMS[arm],
                                        initial_search=review_search,
                                        initial_blocks=context.blocks,
                                        chat_model=chat_model,
                                    )
                            else:
                                reviewed = answer_with_review(
                                    workflow,
                                    case["question"],
                                    language=case["language"],
                                    strategy=REVIEW_ARMS[arm],
                                    initial_search=review_search,
                                    initial_blocks=context.blocks,
                                )
                        result = reviewed.result
                        reading_trace = reviewed.trace
                    else:
                        if not _is_agent_arm(arm):
                            result = workflow.answer_from_context(
                                case["question"],
                                context.blocks,
                                language=case["language"],
                                context_groups=context.context_groups,
                            )
                            if arm == "fixed-rag-followup-once":
                                first_answer_result = result
                                followup_trace = {
                                    "protocol": "fixed-rag-followup-once/v1",
                                    "triggered": result.status
                                    in ("partial", "insufficient_evidence"),
                                    "initial_status": result.status,
                                    "candidate_selection": (
                                        "next_unseen_complete_source_blocks_from_frozen_hybrid_pool"
                                    ),
                                    "candidate_limit": FIXED_FOLLOWUP_BLOCKS,
                                    "initial_hit_block_ids": [
                                        hit.block.block_id for hit in search.hits
                                    ],
                                    "initial_admitted_block_ids": [
                                        block.block_id for block in context.blocks
                                    ],
                                    "candidate_block_ids": [],
                                    "admitted_block_ids": [],
                                    "state": "not_triggered",
                                }
                                if followup_trace["triggered"]:
                                    candidate_value = row.get("candidate_result")
                                    if not isinstance(candidate_value, dict):
                                        raise ValueError(
                                            "fixed follow-up needs its frozen hybrid pool"
                                        )
                                    followup_search = _fixed_followup_search(
                                        search,
                                        _search_result_from_json(candidate_value),
                                        initial_admitted_block_ids=[
                                            block.block_id for block in context.blocks
                                        ],
                                        limit=FIXED_FOLLOWUP_BLOCKS,
                                    )
                                    if followup_search is None:
                                        followup_trace["state"] = "no_unseen_candidates"
                                    else:
                                        followup_blocks = workflow._context_blocks(
                                            followup_search,
                                            workflow._read_hits(followup_search, case["question"]),
                                        )
                                        followup_trace["candidate_block_ids"] = [
                                            block.block_id for block in followup_blocks
                                        ]
                                        expanded, admission_trace = _fixed_followup_admission(
                                            workflow,
                                            case["question"],
                                            case["language"],
                                            context,
                                            followup_search,
                                            tuple(followup_blocks),
                                        )
                                        followup_trace["context_admission"] = admission_trace
                                        followup_trace["admitted_block_ids"] = [
                                            block_id
                                            for block_id in admission_trace.get(
                                                "new_block_ids_admitted", []
                                            )
                                        ]
                                        followup_trace["omitted_block_ids"] = list(
                                            expanded.omitted_block_ids
                                            if expanded is not None
                                            else []
                                        )
                                        followup_trace["exact_input_tokens"] = (
                                            expanded.exact_input_tokens
                                            if expanded is not None
                                            else None
                                        )
                                        if expanded is None:
                                            followup_trace["state"] = admission_trace["state"]
                                        else:
                                            followup_trace["state"] = "started"
                                            try:
                                                result = workflow.answer_from_context(
                                                    case["question"],
                                                    expanded.blocks,
                                                    language=case["language"],
                                                    context_groups=expanded.context_groups,
                                                )
                                                followup_trace["state"] = "completed"
                                                followup_trace["final_status"] = result.status
                                            except Exception as followup_error:
                                                result = first_answer_result
                                                followup_trace["state"] = (
                                                    "failed_fallback_to_initial_answer"
                                                )
                                                followup_trace["error_type"] = type(
                                                    followup_error
                                                ).__name__
                        else:
                            retrieval_session = (
                                retriever.session()
                                if _agent_uses_live_retrieval(arm)
                                and getattr(retriever, "reranker_model_path", None) is None
                                else nullcontext()
                            )
                            with retrieval_session:
                                navigation_answer = ask_with_agent(
                                    workflow,
                                    case["question"],
                                    language=case["language"],
                                    initial_search=search,
                                    initial_blocks=blocks,
                                    delegated=arm == "delegated-agent",
                                    data_tools_enabled=arm != "single-agent-no-tools",
                                    max_model_calls=1 if arm == "single-agent-no-tools" else None,
                                    on_model_call=record_agent_model_call,
                                    on_wire_request=(
                                        capture_agent_wire_request
                                        if global_call_phase == "protocol"
                                        else None
                                    ),
                                )
                            result = navigation_answer.result
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
                    if isinstance(exc, ReviewRunFailure):
                        reading_trace = exc.trace
                    if arm == "fixed-rag-followup-once" and first_answer_result is not None:
                        result = first_answer_result
                        followup_trace = followup_trace or {
                            "protocol": "fixed-rag-followup-once/v1",
                            "triggered": True,
                        }
                        followup_trace.update(
                            {
                                "state": "failed_fallback_to_initial_answer",
                                "error_type": type(exc).__name__,
                            }
                        )
                        completed += 1
                    else:
                        failure = {
                            "error_type": exc.error_type
                            if isinstance(exc, (AgentRunFailure, ReviewRunFailure))
                            else type(exc).__name__,
                            "agent_trace": exc.trace
                            if isinstance(exc, AgentRunFailure)
                            else _review_agent_trace(reading_trace),
                            "reading_trace": reading_trace,
                            "final_answer_attempted": final_attempted,
                        }
                        if reason := getattr(exc, "reason_code", None):
                            failure["reason_code"] = reason
                        if accounting := getattr(exc, "accounting", None):
                            failure["accounting"] = accounting.model_dump(mode="json")
                        if output := getattr(exc, "raw_model_output", None):
                            failure["raw_model_output"] = output
                        if errors := getattr(exc, "validation_errors", ()):
                            failure["validation_errors"] = list(errors)

                        returned = getattr(counted_model, "last_result", None)
                        if final_attempted and returned is not None:
                            if (
                                "raw_model_output" not in failure
                                and returned.raw_model_output is not None
                            ):
                                failure["raw_model_output"] = returned.raw_model_output
                            if (
                                "accounting" not in failure
                                and returned.stats.prompt_eval_count
                                == exact_preflight["exact_final_input_tokens"]
                                and returned.stats.eval_count <= model.config.num_predict
                            ):
                                failure["accounting"] = {
                                    "prompt_eval_count": returned.stats.prompt_eval_count,
                                    "eval_count": returned.stats.eval_count,
                                    "elapsed_ms": returned.stats.elapsed_ms,
                                    "stop_type": "eos",
                                }

            if deadline_expired.is_set():
                keep_initial_followup_answer = (
                    arm == "fixed-rag-followup-once"
                    and first_answer_result is not None
                    and result is not None
                    and followup_trace is not None
                )
                if keep_initial_followup_answer:
                    if followup_trace.get("state") == "started":
                        result = first_answer_result
                        followup_trace["state"] = "deadline_fallback_to_initial_answer"
                    followup_trace["deadline_exceeded"] = True
                elif result is not None:
                    completed -= 1
                    result = None
                    failure = {
                        **(failure or {}),
                        "underlying_error_type": (failure or {}).get("error_type"),
                        "error_type": "CaseDeadlineExceeded",
                        "deadline_seconds": MAX_CASE_WALL_SECONDS,
                    }
                else:
                    failure = {
                        **(failure or {}),
                        "underlying_error_type": (failure or {}).get("error_type"),
                        "error_type": "CaseDeadlineExceeded",
                        "deadline_seconds": MAX_CASE_WALL_SECONDS,
                    }

            diagnostics = _safe_model_diagnostics(model)
            generation_stats = (
                result.generation_stats.model_dump(mode="json")
                if result is not None and result.generation_stats is not None
                else None
            )
            agent_trace = (
                navigation_answer.trace["agent"]
                if navigation_answer is not None
                else _review_agent_trace(reading_trace)
                or (failure.get("agent_trace") if failure else None)
            )
            agent_input = agent_trace.get("agent_input_tokens_exact") if agent_trace else 0
            agent_call_records = agent_trace.get("model_call_records", []) if agent_trace else []
            dispatched_agent_statuses = {
                "dispatched",
                "completed",
                "failed_usage_unknown",
                "failed_accounted",
            }
            agent_final_call = (
                agent_trace.get("final_model_call") if isinstance(agent_trace, dict) else None
            )
            agent_output_lower_bound = sum(
                call.get("output_tokens_observed", 0)
                for call in agent_call_records
                if type(call.get("output_tokens_observed")) is int
            )
            agent_output_unknown = any(
                call.get("status") in dispatched_agent_statuses
                and type(call.get("output_tokens_observed")) is not int
                for call in agent_call_records
            )
            failure_accounting = failure.get("accounting") if failure else None
            final_input = (
                generation_stats.get("prompt_eval_count")
                if generation_stats is not None
                else failure_accounting.get("prompt_eval_count")
                if isinstance(failure_accounting, dict)
                else agent_final_call.get("input_tokens_exact")
                if isinstance(agent_final_call, dict)
                else None
            )
            final_output = (
                generation_stats.get("eval_count")
                if generation_stats is not None
                else failure_accounting.get("eval_count")
                if isinstance(failure_accounting, dict)
                else agent_final_call.get("output_tokens_observed")
                if isinstance(agent_final_call, dict)
                else None
            )
            agent_calls = sum(
                call.get("status") in dispatched_agent_statuses for call in agent_call_records
            )
            navigation_input = sum(
                call.get("input_tokens_exact", 0)
                for call in agent_call_records
                if call.get("status") in dispatched_agent_statuses
                and type(call.get("input_tokens_exact")) is int
            )
            structured_usage = _structured_usage(counted_model.records)
            input_lower_bound = navigation_input + structured_usage["input_tokens_lower_bound"]
            output_lower_bound = (
                agent_output_lower_bound + structured_usage["output_tokens_lower_bound"]
            )
            final_usage_unknown = structured_usage["usage_unknown"]
            answer_generations = [
                value for value in counted_model.records if value["role"] == "answer"
            ]
            record = {
                "arm": arm,
                "case_id": case["case_id"],
                "dataset": case["dataset"],
                "answer_format": _answer_format_for(case),
                "reading_plan_status": reading_plan_status,
                "reading_plan_degraded_reason": (
                    plan_rows[case["case_id"]].get("degraded_reason")
                    if reading_plan_status == "degraded"
                    else None
                ),
                "prompt_revision": prompt_revision,
                "prompt_version": result.prompt_version if result is not None else None,
                "execution": "ok" if result is not None else "failed",
                "failure": failure,
                "result": result.model_dump(mode="json") if result is not None else None,
                "raw_model_output": (
                    answer_generations[-1].get("raw_model_output") if answer_generations else None
                ),
                "structured_generation_records": counted_model.records,
                "structured_generation_usage": structured_usage,
                "reading_trace": reading_trace,
                "followup_trace": followup_trace,
                "checker_is_quality_evaluator": False,
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
                "final_answer_attempted": final_attempted
                or (
                    isinstance(agent_final_call, dict)
                    and agent_final_call.get("status") in dispatched_agent_statuses
                ),
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
                "block_to_article_id": row.get("block_to_article_id", {}),
                "candidate_passage_ids": row.get("candidate_passage_ids", []),
                "candidate_article_ids": row.get("candidate_article_ids", []),
                "initial_top10_passage_ids": row.get("retrieved_passage_ids", []),
                "initial_top10_article_ids": row.get("retrieved_article_ids", []),
                "initial_admitted_block_ids": exact_preflight["admitted_block_ids"],
                "initial_omitted_block_ids": exact_preflight["omitted_block_ids"],
                "retrieval_elapsed_ms": search.elapsed_ms,
                "context_trace": {
                    "retrieved_block_ids": exact_preflight["retrieved_block_ids"],
                    "initial_admitted_block_ids": exact_preflight["admitted_block_ids"],
                    "initial_omitted_block_ids": exact_preflight["omitted_block_ids"],
                    "prompt_budget_admitted_block_ids": _prompt_budget_admitted_block_ids(
                        exact_preflight
                    ),
                    "final_included_block_ids": (
                        list(result.included_block_ids)
                        if result is not None
                        else agent_trace.get("selected_block_ids", [])
                        if isinstance(agent_trace, dict)
                        else []
                    ),
                    "final_omitted_block_ids": (
                        [
                            block_id
                            for block_id in exact_preflight["retrieved_block_ids"]
                            if block_id
                            not in set(
                                result.included_block_ids
                                if result is not None
                                else agent_trace.get("selected_block_ids", [])
                                if isinstance(agent_trace, dict)
                                else []
                            )
                        ]
                        + [
                            block_id
                            for block_id in (
                                result.omitted_block_ids
                                if result is not None
                                else agent_trace.get("omitted_block_ids", [])
                                if isinstance(agent_trace, dict)
                                else []
                            )
                            if block_id not in exact_preflight["retrieved_block_ids"]
                        ]
                    ),
                    "final_input_tokens": (
                        result.budget.exact_input_tokens
                        if result is not None and result.budget is not None
                        else None
                    ),
                },
                "elapsed_ms": (time.monotonic() - case_started) * 1000,
                "case_deadline_seconds": MAX_CASE_WALL_SECONDS,
                "deadline_shared_retrieval_ms": shared_retrieval_seconds * 1000,
                "deadline_accounted_elapsed_ms": (
                    (time.monotonic() - case_started + shared_retrieval_seconds) * 1000
                ),
                "model_call_elapsed_ms": [
                    call["elapsed_ms"]
                    for call in agent_call_records
                    if type(call.get("elapsed_ms")) in (int, float)
                ]
                + structured_usage["elapsed_ms"],
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
            retrieval_trace = (
                record["gpu_retrieval_trace"]
                if isinstance(record["gpu_retrieval_trace"], dict)
                else {}
            )
            if deadline_expired.is_set():
                stopped = "case_deadline_exceeded"
                break
            if _gpu_guard_detected(diagnostics, retrieval_trace):
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


_ZH_T2S_F1_AUXILIARY = "longbench_zh_t2s_f1_nonofficial_v1"


def _frozen_auxiliary_metrics(manifest: dict) -> list[str]:
    metrics = manifest.get("auxiliary_metrics", [])
    if not isinstance(metrics, list) or metrics not in ([], [_ZH_T2S_F1_AUXILIARY]):
        raise ValueError("unsupported or invalid frozen auxiliary metrics")
    return metrics


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


def _longbench_zh_t2s_f1_nonofficial(prediction: str, references: list[str]) -> float:
    """Apply OpenCC t2s to both sides before the unchanged LongBench token F1.

    This is an auxiliary lexical score, not the official benchmark metric or a
    semantic-correctness judgment. Neither the answer nor the references are edited.
    """
    import opencc

    converter = opencc.OpenCC("t2s.json")
    return _longbench_zh_f1(
        converter.convert(prediction), [converter.convert(reference) for reference in references]
    )


def _xrag_zh_char_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Return explicitly non-official character overlap after script normalization."""
    import unicodedata

    import opencc

    converter = opencc.OpenCC("t2s.json")

    def normalize(value: str) -> str:
        return "".join(
            character
            for character in unicodedata.normalize("NFKC", converter.convert(value)).casefold()
            if not character.isspace() and not unicodedata.category(character).startswith("P")
        )

    predicted = normalize(prediction)
    expected = normalize(reference)
    predicted_chars = Counter(predicted)
    expected_chars = Counter(expected)
    overlap = sum((predicted_chars & expected_chars).values())
    f1 = 2 * overlap / (len(predicted) + len(expected)) if overlap else 0.0
    return {
        "answer_char_f1_nonofficial": f1,
        "answer_exact_match_nonofficial": float(bool(expected) and predicted == expected),
    }


def _xrag_cited_article_ids(result: dict | None, block_to_article: dict[str, str]) -> list[str]:
    if result is None:
        return []
    article_ids = []
    for field in ("answer_claims", "detail_claims"):
        for claim in result.get(field, []):
            for citation in claim.get("citations", []):
                article_id = block_to_article.get(citation.get("block_id"))
                if article_id is not None and article_id not in article_ids:
                    article_ids.append(article_id)
    return article_ids


def _xrag_article_metrics(
    gold_row: dict,
    candidate_articles: set[str],
    top10_articles: set[str],
    initial_context_articles: set[str],
    final_context_articles: set[str],
    cited_articles: set[str],
) -> dict[str, float | None]:
    """Score XRAG article labels without treating two partial sources as one sufficient source."""
    support_levels = gold_row["article_support_levels"]
    sufficient = {article for article, level in support_levels.items() if level == 2}
    supporting = {article for article, level in support_levels.items() if level > 0}
    partial = {article for article, level in support_levels.items() if level == 1}
    crossdoc_type = gold_row["crossdoc_type"]
    if type(crossdoc_type) is not int or crossdoc_type not in (0, 1, 2):
        raise ValueError("XRAG case has an invalid cross-document type")
    if crossdoc_type == 2:
        if sufficient or len(partial) != 2:
            raise ValueError("XRAG crossdoc_type=2 must have two partial support articles")
        required = partial
    else:
        if not sufficient:
            raise ValueError("XRAG case has no individually sufficient support article")
        required = sufficient

    def recall(found: set[str], targets: set[str]) -> float:
        return len(found & targets) / len(targets)

    sufficient_metrics = {
        "sufficient_article_recall_candidate_union": recall(candidate_articles, sufficient)
        if sufficient
        else None,
        "sufficient_article_recall_at_10": recall(top10_articles, sufficient)
        if sufficient
        else None,
        "sufficient_article_hit_at_10": float(bool(top10_articles & sufficient))
        if sufficient
        else None,
        "sufficient_article_recall_in_initial_context": recall(initial_context_articles, sufficient)
        if sufficient
        else None,
        "sufficient_article_recall_in_final_context": recall(final_context_articles, sufficient)
        if sufficient
        else None,
        "citation_sufficient_article_precision": len(cited_articles & sufficient)
        / len(cited_articles)
        if cited_articles and sufficient
        else None
        if cited_articles
        else None,
    }
    return {
        **sufficient_metrics,
        "supporting_article_recall_at_10": recall(top10_articles, supporting),
        "required_article_recall_candidate_union": recall(candidate_articles, required)
        if crossdoc_type == 2
        else float(bool(candidate_articles & required)),
        "required_article_recall_at_10": recall(top10_articles, required)
        if crossdoc_type == 2
        else float(bool(top10_articles & required)),
        "required_article_hit_at_10": float(required <= top10_articles)
        if crossdoc_type == 2
        else float(bool(top10_articles & required)),
        "required_article_recall_in_initial_context": recall(initial_context_articles, required)
        if crossdoc_type == 2
        else float(bool(initial_context_articles & required)),
        "required_article_recall_in_final_context": recall(final_context_articles, required)
        if crossdoc_type == 2
        else float(bool(final_context_articles & required)),
        "citation_required_article_recall": recall(cited_articles, required)
        if crossdoc_type == 2
        else float(bool(cited_articles & required)),
        "citation_supporting_article_precision": len(cited_articles & supporting)
        / len(cited_articles)
        if cited_articles
        else None,
    }


def _longbench_v2_choice(prediction: str) -> str | None:
    response = prediction.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def _summary_scoring_text(result: dict | None) -> str:
    """Score the complete user-visible response, not its citation metadata or caveats."""
    if result is None:
        return ""
    text = [result.get("answer", "")]
    text.extend(claim["text"] for claim in result.get("detail_claims", []))
    return "\n".join(value for value in text if value)


def _longbench_summary_score(prediction: str, references: list[str], language: str) -> dict:
    """LongBench's Rouge() L-F1; Chinese uses its original jieba tokenization."""
    try:
        from rouge import Rouge
    except ImportError:
        return {"rouge_l": None, "scoring_unavailable": "official rouge dependency not installed"}
    if (
        not isinstance(references, list)
        or not references
        or any(not isinstance(value, str) or not value for value in references)
    ):
        raise ValueError("summary gold must contain nonempty official reference strings")
    if language == "zh":
        import jieba

        prediction = " ".join(jieba.cut(prediction, cut_all=False))
        references = [" ".join(jieba.cut(value, cut_all=False)) for value in references]
    elif language != "en":
        raise ValueError("unsupported official summary language")
    scores = []
    for reference in references:
        if not prediction.strip():
            scores.append(0.0)
            continue
        try:
            value = Rouge().get_scores([prediction], [reference], avg=True)["rouge-l"]["f"]
        except (ValueError, RecursionError):
            # The pinned upstream scorer assigns zero for invalid/too-long Rouge inputs.
            value = 0.0
        scores.append(value)
    return {
        "rouge_l": max(scores),
        "scoring_unavailable": None,
        "official_metric": "LongBench ROUGE-L F1 / rouge 1.0.1; jieba for VCSUM",
    }


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


def _score_dataset_group(case: dict, input_row: dict) -> str:
    """Keep translated-query diagnostics out of native benchmark aggregates."""
    dataset = case["dataset"]
    if dataset not in ("qasper", "longbench-v1-multifieldqa-zh"):
        return dataset
    question_language = input_row.get("question_language")
    source_language = input_row.get("source_language")
    if (
        isinstance(question_language, str)
        and isinstance(source_language, str)
        and question_language != source_language
    ):
        return f"derived-query-{dataset}-source-{source_language}"
    return dataset


def score_run(
    run_root: Path,
    manifest: dict,
    inputs: list[dict],
    gold: list[dict],
    arms: tuple[str, ...] = ARMS,
) -> dict:
    if (run_root / "scores.json").exists():
        raise ValueError("scores.json already exists; never overwrite a frozen score")
    auxiliary_metrics = _frozen_auxiliary_metrics(manifest)
    if auxiliary_metrics:
        configuration_path = run_root / "configuration.json"
        if not configuration_path.is_file() or configuration_path.is_symlink():
            raise ValueError("auxiliary metrics require a frozen run configuration")
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        if configuration.get("static", {}).get("auxiliary_metrics") != auxiliary_metrics:
            raise ValueError("auxiliary metrics differ from the frozen run configuration")
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
            for arm in arms
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
        is_official_longbench = arm == LONG_BENCH_OFFICIAL_ARM
        datasets: dict[str, list[dict]] = {}
        attempted_generation_calls = total_input = total_output = 0
        unknown_output_rows = 0
        case_elapsed_ms = []
        model_call_elapsed_ms = []
        gpu_peak_memory = []
        gpu_peak_temperature = []
        for case in manifest["cases"]:
            case_id = case["case_id"]
            input_dataset = inputs_by_id[case_id].get("dataset")
            if input_dataset is not None:
                if case.get("dataset", input_dataset) != input_dataset:
                    raise ValueError("manifest dataset differs from the frozen input")
                case = {**case, "dataset": input_dataset}
            raw = records[case_id]
            prediction_result = raw.get("result") if raw.get("execution") == "ok" else None
            prediction_text = (
                raw.get("raw_model_output", "")
                if is_official_longbench
                else prediction_result.get("answer", "")
                if prediction_result
                else ""
            )
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
                            tool_read_ids.extend(
                                event.get("returned_block_ids") or [event.get("block_id")]
                            )
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
                    no_answer = gold_row.get("unanswerable") is True
                    question_score["correct_refusal"] = bool(
                        no_answer and prediction_result.get("status") == "insufficient_evidence"
                    )
                    question_score["false_refusal"] = bool(
                        not no_answer and prediction_result.get("status") == "insufficient_evidence"
                    )
            elif case["dataset"] == "longbench-v2-single-document-qa":
                choice = (
                    _longbench_v2_official_choice(prediction_text)
                    if is_official_longbench and isinstance(prediction_text, str)
                    else _longbench_v2_choice(prediction_text)
                    if prediction_result
                    else None
                )
                question_score = {
                    "case_id": case_id,
                    "accuracy": float(choice == gold_row["answer"]),
                    "extracted_choice": choice,
                    "official_protocol_raw_output_scored": is_official_longbench,
                    "execution": raw["execution"],
                }
            elif case["dataset"] == "longbench-v1-multifieldqa-zh":
                question_score = {
                    "case_id": case_id,
                    "answer_f1": _longbench_zh_f1(prediction_text, gold_row["answers"]),
                    "execution": raw["execution"],
                }
            elif case["dataset"] in SUMMARY_DATASETS:
                native_language = SUMMARY_DATASETS[case["dataset"]]
                output_language = inputs_by_id[case_id].get("language", "")
                matches_language = (
                    output_language == "en"
                    if native_language == "en"
                    else output_language.startswith("zh")
                )
                metric = (
                    _longbench_summary_score(
                        _summary_scoring_text(prediction_result),
                        gold_row["answers"],
                        native_language,
                    )
                    if matches_language
                    else {
                        "rouge_l": None,
                        "scoring_unavailable": "output language differs from native reference",
                    }
                )
                question_score = {
                    "case_id": case_id,
                    **metric,
                    "scoring_text_contract": "answer plus detail_claim texts",
                    "execution": raw["execution"],
                }
            elif case["dataset"] == "derived-crosslingual":
                # Gold stays in its original language; cross-language word overlap would
                # be misleading. These ten translated-query regressions use source review.
                question_score = {
                    "case_id": case_id,
                    "execution": raw["execution"],
                    "answer_available": bool(
                        prediction_result
                        and prediction_result.get("status") in ("answered", "partial")
                    ),
                    "refused": bool(
                        prediction_result
                        and prediction_result.get("status") == "insufficient_evidence"
                    ),
                    "official_correctness_metric": None,
                    "source_language": inputs_by_id[case_id]["source_language"],
                    "review_required": True,
                }
            elif case["dataset"] == "product-functional":
                status = prediction_result.get("status") if prediction_result else None
                question_score = {
                    "case_id": case_id,
                    "execution": raw["execution"],
                    "answer_available": status in ("answered", "partial"),
                    "refused": status == "insufficient_evidence",
                    "claim_count": len(prediction_result.get("answer_claims", []))
                    + len(prediction_result.get("detail_claims", []))
                    if prediction_result
                    else 0,
                    "citation_count": sum(
                        len(claim.get("citations", []))
                        for field in ("answer_claims", "detail_claims")
                        for claim in prediction_result.get(field, [])
                    )
                    if prediction_result
                    else 0,
                    "official_correctness_metric": None,
                    "source_language": inputs_by_id[case_id]["source_language"],
                    "question_language": inputs_by_id[case_id]["question_language"],
                    "answer_language": inputs_by_id[case_id]["language"],
                    "review_required": True,
                }
            elif case["dataset"] == "xrag-zh-mono-crosslingual":
                support_levels = gold_row["article_support_levels"]
                block_to_article = raw.get("block_to_article_id", {})
                candidate_articles = set(raw.get("candidate_article_ids", []))
                top10_articles = set(raw.get("initial_top10_article_ids", []))
                admitted_articles = {
                    block_to_article[block_id]
                    for block_id in raw.get("initial_admitted_block_ids", [])
                    if block_id in block_to_article
                }
                final_block_ids = (
                    prediction_result.get("included_block_ids", []) if prediction_result else []
                )
                final_articles = {
                    block_to_article[block_id]
                    for block_id in final_block_ids
                    if block_id in block_to_article
                }
                citation_articles = set(
                    _xrag_cited_article_ids(prediction_result, block_to_article)
                )
                answer_metrics = _xrag_zh_char_metrics(prediction_text, gold_row["answer"])
                question_score = {
                    "case_id": case_id,
                    "crossdoc_type": gold_row["crossdoc_type"],
                    **answer_metrics,
                    **_xrag_article_metrics(
                        gold_row,
                        candidate_articles,
                        top10_articles,
                        admitted_articles,
                        final_articles,
                        citation_articles,
                    ),
                    "cited_article_ids": sorted(citation_articles),
                    "sufficient_article_ids": sorted(
                        article for article, level in support_levels.items() if level == 2
                    ),
                    "required_article_ids": sorted(
                        article
                        for article, level in support_levels.items()
                        if level == (1 if gold_row["crossdoc_type"] == 2 else 2)
                    ),
                    "execution": raw["execution"],
                }
            else:
                raise ValueError("unknown frozen dataset task")
            native_eligible = case.get("native_metric_eligible", True) is not False
            output_language = inputs_by_id[case_id].get("language")
            reference_language = {
                "qasper": "en",
                "longbench-v1-multifieldqa-zh": "zh",
            }.get(case["dataset"])
            if output_language is not None and reference_language is not None:
                native_eligible = native_eligible and (
                    output_language == "en"
                    if reference_language == "en"
                    else output_language.startswith("zh")
                )
            if "answer_f1" in question_score:
                question_score["native_metric_eligible"] = native_eligible
                if not native_eligible:
                    question_score["answer_f1"] = None
                    question_score["scoring_unavailable"] = (
                        "Frozen eligibility or requested answer language excludes native "
                        "lexical F1; source-based semantic and language review required."
                    )
            if (
                case["dataset"] == "longbench-v1-multifieldqa-zh"
                and _ZH_T2S_F1_AUXILIARY in auxiliary_metrics
            ):
                question_score[_ZH_T2S_F1_AUXILIARY] = (
                    _longbench_zh_t2s_f1_nonofficial(prediction_text, gold_row["answers"])
                    if native_eligible
                    else None
                )
            trace = raw.get("context_trace")
            if is_official_longbench and isinstance(trace, dict):
                # This fixed arm submits one context, without a later Agent/writer stage.
                trace = {
                    "initial_retrieved_block_ids": trace["retrieved_block_ids"],
                    "initial_admitted_block_ids": trace["admitted_block_ids"],
                    "initial_omitted_block_ids": trace["omitted_block_ids"],
                    "final_included_block_ids": trace["admitted_block_ids"],
                    "final_omitted_block_ids": trace["omitted_block_ids"],
                    "final_input_tokens": trace["exact_input_tokens"],
                }
            question_score["context_trace"] = trace
            dataset_name = _score_dataset_group(case, inputs_by_id[case_id])
            if dataset_name == "derived-crosslingual":
                dataset_name += "-source-" + inputs_by_id[case_id]["source_language"]
            datasets.setdefault(dataset_name, []).append(question_score)
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
            for key in (
                "answer_f1",
                _ZH_T2S_F1_AUXILIARY,
                "accuracy",
                "rouge_l",
                "answer_char_f1_nonofficial",
                "answer_exact_match_nonofficial",
                "sufficient_article_recall_candidate_union",
                "sufficient_article_recall_at_10",
                "sufficient_article_hit_at_10",
                "supporting_article_recall_at_10",
                "sufficient_article_recall_in_initial_context",
                "sufficient_article_recall_in_final_context",
                "citation_sufficient_article_precision",
                "citation_supporting_article_precision",
                "citation_required_article_recall",
                "required_article_recall_candidate_union",
                "required_article_recall_at_10",
                "required_article_hit_at_10",
                "required_article_recall_in_initial_context",
                "required_article_recall_in_final_context",
            ):
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
        context_traces = [
            row["context_trace"]
            for rows in datasets.values()
            for row in rows
            if isinstance(row.get("context_trace"), dict)
        ]
        scores[arm] = {
            "datasets": dataset_scores,
            "context_admission": {
                "case_count": len(context_traces),
                "cases_with_initial_omissions": sum(
                    bool(trace["initial_omitted_block_ids"]) for trace in context_traces
                ),
                "initial_omitted_blocks": sum(
                    len(trace["initial_omitted_block_ids"]) for trace in context_traces
                ),
                "cases_with_final_omissions": sum(
                    bool(trace["final_omitted_block_ids"]) for trace in context_traces
                ),
                "final_omitted_blocks": sum(
                    len(trace["final_omitted_block_ids"]) for trace in context_traces
                ),
            },
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
        **({"auxiliary_metrics": auxiliary_metrics} if auxiliary_metrics else {}),
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


def _context_group_limit(static: dict) -> int:
    reading_plan = static.get("reading_plan")
    if isinstance(reading_plan, dict):
        value = reading_plan.get("context_group_limit")
        if type(value) is int and 1 <= value <= MAX_CONTEXT_GROUPS:
            return value
        raise ValueError("reading-plan context group limit is missing")
    return TOP_K


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "plan",
            "planned-retrieval",
            "rerank",
            "preflight",
            "run",
            "score",
            "retrieval-grid",
            "retrieval-score",
            "reranker-grid",
            "reranker-score",
        ),
        required=True,
    )
    parser.add_argument("--frozen-root", type=Path, default=FROZEN_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--resume-grid-after-stop",
        action="store_true",
        help="continue unstarted grid cases after a recorded stop; never retry failures",
    )
    parser.add_argument(
        "--reuse-preparation-from",
        type=Path,
        help="copy only a verified SQLite/index checkpoint from a prior matching prepare run",
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--retrieval-grid",
        action="store_true",
        help="freeze the retrieval-only rewrite × reranker comparison protocol",
    )
    parser.add_argument(
        "--shared-initial-reranker",
        action="store_true",
        help=(
            "rerank the fixed and decomposed initial evidence pools once; Agent follow-up "
            "search remains hybrid to stay within the frozen pair budget"
        ),
    )
    parser.add_argument(
        "--rewrite-initial-only",
        action="store_true",
        help="inspect the first question per language direction before continuing the same grid",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        help="freeze a selected case slice for prepare, preflight, run, and score",
    )
    parser.add_argument(
        "--compare-arms",
        nargs="+",
        choices=ARMS,
        default=list(DEFAULT_ARMS),
        help="arms included in this run configuration and score",
    )
    parser.add_argument(
        "--document-embedding-token-limit",
        type=int,
        default=PREP_DOCUMENT_EMBED_TOKEN_LIMIT,
        help="per-run document embedding cap; never above the frozen global policy ceiling",
    )
    parser.add_argument(
        "--query-embedding-token-limit",
        type=int,
        default=PREP_QUERY_EMBED_TOKEN_LIMIT,
        help="per-run query embedding cap; never above the frozen global policy ceiling",
    )
    parser.add_argument(
        "--expected-static-configuration-sha256",
        help=(
            "require this protocol-frozen hash before preparing, running, or "
            "scoring a LongBench arm"
        ),
    )
    parser.add_argument(
        "--model-size",
        choices=("9b", "4b"),
        default="9b",
        help="select one locally pinned generator model; no fallback or download",
    )
    args = parser.parse_args()
    try:
        if args.phase not in ("score", "retrieval-score", "reranker-score") and not GPU_UUID:
            raise ValueError("set PAGE_CITE_RAG_GPU_UUID to an explicit GPU UUID")
        if (
            args.phase
            in (
                "retrieval-grid",
                "retrieval-score",
                "reranker-grid",
                "reranker-score",
            )
            and not args.retrieval_grid
        ):
            raise ValueError("retrieval-grid and reranker-grid phases require --retrieval-grid")
        if args.resume_grid_after_stop and args.phase not in ("retrieval-grid", "retrieval-score"):
            raise ValueError("--resume-grid-after-stop only applies to the stopped retrieval grid")
        if args.rewrite_initial_only and args.phase != "retrieval-grid":
            raise ValueError("--rewrite-initial-only requires --phase retrieval-grid")
        if args.reuse_preparation_from is not None and args.phase != "prepare":
            raise ValueError("--reuse-preparation-from is valid only with --phase prepare")
        if args.retrieval_grid and args.phase not in (
            "prepare",
            "retrieval-grid",
            "retrieval-score",
            "reranker-grid",
            "reranker-score",
        ):
            raise ValueError(
                "--retrieval-grid is only valid for retrieval-grid prepare, execution, and scoring"
            )
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
        selected_arms = (
            ["fixed-rag", "fixed-rag-reranked"] if args.retrieval_grid else args.compare_arms
        )
        static = static_configuration(
            root,
            manifest,
            selected_case_ids,
            selected_arms,
            generator_model_size=args.model_size,
            document_embedding_token_limit=args.document_embedding_token_limit,
            query_embedding_token_limit=args.query_embedding_token_limit,
            retrieval_grid=args.retrieval_grid,
            shared_initial_reranker=args.shared_initial_reranker,
        )
        if args.resume_grid_after_stop:
            current = static
            static = _validate_stopped_grid_configuration(run_root, current)
            audit_path = run_root / "grid-continuation-configuration.json"
            if not audit_path.exists():
                atomic_json_once(
                    audit_path,
                    {
                        "state": "explicit_continuation_after_gpu_stop",
                        "original_static_configuration_sha256": canonical_hash(static),
                        "current_static_configuration_sha256": canonical_hash(current),
                        "original_source_code_sha256": static["source_code_sha256"],
                        "continuation_source_code_sha256": current["source_code_sha256"],
                        "policy": "continue only unstarted cases; recorded failures remain missing",
                    },
                )
            else:
                saved_audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if saved_audit["current_static_configuration_sha256"] != canonical_hash(current):
                    raise ValueError("continuation code changed after configuration audit")
        verify_protocol_configuration_hash(
            static,
            args.expected_static_configuration_sha256,
            required=LONG_BENCH_OFFICIAL_ARM in args.compare_arms,
        )
        if (
            tuple(selected_arms) == QUALITY_V1_ARMS
            and args.phase in ("prepare", "plan", "planned-retrieval", "preflight", "run", "score")
            and (run_root / "configuration.json").is_file()
        ):
            stage = f"answer-{args.arm}" if args.phase == "run" else args.phase
            static = _validate_downstream_stage_configuration(run_root, static, stage=stage)
        if args.phase == "prepare":
            result = _prepare(
                root,
                run_root,
                manifest,
                selected_inputs,
                static,
                reuse_from=args.reuse_preparation_from,
            )
        elif args.phase == "plan":
            result = _run_reading_plans(run_root, static, selected_inputs)
        elif args.phase == "planned-retrieval":
            result = _run_planned_retrieval(run_root, static, selected_inputs)
        elif args.phase == "rerank":
            result = _rank_prepared_retrieval(run_root, static, selected_inputs)
        elif args.phase == "preflight":
            result = _preflight(run_root, static, selected_inputs)
        elif args.phase == "run":
            if args.arm is None:
                raise ValueError("--phase run requires --arm")
            result = (
                _run_longbench_official_arm(run_root, static, selected_inputs, args.case_id)
                if args.arm == LONG_BENCH_OFFICIAL_ARM
                else _run_arm(run_root, static, selected_inputs, args.arm, args.case_id)
            )
        elif args.phase == "retrieval-grid":
            result = _run_retrieval_grid(
                run_root,
                static,
                selected_inputs,
                rewrite_initial_only=args.rewrite_initial_only,
                allow_recorded_failures=args.resume_grid_after_stop,
            )
        elif args.phase == "reranker-grid":
            downstream_static = _validate_downstream_stage_configuration(
                run_root, static, stage="same-pool-reranker"
            )
            result = _rank_full_sentence_retrieval_grid(
                run_root, downstream_static, selected_inputs
            )
        elif args.phase == "reranker-score":
            selected_gold = load_gold(root, manifest)
            result = _score_same_pool_reranker_grid(
                run_root, manifest, selected_inputs, selected_gold
            )
        elif args.phase == "retrieval-score":
            selected_gold = load_gold(root, manifest)
            if (
                static["retrieval"].get("retrieval_grid", {}).get("protocol")
                == "full-sentence-rewrite-routing/v1"
            ):
                result = _score_full_sentence_retrieval_grid(
                    run_root, manifest, selected_inputs, selected_gold
                )
            else:
                result = _score_retrieval_grid(run_root, manifest, selected_inputs, selected_gold)
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
