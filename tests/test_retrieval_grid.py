from __future__ import annotations

import hashlib
import json
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from page_cite_rag.hybrid import fuse_routed_hybrid_results
from page_cite_rag.models import Block
from page_cite_rag.retrieval import SearchBranchTrace, SearchHit, SearchResult
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.text import TokenBlockFactory

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_reading_eval.py"
_SPEC = spec_from_file_location("reading_eval_grid_under_test", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
reading_eval = module_from_spec(_SPEC)
_SPEC.loader.exec_module(reading_eval)


@pytest.fixture(autouse=True)
def _local_embedding_tokenizer(monkeypatch, embedding_tokenizer_dir):
    monkeypatch.setattr(reading_eval, "DEFAULT_MODEL_DIR", embedding_tokenizer_dir)


def test_grid_continuation_skips_recorded_failures_and_rejects_unknown_attempts():
    inputs = [{"case_id": key} for key in ("done", "failed", "pending")]
    attempts = {"done": {}, "failed": {}}
    assert reading_eval._pending_grid_cases(inputs, attempts, {"done": {}}, {"failed": {}}) == [
        {"case_id": "pending"}
    ]
    with pytest.raises(ValueError, match="already started"):
        reading_eval._pending_grid_cases(inputs, attempts, {"done": {}}, {})
    with pytest.raises(ValueError, match="failure records"):
        reading_eval._pending_grid_cases(inputs, attempts, {"done": {}}, {"done": {}})


def test_grid_continuation_requires_frozen_snapshot_and_unchanged_retrieval(tmp_path):
    old_runner = b"original runner"
    frozen = {
        "source_code_sha256": {
            "scripts/run_reading_eval.py": hashlib.sha256(old_runner).hexdigest(),
            "src/page_cite_rag/hybrid.py": "unchanged",
        },
        "answer": {"retry": False},
        "retrieval": {"top_k": 10},
    }
    current = json.loads(json.dumps(frozen))
    current["source_code_sha256"]["scripts/run_reading_eval.py"] = "new-runner"
    current["answer"]["case_deadline_seconds"] = 600
    (tmp_path / "configuration.json").write_text(json.dumps({"static": frozen}))
    (tmp_path / "GPU-STOP.json").write_text("{}")
    with zipfile.ZipFile(tmp_path / "source-at-revalidation.zip", "w") as archive:
        archive.writestr("scripts/run_reading_eval.py", old_runner)
    assert reading_eval._validate_stopped_grid_configuration(tmp_path, current) == frozen
    current["source_code_sha256"]["src/page_cite_rag/hybrid.py"] = "changed"
    with pytest.raises(ValueError, match="retrieval implementation changed"):
        reading_eval._validate_stopped_grid_configuration(tmp_path, current)


def test_rewrite_initial_cases_include_each_direction_in_deterministic_order():
    cases = [
        {"case_id": "z", "direction": "en->en"},
        {"case_id": "b", "direction": "zh->en"},
        {"case_id": "a", "direction": "en->en"},
        {"case_id": "d", "direction": "en->zh"},
        {"case_id": "c", "direction": "zh->zh"},
    ]
    assert [case["case_id"] for case in reading_eval._rewrite_initial_cases(cases)] == [
        "a",
        "d",
        "b",
        "c",
    ]


def test_rewrite_contract_stops_budget_mismatch_and_keeps_original_query():
    case = {"question": "What is the condition?"}
    reading_eval._validate_rewrite_record(
        case,
        {
            "original_query": case["question"],
            "lexical_query": case["question"] + " condition",
            "status": "rewritten",
            "reason": "keyword_expansion",
        },
        None,
    )
    with pytest.raises(ValueError, match="budget contract"):
        reading_eval._validate_rewrite_record(
            case,
            {
                "original_query": case["question"],
                "lexical_query": case["question"],
                "status": "degraded",
                "reason": "generation_budget_mismatch",
            },
            None,
        )
    with pytest.raises(ValueError, match="original question"):
        reading_eval._validate_rewrite_record(
            case,
            {
                "original_query": case["question"],
                "lexical_query": "condition",
                "status": "rewritten",
                "reason": "keyword_expansion",
            },
            None,
        )


def test_full_sentence_rewrite_contract_accepts_rewrite_and_safe_fallback():
    question = "How does method A compare at >= 90%?"
    rewritten = {
        "original_query": question,
        "lexical_query": "Compare method A at >= 90%.",
        "mode": "full_sentence",
        "status": "rewritten",
        "reason": "full_sentence_semantic_equivalence_unverified",
        "version": "full-sentence-document-language/v1",
    }
    reading_eval._validate_rewrite_record({"question": question}, rewritten, None)
    fallback = {
        **rewritten,
        "lexical_query": question,
        "status": "degraded",
        "reason": "full_sentence_output_out_of_bounds",
    }
    reading_eval._validate_rewrite_record({"question": question}, fallback, None)


def test_persisted_branch_trace_reconstructs_rank_only_search_and_routed_variants():
    document_id = "1" * 64
    blocks = {
        hashlib.sha256(f"{document_id}-{ordinal}".encode()).hexdigest(): Block(
            block_id=hashlib.sha256(f"{document_id}-{ordinal}".encode()).hexdigest(),
            document_id=document_id,
            unit_index=1,
            ordinal=ordinal,
            text=f"block {ordinal}",
            char_start=0,
            char_end=7,
            line_start=1,
            line_end=1,
        )
        for ordinal in range(1, 6)
    }

    class Store:
        def get_block(self, requested_document_id, block_id):
            assert requested_document_id == document_id
            return blocks.get(block_id)

    def branch(method, ordinals, query, index_id):
        return SearchResult(
            document_id=document_id,
            index_id=index_id,
            method=method,
            query=query,
            elapsed_ms=1.0,
            hits=tuple(
                SearchHit(
                    rank=rank,
                    score=1.0,
                    block=blocks[hashlib.sha256(f"{document_id}-{ordinal}".encode()).hexdigest()],
                )
                for rank, ordinal in enumerate(ordinals, 1)
            ),
        )

    original_lexical = branch("bm25", [1, 2, 3], "question", "bm25")
    original_dense = branch("dense", [3, 4, 5], "question", "dense")
    trace_lexical = SearchBranchTrace(
        name="lexical",
        index_id="bm25",
        elapsed_ms=1.0,
        block_ids=tuple(hit.block.block_id for hit in original_lexical.hits),
    )
    trace_dense = SearchBranchTrace(
        name="dense",
        index_id="dense",
        elapsed_ms=1.0,
        block_ids=tuple(hit.block.block_id for hit in original_dense.hits),
    )
    restored = reading_eval._restore_search_branch(
        Store(), document_id, "question", trace_lexical, "bm25"
    )
    restored_dense = reading_eval._restore_search_branch(
        Store(), document_id, "question", trace_dense, "dense"
    )
    expected = fuse_routed_hybrid_results(
        "question", [original_lexical], original_dense, lexical_top_k=3, top_k=6
    )
    actual = fuse_routed_hybrid_results(
        "question", [restored], restored_dense, lexical_top_k=3, top_k=6
    )
    assert [hit.block.block_id for hit in restored.hits] == [
        hit.block.block_id for hit in original_lexical.hits
    ]
    assert [hit.block.block_id for hit in actual.hits] == [
        hit.block.block_id for hit in expected.hits
    ]
    assert len({block_id for trace in actual.branches for block_id in trace.block_ids}) <= 6


def test_case_deadline_cancels_only_the_owned_model_and_retrieval():
    events = []

    class Model:
        def close(self):
            events.append("model_closed")

    class Retrieval:
        def cancel(self):
            events.append("retrieval_cancelled")

    with reading_eval._owned_case_deadline(Model(), Retrieval(), 0.01) as expired:
        assert expired.wait(timeout=1)
    assert events == ["retrieval_cancelled", "model_closed"]

    events.clear()
    with reading_eval._owned_case_deadline(Model(), Retrieval(), 1) as expired:
        assert not expired.is_set()
    assert events == []


@pytest.mark.parametrize("failed", [False, True])
def test_retrieval_score_reads_baseline_reranker_cost_from_grid_rows(tmp_path, failed):
    grid = tmp_path / "retrieval-grid"
    prepared = tmp_path / "prepared"
    grid.mkdir()
    prepared.mkdir()
    variants = {
        name: {
            "candidate_passage_ids": ["p1"],
            "context_passage_ids": ["p1"],
            "retrieval_elapsed_ms": 10,
        }
        for name in (
            "original_hybrid",
            "keyword_expansion_hybrid",
            "original_hybrid_reranked",
            "keyword_expansion_hybrid_reranked",
        )
    }
    row = {
        "case_id": "case",
        "document_cluster": "document",
        "direction": "en->en",
        "variants": variants,
        "base_reranker_pair_tokens": [7, 11],
        "base_reranker_timing_ms": {"total": 13},
        "reranker_pair_tokens": [19],
        "reranker_timing_ms": {"total": 17},
    }
    if failed:
        row["variants"]["keyword_expansion_hybrid_reranked"] = None
        row["reranker_timing_ms"] = None
        row["reranker_pair_tokens"] = []
    (grid / "retrieval-grid.jsonl").write_text(json.dumps(row) + "\n")
    (prepared / "retrieval.jsonl").write_text(
        json.dumps({"case_id": "case", "candidate_result": {"elapsed_ms": 10}}) + "\n"
    )
    inputs = [
        {
            "case_id": "case",
            "direction": "en->en",
            "dataset": "qasper",
            "source": {"paragraphs": [{"passage_id": "p1"}]},
        }
    ]
    gold = [{"case_id": "case", "unanswerable": False, "evidence_reference_passage_ids": [["p1"]]}]
    report = reading_eval._score_retrieval_grid(
        tmp_path, {"files": {"gold.jsonl": {"sha256": "test"}}}, inputs, gold
    )
    cost = report["measured_cost_by_variant"]["original_hybrid_reranked"]
    assert cost["reranker_pair_tokens"] == 18
    assert cost["per_case_component_ms"]["p50"] == 23
    assert "not complete end-to-end" in cost["timing_scope"]
    if failed:
        assert report["rewritten_reranker_missing_count"] == 1
        assert (
            report["case_details"][0]["metrics_by_variant"]["keyword_expansion_hybrid_reranked"]
            is None
        )
        assert report["selection"]["selected_variant"] is None


def test_retrieval_metrics_accept_any_complete_alternative_evidence_set():
    gold = {
        "unanswerable": False,
        "evidence_reference_passage_ids": [["p1", "p2"], ["p3"]],
    }

    metrics = reading_eval._retrieval_evidence_metrics(gold, ["p1", "p2"], {"p1", "p2", "p3"})

    assert metrics == {
        "reference_aware_recall": 1.0,
        "union_recall": pytest.approx(2 / 3),
        "complete_evidence_set_coverage": 1.0,
        "reference_count": 2,
        "gold_evidence_passage_count": 3,
    }


def test_retrieval_metrics_exclude_unmappable_sets_but_keep_valid_alternatives():
    gold = {
        "unanswerable": False,
        "evidence_reference_passage_ids": [["p1", "unmapped"], ["p2"]],
    }

    metrics = reading_eval._retrieval_evidence_metrics(gold, ["p2"], {"p1", "p2"})

    assert metrics is not None
    assert metrics["reference_aware_recall"] == 1.0
    assert metrics["reference_count"] == 1


@pytest.mark.parametrize(
    "gold",
    [
        {"unanswerable": True, "evidence_reference_passage_ids": [["p1"]]},
        {"unanswerable": None, "evidence_reference_passage_ids": None},
        {"unanswerable": False, "evidence_reference_passage_ids": [[]]},
    ],
)
def test_retrieval_metrics_do_not_impute_unanswerable_or_missing_gold(gold):
    assert reading_eval._retrieval_evidence_metrics(gold, ["p1"], {"p1"}) is None


def test_cluster_bootstrap_is_paired_reproducible_and_reports_source_clusters():
    rows = [
        {"document_cluster": "paper-a", "value": 0.25},
        {"document_cluster": "paper-a", "value": 0.75},
        {"document_cluster": "paper-b", "value": 1.0},
    ]

    first = reading_eval._cluster_bootstrap_mean_ci(rows, seed=20261027, iterations=500)
    second = reading_eval._cluster_bootstrap_mean_ci(rows, seed=20261027, iterations=500)

    assert first == second
    assert first["estimate"] == pytest.approx(2 / 3)
    assert first["case_count"] == 3
    assert first["cluster_count"] == 2
    assert first["ci95"][0] <= first["estimate"] <= first["ci95"][1]


def test_append_only_grid_artifacts_reject_duplicate_case_ids(tmp_path):
    path = tmp_path / "rewrites.jsonl"
    reading_eval.append_jsonl_record(path, {"case_id": "case-1", "status": "complete"})
    assert reading_eval.load_unique_jsonl(path) == {
        "case-1": {"case_id": "case-1", "status": "complete"}
    }
    reading_eval.append_jsonl_record(path, {"case_id": "case-1", "status": "duplicate"})

    with pytest.raises(ValueError, match="duplicate case_id"):
        reading_eval.load_unique_jsonl(path)


def test_started_reranker_attempt_without_result_is_not_retried():
    with pytest.raises(ValueError, match="do not retry case case-2"):
        reading_eval.reject_interrupted_attempts(
            {"case-1": {"state": "started"}, "case-2": {"state": "started"}},
            {"case-1": {"state": "complete"}},
            "base reranker",
        )


def test_completed_reranker_attempts_can_resume_after_a_process_restart():
    reading_eval.reject_interrupted_attempts(
        {"case-1": {"state": "started"}},
        {"case-1": {"state": "complete"}},
        "base reranker",
    )


def test_translated_query_variants_share_source_language_metadata(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "reading.sqlite3")
    chunker = TokenBlockFactory(reading_eval.DEFAULT_MODEL_DIR)
    source = {"format": "plain_text", "text": "A public source paragraph."}
    base = {
        "dataset": "longbench-v1-multifieldqa-zh",
        "source_id": "public-source",
        "source": source,
        "source_language": "zh-Hant",
    }

    first_id, _ = reading_eval._case_document(
        store, {**base, "case_id": "source--zh-zh", "language": "zh-Hant"}, chunker=chunker
    )
    second_id, _ = reading_eval._case_document(
        store, {**base, "case_id": "source--en-zh", "language": "en"}, chunker=chunker
    )

    assert first_id == second_id
    assert store.get_document(first_id).language == "zh-Hant"
