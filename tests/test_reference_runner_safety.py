from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def reference_runner(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/compare_reference.py"
    spec = importlib.util.spec_from_file_location("compare_reference_test", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configuration_manifest(tmp_path):
    """Minimal configuration inputs, not copies of historical evaluation data."""

    def create(phase):
        manifest = {
            "phase": phase,
            "case_count": 1,
            "qasper": {"revision": "test", "split": "validation", "sha256": "test-qasper"},
            "xrag": {"revision": "test", "sha256": "test-xrag"},
            "drcd": {"revision": "test", "sha256": "test-drcd"},
            "pdfs": {},
            "parent_q3_sha256": "test-q3",
            "parent_p0_sha256": "test-p0",
            "parent_q5_sha256": "test-q5",
            "parent_q6_sha256": "test-q6",
            "shared": {
                "answer_contract": "test-contract",
                "exact_input_token_cap": 10_000,
                "output_tokens_per_call": 1_536,
                "template_reserve": 1_024,
                "common_context_preflight": True,
            },
            "selection_policy": {"max_sources": 12},
            "selection": {"seed": 42},
            "fallback_policy": "stop on technical selector failure",
            "resource_limits": {"memory_mib": 12_000},
            "generation_limits": {
                "invocation_wall_seconds": 60,
                "invocation_generation_seconds": 30,
            },
        }
        path = tmp_path / f"{phase}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest, path

    return create


def test_missing_gpu_selection_stops_before_execution(reference_runner, monkeypatch, capsys):
    monkeypatch.setattr(reference_runner, "GPU_UUID", "")
    monkeypatch.setattr(sys, "argv", ["compare_reference.py", "--preflight-only"])
    monkeypatch.setattr(
        reference_runner, "execute", lambda *_: pytest.fail("executed without a GPU selection")
    )

    assert reference_runner.main() == 1
    assert "PAGE_CITE_RAG_GPU_UUID" in capsys.readouterr().err


def test_new_output_directory_is_not_created_during_validation(reference_runner, tmp_path):
    target = tmp_path / "new-run"

    assert reference_runner.validate_output_target(target, False, {"run": 1}) is None
    assert not target.exists()


def test_nonempty_output_without_configuration_is_rejected(reference_runner, tmp_path):
    target = tmp_path / "partial-run"
    target.mkdir()
    (target / "A").mkdir()
    (target / "A" / "question.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="no configuration.json"):
        reference_runner.validate_output_target(target, True, {"run": 1})


def test_execute_validates_resume_before_starting_embedding_server(
    reference_runner, tmp_path, monkeypatch
):
    target = tmp_path / "partial-run"
    target.mkdir()
    (target / "stale.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reference_runner, "read_p0", lambda: {})
    monkeypatch.setattr(
        reference_runner,
        "static_configuration",
        lambda manifest, *, final_k: {"run": 1},
    )
    monkeypatch.setattr(
        reference_runner,
        "NativeEmbeddingServer",
        lambda *args, **kwargs: pytest.fail("embedding server started before resume validation"),
    )

    with pytest.raises(ValueError, match="no configuration.json"):
        reference_runner.execute(Namespace(output_dir=target, resume=True, run=True, final_k=10))


def test_existing_run_requires_resume(reference_runner, tmp_path):
    target = tmp_path / "existing-run"
    target.mkdir()
    (target / "configuration.json").write_text(json.dumps({"static": {"run": 1}}), encoding="utf-8")

    with pytest.raises(ValueError, match="pass --resume"):
        reference_runner.validate_output_target(target, False, {"run": 1})


def test_resume_rejects_different_frozen_configuration(reference_runner, tmp_path):
    target = tmp_path / "existing-run"
    target.mkdir()
    (target / "configuration.json").write_text(json.dumps({"static": {"run": 1}}), encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to mix"):
        reference_runner.validate_output_target(target, True, {"run": 2})


def test_resume_accepts_matching_configuration(reference_runner, tmp_path):
    target = tmp_path / "existing-run"
    target.mkdir()
    config = {"static": {"run": 1}, "full": {"input_tokens": 100}}
    (target / "configuration.json").write_text(json.dumps(config), encoding="utf-8")

    assert reference_runner.validate_output_target(target, True, {"run": 1}) == config


def test_q6_resume_allows_only_full_answer_and_runner_accounting_hash_updates(reference_runner):
    original = {
        "run_version": "quality-convergence-q6-a1-b1-k20-v1",
        "models": {"generator": "pinned-local"},
        "source_code_sha256": {
            "runner": "old-runner",
            "full_answer": "old-answer",
            "llama": "same",
        },
    }
    continued = {
        **original,
        "source_code_sha256": {
            "runner": "new-runner",
            "full_answer": "new-answer",
            "llama": "same",
        },
    }
    unrelated_change = {**continued, "models": {"generator": "other"}}
    retrieval_change = {
        **continued,
        "source_code_sha256": {
            **continued["source_code_sha256"],
            "retrieval": "changed",
        },
    }

    assert reference_runner._q6_failure_accounting_code_update(original, continued)
    assert not reference_runner._q6_failure_accounting_code_update(original, unrelated_change)
    assert not reference_runner._q6_failure_accounting_code_update(original, retrieval_change)

    confirmation = {
        **original,
        "run_version": "quality-convergence-q6-a1-confirmation-v1",
    }
    continued_confirmation = {
        **confirmation,
        "source_code_sha256": continued["source_code_sha256"],
    }
    assert reference_runner._q6_failure_accounting_code_update(confirmation, continued_confirmation)


def test_q6_confirmation_selection_is_deterministic_stratified_and_one_per_paper(
    reference_runner, monkeypatch
):
    monkeypatch.setattr(
        reference_runner,
        "align_question",
        lambda paper, question: SimpleNamespace(
            exclusion_reason=(
                "evidence_not_aligned" if question.question_id == "question-1" else None
            )
        ),
    )
    papers = []
    for index in range(12):
        unanswerable = index >= 8
        papers.append(
            SimpleNamespace(
                paper_id=f"paper-{index}",
                questions=(
                    SimpleNamespace(
                        question_id=f"question-{index}",
                        annotations=(SimpleNamespace(unanswerable=unanswerable),),
                    ),
                ),
            )
        )

    first = reference_runner.q6_confirmation_case_specs(papers, {"paper-0"})
    second = reference_runner.q6_confirmation_case_specs(list(reversed(papers)), {"paper-0"})

    assert first == second
    assert len(first) == 8
    assert len({case["paper_id"] for case in first}) == 8
    assert "paper-0" not in {case["paper_id"] for case in first}
    assert "question-1" not in {case["question_id"] for case in first}
    selected = {paper.paper_id: paper for paper in papers}
    selected_labels = [
        selected[case["paper_id"]].questions[0].annotations[0].unanswerable for case in first
    ]
    assert selected_labels.count(False) == 6
    assert selected_labels.count(True) == 2


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {
                "result": {"budget": None, "generation_stats": None},
                "model_diagnostics": {
                    "last_completion": {"tokens_evaluated": 5, "tokens_predicted": 2}
                },
            },
            (5, 2),
        ),
        (
            {
                "result": {
                    "budget": {"exact_input_tokens": 11},
                    "generation_stats": {"eval_count": 4},
                }
            },
            (11, 4),
        ),
        ({"input_tokens": 8, "output_tokens": 3, "result": None}, (8, 3)),
    ],
)
def test_resume_token_usage_accepts_null_result_fields(reference_runner, record, expected):
    assert reference_runner.resumed_record_token_usage(record) == expected


def test_q6_output_validation_trace_serializes_pydantic_accounting(reference_runner):
    from page_cite_rag.generation import OutputFailureAccounting, OutputValidationError

    error = OutputValidationError(
        "schema_invalid",
        OutputFailureAccounting(
            prompt_eval_count=100,
            eval_count=3,
            elapsed_ms=1.0,
            stop_type="eos",
        ),
        raw_model_output='{"invalid":true}',
        validation_errors=({"type": "missing", "loc": ["answer_claims"]},),
    )

    trace = reference_runner._q6_output_validation_trace(error, "A1")

    assert trace == {
        "failure_stage": "A1-final",
        "error_type": "OutputValidationError",
        "raw_model_output": '{"invalid":true}',
        "validation_errors": [{"type": "missing", "loc": ["answer_claims"]}],
        "failure_accounting": {
            "prompt_eval_count": 100,
            "eval_count": 3,
            "elapsed_ms": 1.0,
            "stop_type": "eos",
        },
    }


def test_q6_aggregate_marks_unknown_failure_usage_as_incomplete(reference_runner, tmp_path):
    target = tmp_path / "q6-run"
    result = target / "A1" / "case-1.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "execution": "failed",
                "completion_calls": 1,
                "cooldown_seconds_total": 0,
                "cooldown_usage_unknown": True,
                "stage_usage": [
                    {
                        "stage": "A1-final",
                        "calls": 1,
                        "status": "failed_recording_error",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "elapsed_ms": 0,
                        "usage_unknown": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (target / "invocation-ledger.jsonl").write_text("", encoding="utf-8")

    summary = reference_runner._q6_aggregate(target, ["case-1"], ("A1",), wall_seconds=0)

    assert summary["completion_calls"] == 1
    assert summary["unknown_usage_stages"] == 1
    assert summary["unknown_cooldown_slots"] == 1
    assert not summary["token_totals_complete"]


@pytest.mark.parametrize(
    ("phase", "guard_failures", "client_closed", "error", "expected"),
    [
        ("Q5", [], False, ValueError(), True),
        ("Q6", [], False, ValueError(), True),
        ("Q5", ["TimeoutError"], False, ValueError(), False),
        ("Q5", [], True, ValueError(), False),
        ("Q3", [], False, ValueError(), False),
        ("Q5", [], False, KeyboardInterrupt(), False),
    ],
)
def test_only_q5_q6_continue_after_recording_non_guard_failures(
    reference_runner, phase, guard_failures, client_closed, error, expected
):
    assert (
        reference_runner.continue_after_case_failure(
            phase, guard_failures, client_closed=client_closed, error=error
        )
        is expected
    )


@pytest.mark.parametrize(
    ("error_type", "error", "expected"),
    [
        ("ModelError", "generation client is closed", True),
        ("OutputValidationError", "schema_invalid", False),
        ("ModelError", "request timed out", False),
    ],
)
def test_identifies_closed_client_without_a_new_generation_call(
    reference_runner, error_type, error, expected
):
    assert reference_runner.client_closed_before_generation(error_type, error) is expected


@pytest.mark.parametrize("state", ["missing", "unknown", "changed"])
def test_manifest_rejects_missing_unrecognized_or_changed_files(
    reference_runner, tmp_path, monkeypatch, state
):
    path = tmp_path / "manifest.json"
    if state != "missing":
        path.write_text("{}", encoding="utf-8")
    if state == "changed":
        monkeypatch.setattr(reference_runner, "Q4_MANIFEST_PATH", path)
        monkeypatch.setattr(reference_runner, "Q4_MANIFEST_SHA256", "not-the-file-hash")

    with pytest.raises(ValueError, match="missing|checksum changed"):
        reference_runner.read_manifest(path)


def test_q4_static_configuration_records_comparison_and_agent_limits(
    reference_runner, configuration_manifest
):
    manifest, path = configuration_manifest("Q4")

    static = reference_runner.static_configuration(
        manifest,
        final_k=10,
        manifest_path=path,
    )

    assert set(static["arms"]) == {"fixed", "agent"}
    assert static["agent"]["model_call_limit"] == 2
    assert static["agent"]["tool_call_limit"] == 4
    assert static["agent"]["final_answer_calls"] == 1
    assert static["shared"]["exact_input_token_cap"] == 10_000
    assert "reranker" not in static["models"]
    assert static["manifest_sha256"] == reference_runner.sha256(path)


def test_q5_static_configuration_records_shared_answer_and_failure_contracts(
    reference_runner, configuration_manifest
):
    manifest, path = configuration_manifest("Q5")
    static = reference_runner.static_configuration(
        manifest,
        final_k=10,
        manifest_path=path,
    )
    assert static["shared"]["same_final_answer_schema_and_prompt"] is True
    assert static["shared"]["continue_after_recorded_non_guard_case_failure"] is True
    assert static["shared"]["stop_after_generation_client_closes"] is True
    assert static["evidence_first"]["max_completion_calls_per_case"] == 2
    assert static["parent_q3_sha256"] == manifest["parent_q3_sha256"]
    assert static["manifest_sha256"] == reference_runner.sha256(path)


def test_q6_static_configuration_records_candidate_and_selection_policy(
    reference_runner, configuration_manifest
):
    manifest, path = configuration_manifest("Q6")
    static = reference_runner.static_configuration(
        manifest,
        final_k=10,
        manifest_path=path,
    )
    assert static["shared"]["candidate_pool_k"] == 20
    assert static["arms"]["A1"].startswith("common top-20")
    assert static["fallback_policy"] == manifest["fallback_policy"]
    assert static["selection"] == manifest["selection_policy"]
    assert static["manifest_sha256"] == reference_runner.sha256(path)


def test_confirmation_static_configuration_records_single_arm_and_lineage(
    reference_runner, configuration_manifest
):
    manifest, path = configuration_manifest("Q6-confirmation")
    static = reference_runner.static_configuration(
        manifest,
        final_k=10,
        manifest_path=path,
    )

    assert static["run_version"] == "quality-convergence-q6-a1-confirmation-v1"
    assert set(static["arms"]) == {"A1"}
    assert static["parent_q6_sha256"] == manifest["parent_q6_sha256"]
    assert static["confirmation_selection"] == manifest["selection"]
    assert static["manifest_sha256"] == reference_runner.sha256(path)


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        ({"selection": None}, {}),
        ({"selection": {"generation_stats": {"eval_count": 12}}}, {"eval_count": 12}),
    ],
)
def test_evidence_selection_stats_accepts_failed_and_successful_traces(
    reference_runner, trace, expected
):
    assert reference_runner.evidence_selection_generation_stats(trace) == expected


def test_preflight_uses_runtime_context_admission_for_both_arms(reference_runner, monkeypatch):
    block = SimpleNamespace(block_id="block-1")
    group = SimpleNamespace(source_block_ids=(block.block_id,))
    calls = []
    budget = SimpleNamespace(
        exact_input_tokens=123,
        context_limit=16_384,
        output_reserved=1_536,
    )

    class Workflow:
        prompt_version = "test"

        def _context_blocks(self, _result, blocks):
            return blocks

        def _read_hits(self, _result, _question):
            return (block,)

        def _context(self, question, language, mode, blocks, requested, *, context_groups=()):
            calls.append((question, language, mode, tuple(blocks), tuple(context_groups)))
            return tuple(blocks), (), (), budget

    case = reference_runner.Case(
        case_id="pdf-1",
        kind="pdf",
        question="What is the result?",
        language="en",
        document_id="document-1",
        store=object(),
        a_result=SimpleNamespace(context_groups=(group,)),
        r_result=SimpleNamespace(context_groups=(group,)),
        source_blocks=(block,),
        source_context_groups=(group,),
    )
    workflow = Workflow()
    monkeypatch.setattr(
        reference_runner, "FrozenSearchRetriever", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(reference_runner, "document_workflow", lambda *_args, **_kwargs: workflow)

    ready, admission = reference_runner.preflight_workflows(
        {case.case_id: case}, SimpleNamespace(identity={"model": "test"}), object(), final_k=5
    )

    assert len(calls) == 3
    assert calls[0][3] == (block,)
    assert calls[0][4] == (group,)
    assert calls[1][3] == (block,)
    assert calls[1][4] == (group,)
    assert calls[2][3] == (block,)
    assert calls[2][4] == (group,)
    assert ready == {"pdf-1/A": workflow, "pdf-1/R": workflow, "pdf-1/C": workflow}
    assert admission["pdf-1/A"]["exact_input_tokens"] == 123
    assert admission["pdf-1/R"]["exact_input_tokens"] == 123
    assert admission["pdf-1/C"]["eligible"] is True


def test_docling_final_k_keeps_complete_ranked_chunks(reference_runner):
    from page_cite_rag.models import Block
    from page_cite_rag.retrieval import SearchContextGroup, SearchHit, SearchResult

    document_id = "a" * 64
    blocks = tuple(
        Block(
            document_id=document_id,
            block_id=char * 64,
            ordinal=index,
            unit_index=index,
            page_number=index,
            line_start=1,
            line_end=1,
            char_start=0,
            char_end=len(text),
            text=text,
        )
        for index, (char, text) in enumerate((("b", "first"), ("c", "second")), 1)
    )
    groups = tuple(
        SearchContextGroup(
            chunk_index=index - 1,
            text=f"chunk {index}",
            source_refs=(f"page:{index}",),
            source_block_ids=(block.block_id,),
        )
        for index, block in enumerate(blocks, 1)
    )
    result = SearchResult(
        document_id=document_id,
        index_id="index",
        method="hybrid",
        query="question",
        hits=tuple(
            SearchHit(rank=index, score=1 / index, block=block)
            for index, block in enumerate(blocks, 1)
        ),
        elapsed_ms=1,
        context_groups=groups,
    )

    limited = reference_runner._limit_result(result, 1)

    assert limited.context_groups == groups[:1]
    assert tuple(hit.block.block_id for hit in limited.hits) == (blocks[0].block_id,)


def test_preflight_supports_single_arm_without_whole_document_diagnostic(
    reference_runner, monkeypatch
):
    block = SimpleNamespace(block_id="block-1")
    group = SimpleNamespace(source_block_ids=(block.block_id,))
    calls = []
    budget = SimpleNamespace(exact_input_tokens=123, context_limit=16_384, output_reserved=1_536)

    class Workflow:
        prompt_version = "test"

        def _context_blocks(self, _result, blocks):
            return blocks

        def _read_hits(self, _result, _question):
            return (block,)

        def _context(self, question, language, mode, blocks, requested, *, context_groups=()):
            calls.append((question, language, mode, tuple(blocks), tuple(context_groups)))
            return tuple(blocks), (), (), budget

    case = reference_runner.Case(
        case_id="question-1",
        kind="qasper",
        question="What is the result?",
        language="en",
        document_id="document-1",
        store=object(),
        a_result=SimpleNamespace(context_groups=(group,)),
        source_blocks=(block,),
    )
    workflow = Workflow()
    monkeypatch.setattr(
        reference_runner, "FrozenSearchRetriever", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(reference_runner, "document_workflow", lambda *_args, **_kwargs: workflow)

    ready, admission = reference_runner.preflight_workflows(
        {case.case_id: case},
        SimpleNamespace(identity={"model": "test"}),
        object(),
        final_k=5,
        arms=("A",),
        include_full_document=False,
    )

    assert len(calls) == 1
    assert ready == {"question-1/A": workflow}
    assert tuple(admission) == ("question-1/A",)
    assert admission["question-1/A"]["exact_input_tokens"] == 123


def test_character_f1_is_whitespace_and_punctuation_invariant(reference_runner):
    assert reference_runner.character_f1("「答案，答案！」", "答案答案") == 1.0
    assert reference_runner.character_f1("哈哈哈", "哈哈") == pytest.approx(0.8)


def test_qasper_scoring_keeps_preflight_ineligible_questions_in_fixed_gold_denominator(
    reference_runner, tmp_path, monkeypatch
):
    output = tmp_path / "run"
    output.mkdir()
    (output / "preflight.json").write_text(
        json.dumps({"case-1/A": {"eligible": True}, "case-1/R": {"eligible": False}}),
        encoding="utf-8",
    )
    case = {"id": "case-1", "question_id": "question-1"}
    gold = {
        "question-1": (
            type(
                "Reference",
                (),
                {"answer": "alpha", "evidence": ("passage",), "answer_type": "extractive"},
            )(),
        )
    }
    monkeypatch.setattr(reference_runner, "read_answer_references", lambda *_: gold)
    path = reference_runner.result_path(output, "A", "case-1")
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "execution": "ok",
                "result": {
                    "answer": "alpha",
                    "answer_claims": [{"text": "alpha", "citations": [{"quote": "passage"}]}],
                },
            }
        ),
        encoding="utf-8",
    )

    scored = reference_runner.score_qasper(output, [case])["scores"]

    assert scored["A"]["applicable_questions"] == 1
    assert scored["A"]["metrics"]["Answer F1"] == 1
    assert scored["A"]["metrics"]["Evidence F1"] == 1
    assert scored["R"]["applicable_questions"] == 1
    assert scored["R"]["not_applicable_questions"] == 1
    assert scored["R"]["metrics"]["Answer F1"] == 0
    assert scored["R"]["metrics"]["Missing predictions"] == 1


def test_qasper_scorer_maps_explicit_refusal_to_unanswerable(
    reference_runner, tmp_path, monkeypatch
):
    output = tmp_path / "run"
    (output / "A").mkdir(parents=True)
    case = {"id": "no-answer", "question_id": "question-none"}
    gold = {
        "question-none": (
            type(
                "Reference",
                (),
                {"answer": "Unanswerable", "evidence": (), "answer_type": "none"},
            )(),
        )
    }
    monkeypatch.setattr(reference_runner, "read_answer_references", lambda *_: gold)
    path = reference_runner.result_path(output, "A", "no-answer")
    path.write_text(
        json.dumps(
            {
                "execution": "ok",
                "result": {
                    "status": "insufficient_evidence",
                    "answer": "The supplied evidence is insufficient.",
                    "answer_claims": [],
                    "detail_claims": [],
                },
            }
        ),
        encoding="utf-8",
    )

    scores = reference_runner.score_qasper(output, [case], split="validation", arms=("A",))

    assert scores["scorer_version"] == "gold-fixed-subgroups/status-normalized/evidence-pr-v4"
    assert scores["scores"]["A"]["metrics"]["Answer F1"] == 1.0
    assert scores["scores"]["A"]["gold_fixed_groups"]["unanswerable"]["questions"] == 1
    assert scores["scores"]["A"]["correct_refusals"] == 1


def test_qasper_fallback_answer_is_scored_separately_from_raw_b1(
    reference_runner, tmp_path, monkeypatch
):
    output = tmp_path / "run"
    path = reference_runner.result_path(output, "B1", "case-1")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "execution": "fallback",
                "fallback": {
                    "execution": "ok",
                    "result": {
                        "status": "answered",
                        "answer": "alpha",
                        "answer_claims": [{"text": "alpha", "citations": [{"quote": "passage"}]}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    case = {"id": "case-1", "question_id": "question-1"}
    gold = {
        "question-1": (
            type(
                "Reference",
                (),
                {"answer": "alpha", "evidence": ("passage",), "answer_type": "extractive"},
            )(),
        )
    }
    monkeypatch.setattr(reference_runner, "read_answer_references", lambda *_: gold)

    scores = reference_runner.score_qasper(
        output,
        [case],
        split="validation",
        arms=("B1", "B1_fallback_served"),
    )["scores"]

    assert scores["B1"]["metrics"]["Missing predictions"] == 1
    assert scores["B1_fallback_served"]["metrics"]["Answer F1"] == 1
