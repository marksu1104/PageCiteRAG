"""Single-user, bounded local application runtime; no GPU model at import time."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from page_cite_rag.application import (
    FrozenSearchRetriever,
    RetrievalMethod,
    ask_document,
    document_retriever_parts,
    document_workflow,
    explain_source,
)
from page_cite_rag.hybrid import Retriever
from page_cite_rag.retrieval import SearchResult
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.workflows import AnswerResult, Language, read_verified_source


class ServiceBusyError(RuntimeError):
    """The fixed local queue cannot admit another request."""


class ServiceStoppedError(RuntimeError):
    """The local service is stopped or a request was explicitly cancelled."""


class ServiceDeadlineError(TimeoutError):
    """A queued or running request exceeded its total deadline."""


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    database_path: Path
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    method: RetrievalMethod = "bm25"
    top_k: int = Field(default=6, ge=1, le=20)
    neighbor_blocks: int = Field(default=1, ge=0, le=2)
    queue_capacity: int = Field(default=2, ge=0, le=8)
    request_timeout_seconds: float = Field(default=180, gt=1, le=300)
    idle_seconds: float = Field(default=300, gt=0, le=300)


class ServiceTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_id: str
    queue_ms: float = Field(ge=0, allow_inf_nan=False)
    startup_ms: float = Field(ge=0, allow_inf_nan=False)
    retrieval_stage_ms: float = Field(ge=0, allow_inf_nan=False)
    workflow_ms: float = Field(ge=0, allow_inf_nan=False)
    generation_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    total_ms: float = Field(ge=0, allow_inf_nan=False)
    load_count: int = Field(ge=0)
    model_pid: int | None = None
    retrieval_mode: Literal["resident", "staged"]


class ServiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    result: AnswerResult
    trace: ServiceTrace


class BlockPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    block_id: str
    ordinal: int
    page_number: int | None
    line_start: int
    line_end: int
    preview: str


class BlockPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    blocks: list[BlockPreview]
    next_after: int | None


@dataclass
class _Job:
    request_id: str
    mode: Literal["ask", "explain"]
    question: str
    language: Language
    rewrite_query: bool
    block_id: str | None
    created: float
    deadline: float
    future: Future[ServiceAnswer] = field(default_factory=Future)
    cancel_reason: Literal["cancelled", "deadline"] | None = None
    deadline_timer: threading.Timer | None = None


class DocumentService:
    """One worker, bounded waiters and owned model lifetime for a bound document.

    Model calls are synchronous in the worker. A deadline/cancellation closes only
    this service's model, interrupting its active local HTTP request. No hidden
    retry or CPU fallback is performed. The model is not loaded on startup.
    """

    def __init__(
        self,
        config: ServiceConfig,
        model_config,
        model_factory: Callable[[], object],
        *,
        encoder=None,
        versioned_dense: bool = False,
        staged_search: Callable[[str, int], SearchResult] | None = None,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self._model_factory = model_factory
        self._staged_search = staged_search
        self._lock = threading.Condition()
        self._jobs: deque[_Job] = deque()
        self._active: _Job | None = None
        self._model = None
        self._worker_thread: threading.Thread | None = None
        self._idle_timer: threading.Timer | None = None
        self._stopped = False
        self.load_count = 0
        if not config.database_path.expanduser().is_file():
            raise ValueError("database was not found")
        self.store = SQLiteDocumentStore(config.database_path, read_only=True)
        record = self.store.get_document(config.document_id)
        if record is None:
            raise ValueError("bound document was not found")
        self._docling_source = False
        if record.parser_name == "docling":
            from page_cite_rag.docling_document import has_native_source

            self._docling_source = has_native_source(self.store, config.document_id)
        if staged_search is None:
            if self._docling_source:
                from page_cite_rag.application import document_retriever

                self._lexical, self._dense = None, None
                self.retriever = document_retriever(
                    self.store, config.document_id, config.method, encoder=encoder
                )
            else:
                self._lexical, self._dense = document_retriever_parts(
                    self.store,
                    config.document_id,
                    config.method,
                    encoder=encoder,
                    versioned_dense=versioned_dense,
                )
            if not self._docling_source and config.method == "hybrid":
                from page_cite_rag.hybrid import HybridRetriever

                self.retriever: Retriever | None = HybridRetriever(self._lexical, self._dense)
            elif not self._docling_source:
                self.retriever = self._lexical if self._lexical is not None else self._dense
        else:
            if config.method == "bm25":
                raise ValueError("staged GPU retrieval requires dense or hybrid")
            self._lexical, self._dense = None, None
            self.retriever = None

    def start(self) -> None:
        with self._lock:
            if self._stopped:
                raise ServiceStoppedError("service cannot restart after shutdown")
            if self._worker_thread is not None:
                return
            self._worker_thread = threading.Thread(target=self._worker, name="page-cite-rag-worker")
            self._worker_thread.start()

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _release_idle_model(self) -> None:
        with self._lock:
            if self._active is not None or self._jobs or self._stopped:
                return
            model, self._model = self._model, None
        if model is not None:
            model.close()

    def _schedule_idle(self) -> None:
        with self._lock:
            self._cancel_idle_timer()
            if self._model is not None and not self._stopped:
                self._idle_timer = threading.Timer(
                    self.config.idle_seconds, self._release_idle_model
                )
                self._idle_timer.daemon = True
                self._idle_timer.start()

    def submit(
        self,
        mode: Literal["ask", "explain"],
        question: str,
        *,
        language: Language = "en",
        block_id: str | None = None,
        rewrite_query: bool = False,
    ) -> tuple[str, Future[ServiceAnswer]]:
        if mode not in ("ask", "explain") or not question.strip() or len(question) > 4000:
            raise ValueError("invalid mode or question")
        if language not in ("en", "zh-Hant", "auto"):
            raise ValueError("invalid answer language")
        if type(rewrite_query) is not bool:
            raise ValueError("query rewrite option must be boolean")
        if rewrite_query and (
            mode != "ask" or self._staged_search is not None or self.config.method == "dense"
        ):
            raise ValueError("query rewriting requires resident BM25 or hybrid ask")
        question.encode("utf-8")
        if mode == "explain":
            if block_id is None:
                raise ValueError("explanation needs a source block")
            read_verified_source(self.store, self.config.document_id, block_id)
        elif block_id is not None:
            raise ValueError("ask does not accept a source block")
        now = time.monotonic()
        job = _Job(
            request_id=uuid.uuid4().hex,
            mode=mode,
            question=question,
            language=language,
            rewrite_query=rewrite_query,
            block_id=block_id,
            created=now,
            deadline=now + self.config.request_timeout_seconds,
        )
        with self._lock:
            if self._stopped or self._worker_thread is None:
                raise ServiceStoppedError("service is not running")
            if len(self._jobs) + int(self._active is not None) >= 1 + self.config.queue_capacity:
                raise ServiceBusyError("local model queue is full")
            self._cancel_idle_timer()
            self._jobs.append(job)
            job.deadline_timer = threading.Timer(
                self.config.request_timeout_seconds,
                lambda: self.cancel(job.request_id, deadline=True),
            )
            job.deadline_timer.daemon = True
            job.deadline_timer.start()
            self._lock.notify()
        return job.request_id, job.future

    def cancel(self, request_id: str, *, deadline: bool = False) -> bool:
        with self._lock:
            for job in self._jobs:
                if job.request_id == request_id:
                    self._jobs.remove(job)
                    job.cancel_reason = "deadline" if deadline else "cancelled"
                    job.deadline_timer.cancel()
                    job.future.set_exception(self._cancel_exception(job))
                    return True
            if self._active is None or self._active.request_id != request_id:
                return False
            if self._active.future.done():
                return False
            self._active.cancel_reason = "deadline" if deadline else "cancelled"
            model, self._model = self._model, None
        if model is not None:
            model.close()
        cancel_stage = getattr(self._staged_search, "cancel", None)
        if callable(cancel_stage):
            cancel_stage()
        return True

    @staticmethod
    def _cancel_exception(job: _Job) -> Exception:
        if job.cancel_reason == "deadline":
            return ServiceDeadlineError("local request deadline exceeded")
        return ServiceStoppedError("local request was cancelled")

    def _check_job(self, job: _Job) -> None:
        if job.cancel_reason is not None:
            raise self._cancel_exception(job)
        if time.monotonic() >= job.deadline:
            job.cancel_reason = "deadline"
            raise self._cancel_exception(job)

    def _get_model(self, job: _Job) -> tuple[object, float]:
        with self._lock:
            model = self._model
        if model is not None:
            return model, 0.0
        started = time.perf_counter()
        model = self._model_factory()
        elapsed = (time.perf_counter() - started) * 1000
        with self._lock:
            if job.cancel_reason is None and not self._stopped:
                self._model = model
                self.load_count += 1
                return model, elapsed
        model.close()
        self._check_job(job)
        raise ServiceStoppedError("service stopped during model startup")

    def _execute(self, job: _Job, queue_ms: float) -> ServiceAnswer:
        self._check_job(job)
        retrieval_stage_ms = 0.0
        retriever = self.retriever
        if job.mode == "ask" and self._staged_search is not None:
            # Existing GPU guard rejects another compute PID. Stage query embedding
            # before loading the LLM; report the real reload cost, never overlap them.
            with self._lock:
                old_model, self._model = self._model, None
            if old_model is not None:
                old_model.close()
            started = time.perf_counter()
            search = self._staged_search(job.question, self.config.top_k)
            retrieval_stage_ms = (time.perf_counter() - started) * 1000
            if (
                search.document_id != self.config.document_id
                or search.query != job.question
                or search.method != self.config.method
            ):
                raise ValueError("staged retrieval scope/query changed")
            retriever = FrozenSearchRetriever(search, top_k=self.config.top_k)
        self._check_job(job)
        model, startup_ms = self._get_model(job)
        self._check_job(job)
        if job.rewrite_query:
            from page_cite_rag.query import QueryRewriter

            if self._docling_source:
                from page_cite_rag.docling_retrieval import RewritingDoclingRetriever

                retriever = RewritingDoclingRetriever(
                    self.retriever, QueryRewriter(model, self.model_config)
                )
            else:
                from page_cite_rag.query_retrieval import RewritingRetriever

                retriever = RewritingRetriever(
                    self._lexical, QueryRewriter(model, self.model_config), self._dense
                )
        workflow = document_workflow(
            self.store,
            self.config.document_id,
            retriever if job.mode == "ask" else None,
            model,
            self.model_config,
            top_k=self.config.top_k,
            neighbor_blocks=(self.config.neighbor_blocks if job.mode == "explain" else 0),
        )
        started = time.perf_counter()
        if job.mode == "ask":
            result = ask_document(workflow, job.question, language=job.language)
        else:
            result = explain_source(workflow, job.block_id, job.question, language=job.language)
        workflow_ms = (time.perf_counter() - started) * 1000
        self._check_job(job)
        observations = getattr(model, "observations", None)
        pid = observations().get("server_pid") if callable(observations) else None
        return ServiceAnswer(
            result=result,
            trace=ServiceTrace(
                request_id=job.request_id,
                queue_ms=queue_ms,
                startup_ms=startup_ms,
                retrieval_stage_ms=retrieval_stage_ms,
                workflow_ms=workflow_ms,
                generation_ms=(
                    result.generation_stats.elapsed_ms if result.generation_stats else None
                ),
                total_ms=(time.monotonic() - job.created) * 1000,
                load_count=self.load_count,
                model_pid=pid,
                retrieval_mode="staged" if self._staged_search is not None else "resident",
            ),
        )

    def _worker(self) -> None:
        while True:
            with self._lock:
                while not self._jobs and not self._stopped:
                    self._lock.wait()
                if self._stopped:
                    return
                job = self._jobs.popleft()
                self._active = job
            queue_ms = (time.monotonic() - job.created) * 1000
            try:
                reply = self._execute(job, queue_ms)
                if job.cancel_reason is not None:
                    raise self._cancel_exception(job)
                job.future.set_result(reply)
            except BaseException as exc:
                with self._lock:
                    model, self._model = self._model, None
                if model is not None:
                    model.close()
                if not job.future.done():
                    job.future.set_exception(
                        self._cancel_exception(job) if job.cancel_reason is not None else exc
                    )
            finally:
                job.deadline_timer.cancel()
                if job.deadline_timer is not threading.current_thread():
                    job.deadline_timer.join(timeout=5)
                with self._lock:
                    self._active = None
                self._schedule_idle()

    def document(self) -> dict:
        metadata = self.store.get_document(self.config.document_id)
        report = {
            "document": metadata.model_dump(mode="json"),
            "units": [
                unit.model_dump(mode="json")
                for unit in self.store.get_unit_summaries(self.config.document_id)
            ],
        }
        if self._docling_source:
            from page_cite_rag.docling_document import load_bundle

            bundle = load_bundle(self.store, self.config.document_id)
            report["coverage"] = bundle.coverage
            report["sections"] = (
                [section.model_dump(mode="json") for section in bundle.outline.sections]
                if bundle.outline is not None
                else []
            )
        return report

    def source(self, block_id: str):
        return read_verified_source(self.store, self.config.document_id, block_id)

    def source_details(self, block_id: str) -> dict:
        block = self.source(block_id)
        report = block.model_dump(mode="json")
        if self._docling_source:
            from page_cite_rag.docling_document import load_bundle

            bundle = load_bundle(self.store, self.config.document_id)
            report["native_location"] = bundle.source.location_for(
                self.config.document_id, block_id
            )
        else:
            location = self.store.get_source_location(self.config.document_id, block_id)
            if location is not None:
                report["source_location"] = location
        return report

    def blocks_page(self, *, after: int = 0, limit: int = 10) -> BlockPage:
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("invalid block page bounds")
        # Keep the frozen store implementation unchanged for historical replay.
        # Read only bounded IDs here; the shared source validator reads each block.
        uri = self.store.database_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=30)) as connection:
            candidates = connection.execute(
                "SELECT block_id FROM blocks WHERE document_id = ? AND ordinal > ? "
                "ORDER BY ordinal LIMIT ?",
                (self.config.document_id, after, limit + 1),
            ).fetchall()
        blocks = [
            read_verified_source(self.store, self.config.document_id, block_id)
            for (block_id,) in candidates[:limit]
        ]
        return BlockPage(
            blocks=[
                BlockPreview(
                    block_id=block.block_id,
                    ordinal=block.ordinal,
                    page_number=block.page_number,
                    line_start=block.line_start,
                    line_end=block.line_end,
                    preview=block.text[:240] + ("…" if len(block.text) > 240 else ""),
                )
                for block in blocks
            ],
            next_after=blocks[-1].ordinal if len(candidates) > limit else None,
        )

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._worker_thread is not None and not self._stopped,
                "active_request": self._active.request_id if self._active else None,
                "queued": len(self._jobs),
                "model_loaded": self._model is not None,
                "load_count": self.load_count,
                "document_id": self.config.document_id,
                "method": self.config.method,
            }

    def shutdown(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._cancel_idle_timer()
            for job in self._jobs:
                job.cancel_reason = "cancelled"
                job.deadline_timer.cancel()
                job.future.set_exception(self._cancel_exception(job))
            self._jobs.clear()
            active_id = self._active.request_id if self._active else None
            model, self._model = self._model, None
            self._lock.notify_all()
        if active_id is not None:
            self.cancel(active_id)
        if model is not None:
            model.close()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10)
            if self._worker_thread.is_alive():
                raise ServiceStoppedError("worker did not stop after owned model cancellation")

    def __enter__(self) -> DocumentService:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()
