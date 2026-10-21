"""Loopback-only, single-document FastAPI adapter over the shared application core."""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from page_cite_rag.generation import ModelError
from page_cite_rag.retrieval import RetrievalError
from page_cite_rag.service import (
    BlockPage,
    DocumentService,
    ServiceAnswer,
    ServiceBusyError,
    ServiceDeadlineError,
    ServiceStoppedError,
)
from page_cite_rag.workflows import WorkflowError


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=4000)
    language: Literal["en", "zh-Hant", "auto"] = "en"


class AskRequest(QuestionRequest):
    rewrite_query: bool = False


class ExplainRequest(QuestionRequest):
    block_id: str = Field(pattern=r"^[0-9a-f]{64}$")


def _failure(error: Exception) -> HTTPException:
    if isinstance(error, ServiceBusyError):
        return HTTPException(429, "local model queue is full")
    if isinstance(error, ServiceDeadlineError):
        return HTTPException(504, "local request deadline exceeded")
    if isinstance(error, ServiceStoppedError):
        return HTTPException(503, "local service is stopped or request was cancelled")
    if isinstance(error, WorkflowError):
        return HTTPException(404, "source block is outside the bound document or missing")
    if isinstance(error, (ValueError, RetrievalError)):
        return HTTPException(422, "invalid request or unavailable index")
    if isinstance(error, ModelError):
        return HTTPException(502, "local model failed; no fallback or retry")
    return HTTPException(500, "local request failed")


async def _answer(service: DocumentService, request: Request, mode: str, payload) -> ServiceAnswer:
    try:
        request_id, future = service.submit(
            mode,
            payload.question,
            language=payload.language,
            block_id=getattr(payload, "block_id", None),
            rewrite_query=getattr(payload, "rewrite_query", False),
        )
    except Exception as error:
        raise _failure(error) from None
    wrapped = asyncio.wrap_future(future)
    try:
        while True:
            completed, _ = await asyncio.wait({wrapped}, timeout=0.2)
            if completed:
                return wrapped.result()
            if await request.is_disconnected():
                service.cancel(request_id)
                raise HTTPException(499, "client disconnected; local request cancelled")
    except asyncio.CancelledError:
        service.cancel(request_id)
        raise
    except HTTPException:
        raise
    except Exception as error:
        raise _failure(error) from None


def create_app(
    service: DocumentService, *, api_token: str, allow_test_client: bool = False
) -> FastAPI:
    """No import-time model load; require a caller-supplied token and loopback client."""
    if not isinstance(api_token, str) or not api_token.strip():
        raise ValueError("a nonempty local API token is required")
    expected_authorization = f"Bearer {api_token}".encode("utf-8")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(title="PageCiteRAG local document reader", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def local_client_only(request: Request, call_next):
        allowed = {"127.0.0.1", "::1"}
        if allow_test_client:
            allowed.add("testclient")
        if request.client is None or request.client.host not in allowed:
            return JSONResponse(status_code=403, content={"detail": "loopback clients only"})
        supplied_authorization = request.headers.get("authorization", "").encode("utf-8")
        if not hmac.compare_digest(supplied_authorization, expected_authorization):
            return JSONResponse(status_code=401, content={"detail": "bearer token required"})
        return await call_next(request)

    @app.get("/health")
    async def health():
        return service.status()

    @app.get("/v1/document")
    async def document():
        return service.document()

    @app.get("/v1/blocks", response_model=BlockPage)
    async def blocks(
        after: int = Query(default=0, ge=0), limit: int = Query(default=10, ge=1, le=50)
    ):
        try:
            return service.blocks_page(after=after, limit=limit)
        except Exception as error:
            raise _failure(error) from None

    @app.get("/v1/source/{block_id}")
    async def source(block_id: str):
        try:
            return service.source_details(block_id)
        except Exception as error:
            raise _failure(error) from None

    @app.post("/v1/ask", response_model=ServiceAnswer)
    async def ask(request: Request, payload: AskRequest):
        return await _answer(service, request, "ask", payload)

    @app.post("/v1/explain", response_model=ServiceAnswer)
    async def explain(request: Request, payload: ExplainRequest):
        return await _answer(service, request, "explain", payload)

    return app
