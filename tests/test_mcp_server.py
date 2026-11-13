"""MCP protocol coverage uses memory streams and fake/local test models only."""

from __future__ import annotations

import asyncio
import builtins
import json
from concurrent.futures import Future
from contextlib import asynccontextmanager

import anyio
import pytest

pytest.importorskip("mcp")
from mcp.client import ClientSession  # noqa: E402
from mcp.shared.memory import create_client_server_memory_streams  # noqa: E402

from page_cite_rag.api import _failure  # noqa: E402
from page_cite_rag.generation import ModelError  # noqa: E402
from page_cite_rag.mcp_server import PRIVACY_NOTICE, create_server  # noqa: E402
from page_cite_rag.retrieval import RetrievalError  # noqa: E402
from page_cite_rag.service import (  # noqa: E402
    ServiceBusyError,
    ServiceDeadlineError,
    ServiceStoppedError,
)
from page_cite_rag.workflows import WorkflowError  # noqa: E402
from tests import test_service  # noqa: E402

prepared = test_service.prepared
make_service = test_service.make_service


@asynccontextmanager
async def connected(service):
    server = create_server(service)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(server.run, *server_streams, server.create_initialization_options())
            async with ClientSession(*client_streams, read_timeout_seconds=30) as client:
                await client.initialize()
                yield client
            tasks.cancel_scope.cancel()


def test_all_tools_protocol_and_service_outputs(prepared):
    service, clients = make_service(prepared)
    block_id = prepared[2][0].block_id

    async def check():
        async with connected(service) as client:
            tools = (await client.list_tools()).tools
            assert {tool.name for tool in tools} == {
                "document_info",
                "list_passages",
                "read_passage",
                "ask",
                "explain",
            }
            for tool in tools:
                assert tool.description.startswith(PRIVACY_NOTICE)
                assert "rewrite_query" not in tool.input_schema.get("properties", {})
            for name, arguments, expected in [
                ("document_info", {}, service.document()),
                ("list_passages", {}, service.blocks_page().model_dump(mode="json")),
                ("read_passage", {"block_id": block_id}, service.source_details(block_id)),
            ]:
                result = await client.call_tool(name, arguments)
                assert not result.is_error
                assert result.structured_content == expected
                assert json.loads(result.content[0].text) == expected
            assert clients == []  # Navigation never loads a model.
            for name, arguments in [
                ("ask", {"question": "result significant", "language": "en"}),
                ("explain", {"block_id": block_id, "question": "Explain this source."}),
            ]:
                result = await client.call_tool(name, arguments)
                assert not result.is_error
                output = result.structured_content
                assert output["result"]["answer"]
                citation = output["result"]["answer_claims"][0]["citations"][0]
                assert citation["quote"] in service.source(citation["block_id"]).text
                assert {
                    "block_id",
                    "quote",
                    "page_number",
                    "line_start",
                    "line_end",
                } <= citation.keys()
                assert {"request_id", "reading_trace", "retrieval_mode"} <= output["trace"].keys()
                assert json.loads(result.content[0].text) == output
            empty = await client.call_tool("list_passages", {"page": 100000})
            assert empty.structured_content == {"blocks": [], "next_after": None}

    asyncio.run(check())
    assert not service.status()["running"]
    assert clients[0].close_count == 1


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("ask", {}),
        ("ask", {"question": ""}),
        ("ask", {"question": "x" * 4001}),
        ("ask", {"question": 123}),
        ("ask", {"question": "ok", "language": "zh"}),
        ("ask", {"question": "ok", "rewrite_query": True}),
        ("explain", {"question": "ok", "block_id": "../../secret"}),
        ("read_passage", {"block_id": "A" * 64}),
        ("read_passage", {"block_id": 1}),
        ("list_passages", {"page": 0}),
        ("list_passages", {"page": True}),
        ("list_passages", {"page": "1"}),
        ("list_passages", {"limit": 51}),
        ("document_info", {"unexpected": "/private/path"}),
    ],
)
def test_invalid_arguments_are_safe_tool_errors(prepared, name, arguments):
    service, clients = make_service(prepared)

    async def check():
        async with connected(service) as client:
            result = await client.call_tool(name, arguments)
            assert result.is_error
            assert result.content[0].text == "invalid request or unavailable index"

    asyncio.run(check())
    assert clients == []


@pytest.mark.parametrize(
    "error_type",
    [
        ServiceBusyError,
        ServiceDeadlineError,
        ServiceStoppedError,
        WorkflowError,
        ValueError,
        RetrievalError,
        ModelError,
        RuntimeError,
    ],
)
@pytest.mark.parametrize(
    "tool", ["document_info", "list_passages", "read_passage", "ask", "explain"]
)
def test_service_errors_are_safe(prepared, monkeypatch, error_type, tool):
    service, _ = make_service(prepared)
    error = error_type("secret /private/path")

    def fail(*args, **kwargs):
        raise error

    def failed_future(*args, **kwargs):
        future = Future()
        future.set_exception(error)
        return "request", future

    for method in ("document", "blocks_page", "source_details"):
        monkeypatch.setattr(service, method, fail)
    monkeypatch.setattr(service, "submit", failed_future)

    async def check():
        async with connected(service) as client:
            arguments = {
                "document_info": {},
                "list_passages": {},
                "read_passage": {"block_id": "a" * 64},
                "ask": {"question": "ok"},
                "explain": {"question": "ok", "block_id": "a" * 64},
            }[tool]
            result = await client.call_tool(tool, arguments)
            assert result.is_error
            assert result.content[0].text == _failure(error).detail
            assert "secret" not in result.content[0].text

    asyncio.run(check())


def test_passage_pages_follow_api_cursor(prepared, monkeypatch):
    from page_cite_rag.service import BlockPage

    service, _ = make_service(prepared)
    calls = []

    def page(*, after, limit):
        calls.append((after, limit))
        return BlockPage(blocks=[], next_after=30 if after == 0 else None)

    monkeypatch.setattr(service, "blocks_page", page)

    async def check():
        async with connected(service) as client:
            result = await client.call_tool("list_passages", {"page": 2})
            assert not result.is_error

    asyncio.run(check())
    assert calls == [(0, 10), (30, 10)]


def test_mcp_cli_shares_options_and_runtime(monkeypatch):
    from page_cite_rag import cli, mcp_server

    parser = cli.build_parser()
    serve = vars(parser.parse_args(["serve", "a" * 64]))
    mcp = vars(parser.parse_args(["mcp", "a" * 64]))
    serve.pop("command")
    serve.pop("port")  # Only the HTTP transport has a port.
    mcp.pop("command")
    assert mcp == serve
    subparsers = next(a for a in parser._actions if a.dest == "command")
    serve_actions = {tuple(a.option_strings): a for a in subparsers.choices["serve"]._actions}
    for action in subparsers.choices["mcp"]._actions:
        other = serve_actions[tuple(action.option_strings)]
        assert (action.type, action.choices, action.required, action.nargs) == (
            other.type,
            other.choices,
            other.required,
            other.nargs,
        )
    captured = []
    runtime = object()
    monkeypatch.setattr(cli, "_create_document_service", lambda args: runtime)
    monkeypatch.setattr(mcp_server, "run_stdio", captured.append)
    monkeypatch.delenv("PAGE_CITE_RAG_API_TOKEN", raising=False)
    assert cli.main(["mcp", "a" * 64]) == 0
    assert captured == [runtime]


def test_missing_mcp_dependency_is_actionable(monkeypatch, capsys):
    from page_cite_rag import cli

    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "page_cite_rag.mcp_server":
            raise ImportError("secret path")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert cli.main(["mcp", "a" * 64]) == 1
    assert "uv sync --extra mcp" in capsys.readouterr().err


def test_cancellation_reaches_owned_service_request(prepared, monkeypatch):
    service, _ = make_service(prepared)
    future = Future()
    cancelled = []
    monkeypatch.setattr(service, "submit", lambda *args, **kwargs: ("owned-request", future))
    monkeypatch.setattr(service, "cancel", cancelled.append)

    async def check():
        from mcp import types

        server = create_server(service)
        handler = server.get_request_handler("tools/call").handler
        task = asyncio.create_task(
            handler(None, types.CallToolRequestParams(name="ask", arguments={"question": "ok"}))
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not future.cancelled()

    asyncio.run(check())
    assert cancelled == ["owned-request"]


def test_submit_failure_is_a_safe_tool_error(prepared, monkeypatch):
    service, _ = make_service(prepared)

    def busy(*args, **kwargs):
        raise ServiceBusyError("secret path")

    monkeypatch.setattr(service, "submit", busy)

    async def check():
        async with connected(service) as client:
            result = await client.call_tool("ask", {"question": "ok"})
            assert result.is_error
            assert result.content[0].text == "local model queue is full"

    asyncio.run(check())
