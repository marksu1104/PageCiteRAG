"""Single-document MCP stdio adapter; all document work stays in DocumentService."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, ConfigDict, Field

from page_cite_rag.api import ExplainRequest, QuestionRequest, _failure
from page_cite_rag.application import Mode
from page_cite_rag.service import BlockPage, DocumentService

PRIVACY_NOTICE = (
    "Returns original document text; if your MCP client sends it to a cloud model, the "
    "document leaves this machine. "
)


class _EmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _PageRequest(_EmptyRequest):
    page: int = Field(default=1, ge=1)


class _SourceRequest(_EmptyRequest):
    block_id: str = ExplainRequest.model_fields["block_id"]


class _AskRequest(QuestionRequest):
    mode: Mode | None = None


_TOOL_REQUESTS = {
    "document_info": (
        _EmptyRequest,
        "Metadata of the one document this server serves (source name, language, blocks).",
    ),
    "list_passages": (
        _PageRequest,
        "List the document's passages in reading order, 10 per page, with block IDs.",
    ),
    "read_passage": (
        _SourceRequest,
        "Original text and page/line location of one passage, by block ID.",
    ),
    "ask": (
        _AskRequest,
        "Answer a question about the document; returns the answer, the cited original "
        "quotes with block IDs and page/line locations, and the mode used.",
    ),
    "explain": (
        ExplainRequest,
        "Explain one passage (by block ID) for a question, citing the original text.",
    ),
}


def create_server(service: DocumentService) -> Server:
    """Bind exactly one existing service; no model loads during construction."""

    @asynccontextmanager
    async def lifespan(_server):
        service.start()
        try:
            yield {}
        finally:
            service.shutdown()

    async def list_tools(_context, _params):
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=name,
                    description=PRIVACY_NOTICE + description,
                    input_schema=request.model_json_schema(),
                )
                for name, (request, description) in _TOOL_REQUESTS.items()
            ]
        )

    async def call_tool(_context, params):
        try:
            if params.name not in _TOOL_REQUESTS:
                raise ValueError("unknown tool")
            request = _TOOL_REQUESTS[params.name][0].model_validate(params.arguments or {})
            if params.name == "document_info":
                result = service.document()
            elif params.name == "list_passages":
                # Follow the API's cursor rather than assuming contiguous ordinals.
                after = 0
                for page in range(1, request.page + 1):
                    result = service.blocks_page(after=after, limit=10)
                    if page == request.page:
                        break
                    if result.next_after is None:
                        result = BlockPage(blocks=[], next_after=None)
                        break
                    after = result.next_after
            elif params.name == "read_passage":
                result = service.source_details(request.block_id)
            else:
                request_id, future = service.submit(
                    params.name,
                    request.question,
                    language=request.language,
                    block_id=getattr(request, "block_id", None),
                    rewrite_query=False,
                    **({"mode": request.mode} if params.name == "ask" else {}),
                )
                try:
                    wrapped = asyncio.wrap_future(future)
                    while True:
                        completed, _ = await asyncio.wait({wrapped}, timeout=0.2)
                        if completed:
                            result = wrapped.result()
                            break
                except asyncio.CancelledError:
                    service.cancel(request_id)
                    raise
            if isinstance(result, BaseModel):
                result = result.model_dump(mode="json")
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))
                ],
                structured_content=result,
            )
        except Exception as error:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=_failure(error).detail)],
                is_error=True,
            )

    return Server(
        "PageCiteRAG",
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def run_stdio(service: DocumentService) -> None:
    """Run the official stdio transport; stdout is reserved for MCP messages."""
    server = create_server(service)

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())
