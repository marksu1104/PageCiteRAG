"""Browser reader: public static page, token-protected data and page images, verified PDF."""

from __future__ import annotations

import asyncio
import hashlib

import httpx
import pypdfium2
import pytest

from page_cite_rag import web
from page_cite_rag.api import create_app
from tests.test_service import make_service, prepared  # noqa: F401  (pytest fixture)

TOKEN = "test-local-token"


def make_pdf(path, sizes=((612, 792), (300, 400))):
    document = pypdfium2.PdfDocument.new()
    for width, height in sizes:
        document.new_page(width, height)
    document.save(path)
    document.close()
    return path


def run_client(app, check):
    async def main():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as anon:
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://127.0.0.1",
                    headers={"Authorization": f"Bearer {TOKEN}"},
                ) as client:
                    await check(anon, client)

    asyncio.run(main())


def test_reader_page_is_public_but_every_data_route_still_needs_the_token(prepared):  # noqa: F811
    service, _ = make_service(prepared)
    app = create_app(service, api_token=TOKEN, allow_test_client=True)

    async def check(anon, client):
        page = await anon.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "script-src 'self'" in page.headers["content-security-policy"]
        assert page.headers["x-content-type-options"] == "nosniff"
        assert '<script src="/app.js"' in page.text
        script = await anon.get("/app.js")
        assert script.headers["content-type"].startswith("text/javascript")
        assert (await anon.get("/app.css")).headers["content-type"].startswith("text/css")
        assert (await anon.get("/favicon.svg")).headers["content-type"] == "image/svg+xml"
        for path in ("/health", "/v1/document", "/v1/pages/1", "/app.js.map", "/index.html"):
            assert (await anon.get(path)).status_code == 401, path
        assert (await anon.post("/", json={})).status_code == 401
        missing = await client.get("/v1/pages/1")
        assert missing.status_code == 404 and "--pdf" in missing.json()["detail"]

    run_client(app, check)


def test_page_images_come_from_the_bound_pdf_with_their_point_size(prepared, tmp_path):  # noqa: F811
    service, _ = make_service(prepared)
    pdf = web.PdfPages(make_pdf(tmp_path / "two.pdf"))
    app = create_app(service, api_token=TOKEN, allow_test_client=True, pdf=pdf)

    async def check(anon, client):
        assert (await anon.get("/v1/pages/2")).status_code == 401
        second = await client.get("/v1/pages/2")
        assert second.status_code == 200
        assert second.headers["content-type"] == "image/png"
        assert second.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert second.headers["x-page-width-points"] == "300.00"
        assert second.headers["x-page-height-points"] == "400.00"
        assert second.headers["cache-control"] == "no-store"
        assert (await client.get("/v1/pages/3")).status_code == 404
        assert (await client.get("/v1/pages/0")).status_code == 422

    run_client(app, check)
    assert pdf.render(2) is pdf.render(2)  # cached after the first render


def test_ask_accepts_the_same_per_request_mode_as_the_mcp_tool(prepared):  # noqa: F811
    service, _ = make_service(prepared)
    app = create_app(service, api_token=TOKEN, allow_test_client=True)

    async def check(_anon, client):
        default = await client.post("/v1/ask", json={"question": "result significant"})
        automatic = await client.post(
            "/v1/ask", json={"question": "result significant", "mode": "auto"}
        )
        assert default.status_code == automatic.status_code == 200
        assert default.json()["trace"]["reading_trace"]["requested_mode"] == "detail"
        assert automatic.json()["trace"]["reading_trace"]["requested_mode"] == "auto"
        invalid = await client.post("/v1/ask", json={"question": "a", "mode": "fixed"})
        assert invalid.status_code == 422

    run_client(app, check)


def test_verified_pdf_rejects_a_different_or_unreadable_file(tmp_path):
    pdf = make_pdf(tmp_path / "paper.pdf")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert web.verified_pdf(pdf, digest).page_count == 2
    with pytest.raises(ValueError, match="does not match the ingested document"):
        web.verified_pdf(pdf, "0" * 64)
    text = tmp_path / "notes.txt"
    text.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(ValueError, match="not a readable PDF"):
        web.verified_pdf(text, hashlib.sha256(text.read_bytes()).hexdigest())


def test_serve_binds_the_pdf_only_when_it_is_the_ingested_file(tmp_path, monkeypatch, capsys):
    import uvicorn

    from page_cite_rag import api, cli

    pdf = make_pdf(tmp_path / "paper.pdf")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    captured = {}

    class Runtime:
        def __init__(self, source_sha256):
            self.source_sha256 = source_sha256

        def document(self):
            return {"document": {"source_sha256": self.source_sha256}}

    runtime = Runtime(digest)
    monkeypatch.setenv("PAGE_CITE_RAG_API_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_create_document_service", lambda _args: runtime)
    monkeypatch.setattr(api, "create_app", lambda service, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)

    assert cli.main(["serve", "a" * 64, "--pdf", str(pdf)]) == 0
    assert captured["pdf"].page_count == 2
    assert "reader at http://127.0.0.1:8765/" in capsys.readouterr().err

    runtime.source_sha256 = "0" * 64
    captured.clear()
    assert cli.main(["serve", "a" * 64, "--pdf", str(pdf)]) == 1
    error = capsys.readouterr().err
    assert "does not match the ingested document" in error and str(tmp_path) not in error
    assert captured == {}

    assert cli.main(["serve", "a" * 64]) == 0
    assert captured["pdf"] is None


def test_reader_inserts_document_text_as_text_only():
    script = web.asset("/app.js")[0].decode("utf-8")
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script
    assert "eval(" not in script
