"""Browser reader for the local API: static page assets and verified PDF page images.

The page itself holds no document data, so it is served without the API token; every
request it makes for answers, sources and page images carries the token like any other
client. Page images come only from the original PDF whose SHA-256 equals the ingested
document's, so a highlighted region always refers to the file that was parsed.
"""

from __future__ import annotations

import hashlib
import io
import threading
from importlib import resources
from pathlib import Path

ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
# Same-origin only: no inline script or style, no third-party loads, page images as blobs.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
# 2x the PDF's 72 points per inch: sharp on high-density screens, about 0.3-0.6 MB per page.
RENDER_SCALE = 2.0
MAX_CACHED_PAGES = 8


def asset(path: str) -> tuple[bytes, str] | None:
    """Return the packaged asset for a public path, or None for anything else."""
    entry = ASSETS.get(path)
    if entry is None:
        return None
    name, media_type = entry
    return resources.files("page_cite_rag").joinpath("webui", name).read_bytes(), media_type


class PdfPages:
    """Render pages of one verified PDF to PNG; pdfium is not thread-safe, so one lock."""

    def __init__(self, path: Path) -> None:
        import pypdfium2

        self._document = pypdfium2.PdfDocument(str(path))
        self.page_count = len(self._document)
        self._lock = threading.Lock()
        self._cache: dict[int, tuple[bytes, float, float]] = {}

    def render(self, page_number: int) -> tuple[bytes, float, float]:
        """PNG bytes and the page size in PDF points for a 1-based page number."""
        if not 1 <= page_number <= self.page_count:
            raise IndexError("page outside the bound PDF")
        with self._lock:
            cached = self._cache.get(page_number)
            if cached is not None:
                return cached
            page = self._document[page_number - 1]
            try:
                width, height = page.get_size()
                image = page.render(scale=RENDER_SCALE).to_pil()
            finally:
                page.close()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            rendered = (buffer.getvalue(), float(width), float(height))
            if len(self._cache) >= MAX_CACHED_PAGES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[page_number] = rendered
            return rendered


def verified_pdf(path: str | Path, expected_sha256: str) -> PdfPages:
    """Open the PDF only if it is byte-for-byte the file that was ingested."""
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError("--pdf does not match the ingested document (SHA-256 differs)")
    import pypdfium2

    try:
        return PdfPages(path)
    except pypdfium2.PdfiumError:
        raise ValueError("--pdf is not a readable PDF") from None
