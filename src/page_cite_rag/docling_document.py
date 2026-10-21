"""Persist Docling's native PDF sources through the existing SQLite document store.

The SQLite units are virtual native items or cells. Their character offsets refer to
the saved native text, while PDF page and bbox provenance is stored separately.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from page_cite_rag.docling_adapter import ADAPTER_VERSION, DoclingSourceStore
from page_cite_rag.docling_source import read_native_item
from page_cite_rag.embedding import DEFAULT_MODEL_DIR
from page_cite_rag.models import Block, DocumentMetadata, ParsedUnit
from page_cite_rag.source_outline import OUTLINE_KIND, SourceOutline, SourceSection
from page_cite_rag.store import SQLiteDocumentStore, StoreError

SOURCE_KIND = "docling-native-document-v1"
_SOURCE_VERSION = "docling-native-document/v1"
DEFAULT_ARTIFACTS = Path("models/docling-pinned")
_MODEL_HASHES = {
    "00333a43451945aaf89db8ca9c0a17e75d1537c17db60fdb91aa95f4c7929e0c",
    "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9",
}
_MODEL_CONFIG_HASHES = {
    "docling-project--docling-layout-heron/config.json": "fdea30805ce2f5666b147fca941dcdd27ad468e27d6ed21902207d3da056a97d",  # noqa: E501
    "docling-project--docling-layout-heron/preprocessor_config.json": "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",  # noqa: E501
    "docling-project--docling-models/model_artifacts/tableformer/accurate/tm_config.json": "984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd",  # noqa: E501
}


class DoclingDocumentError(ValueError):
    """A Docling source, local dependency, or saved source contract is invalid."""


def has_native_source(store: SQLiteDocumentStore, document_id: str) -> bool:
    """Distinguish the product's saved Docling document from a Docling parser adapter."""
    return store.get_retrieval_manifest(document_id, SOURCE_KIND) is not None


@dataclass(frozen=True)
class DoclingDocumentBundle:
    metadata: DocumentMetadata
    source: DoclingSourceStore
    blocks: tuple[Block, ...]
    document: dict
    chunks: list[dict]
    coverage: dict
    outline: SourceOutline | None

    def source_manifests(self) -> dict[str, str]:
        payload = {
            "version": _SOURCE_VERSION,
            "source_sha256": self.metadata.source_sha256,
            "source_name": self.metadata.source_name,
            "page_count": self.metadata.page_count,
            "parser_version": self.metadata.parser_version,
            "document": self.document,
            "chunks": self.chunks,
            "coverage": self.coverage,
        }
        manifests = {SOURCE_KIND: json.dumps(payload, ensure_ascii=False, sort_keys=True)}
        if self.outline is not None:
            manifests[OUTLINE_KIND] = self.outline.model_dump_json()
        return manifests

    def units(self) -> list[ParsedUnit]:
        units = [
            self.source.get_unit(self.metadata.document_id, block.unit_index)
            for block in self.blocks
        ]
        if any(unit is None for unit in units):
            raise DoclingDocumentError("native source unit is missing")
        return units

    def block_factory(self, _document: DocumentMetadata, unit: ParsedUnit, ordinal: int):
        block = self.blocks[ordinal - 1]
        if block.unit_index != unit.unit_index:
            raise DoclingDocumentError("native source unit order changed")
        yield block


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outline(source: DoclingSourceStore, chunks: list[dict], metadata: DocumentMetadata):
    blocks = source.get_blocks(metadata.document_id)
    if not blocks:
        return None
    titles: dict[str, str | None] = {}
    for index, entry in enumerate(chunks):
        headings = entry.get("chunk", {}).get("meta", {}).get("headings") or []
        title = headings[-1].strip()[:512] if headings and isinstance(headings[-1], str) else None
        for block_id in source.chunk_block_ids[index]:
            titles.setdefault(block_id, title)
    groups: list[tuple[str | None, int, int]] = []
    for block in blocks:
        title = titles.get(block.block_id)
        if groups and groups[-1][0] == title:
            groups[-1] = (title, groups[-1][1], block.ordinal)
        else:
            groups.append((title, block.ordinal, block.ordinal))
    return SourceOutline(
        document_id=metadata.document_id,
        source_sha256=metadata.source_sha256,
        origin="docling_headings",
        block_count=len(blocks),
        sections=tuple(
            SourceSection(
                section_id=f"section:{index:04d}",
                title=title,
                start_ordinal=start,
                end_ordinal=end,
            )
            for index, (title, start, end) in enumerate(groups)
        ),
    )


def bundle_from_native(
    document: dict,
    chunks: list[dict],
    *,
    source_sha256: str,
    source_name: str,
    page_count: int,
    parser_version: str,
    language: str | None = None,
    usage_scope: str = "not-recorded",
) -> DoclingDocumentBundle:
    """Validate a native capture before inserting it; no PDF bytes are needed on reload."""
    if not isinstance(parser_version, str) or not parser_version:
        raise DoclingDocumentError("Docling parser version is missing")
    for entry in chunks:
        if not isinstance(entry, dict) or entry.get("profile", parser_version) != parser_version:
            raise DoclingDocumentError("Docling chunk parser profile changed")
    chunks = [{**entry, "profile": parser_version} for entry in chunks]
    source = DoclingSourceStore(
        document,
        chunks,
        source_sha256=source_sha256,
        source_name=source_name,
        page_count=page_count,
    )
    if any(not isinstance(entry.get("search_text"), str) for entry in chunks):
        raise DoclingDocumentError("Docling chunk search text is missing")
    covered = {ref for refs in source.chunk_native_refs for ref in refs}
    missing: list[str] = []
    unsupported: list[str] = []
    for collection in ("texts", "tables", "pictures", "key_value_items", "form_items"):
        for item in document.get(collection, []):
            ref = item.get("self_ref") if isinstance(item, dict) else None
            if not isinstance(ref, str):
                raise DoclingDocumentError("native item identity is missing")
            native = read_native_item(document, ref, page_count=page_count)
            if native["saved_item_text"] is None and native["table"] is None:
                unsupported.append(ref)
            elif ref not in covered:
                missing.append(ref)
    if missing:
        labels: dict[str, int] = {}
        for ref in missing:
            item = read_native_item(document, ref, page_count=page_count)
            label = str(item["kind"])
            labels[label] = labels.get(label, 0) + 1
        raise DoclingDocumentError(
            f"Docling chunks omitted {len(missing)} readable native items: "
            f"{', '.join(missing[:5])}; kinds={labels}"
        )
    metadata = DocumentMetadata(
        document_id=source.document_id,
        source_sha256=source_sha256,
        source_name=source_name,
        media_type="application/pdf",
        parser_name="docling",
        parser_version=parser_version,
        language=language,
        usage_scope=usage_scope,
        page_count=page_count,
    )
    coverage = {
        "readable_source_units": len(source.get_blocks(metadata.document_id)),
        "unsupported_native_refs": unsupported,
        "chunk_count": len(chunks),
        "recovered_search_windows": sum(
            entry.get("origin") == "native-item-recovery-v1" for entry in chunks
        ),
    }
    return DoclingDocumentBundle(
        metadata=metadata,
        source=source,
        blocks=tuple(source.get_blocks(metadata.document_id)),
        document=document,
        chunks=chunks,
        coverage=coverage,
        outline=_outline(source, chunks, metadata),
    )


def _check_local_models(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise DoclingDocumentError("provide a prepared local Docling model directory")
    hashes: dict[str, str] = {}
    for path in directory.rglob("*.safetensors"):
        if path.is_symlink():
            raise DoclingDocumentError("Docling model artifacts may not be symlinks")
        hashes[str(path.relative_to(directory))] = _sha256_file(path)
    if not _MODEL_HASHES.issubset(hashes.values()):
        raise DoclingDocumentError("prepared Docling layout and table model hashes are missing")
    for name, expected in _MODEL_CONFIG_HASHES.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise DoclingDocumentError("prepared Docling model configuration changed")
    return hashes


def prepare_models(directory: Path = DEFAULT_ARTIFACTS) -> dict:
    """Explicitly fetch only pinned public Docling layout and table assets."""
    from huggingface_hub import snapshot_download

    if directory.is_symlink():
        raise DoclingDocumentError("Docling model destination may not be a symlink")
    destinations = (
        (
            "docling-project/docling-layout-heron",
            "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
            ("config.json", "preprocessor_config.json", "model.safetensors"),
        ),
        (
            "docling-project/docling-models",
            "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
            (
                "model_artifacts/tableformer/accurate/tm_config.json",
                "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors",
            ),
        ),
    )
    for repository, revision, files in destinations:
        target = directory / repository.replace("/", "--")
        if target.is_symlink():
            raise DoclingDocumentError("Docling model destination may not be a symlink")
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=target,
            allow_patterns=list(files),
            max_workers=2,
        )
    hashes = _check_local_models(directory)
    return {"artifacts_path": str(directory), "weight_sha256": hashes}


def _recover_omitted_items(
    document: dict, chunks: list[dict], *, page_count: int, tokenizer
) -> int:
    """Keep native readable items searchable when HybridChunker skips them.

    These windows navigate to the complete saved native item/cell; their decoded
    search text is never used as a quotation or claimed to be PDF character offsets.
    """
    covered = {item["self_ref"] for entry in chunks for item in entry["chunk"]["meta"]["doc_items"]}
    recovered = 0
    for collection in ("texts", "tables", "key_value_items", "form_items"):
        for item in document.get(collection, []):
            ref = item["self_ref"]
            if ref in covered:
                continue
            native = read_native_item(document, ref, page_count=page_count)
            source_text = native["saved_item_text"]
            if source_text is None and native["table"] is not None:
                source_text = "\n".join(
                    cell["text"] for cell in native["table"]["cells"] if cell["text"].strip()
                )
            if not source_text or not source_text.strip():
                continue
            tokens = tokenizer.encode(source_text, add_special_tokens=False)
            for start in range(0, len(tokens), 400):
                window = tokenizer.decode(tokens[start : start + 400]).strip()
                if not window:
                    continue
                chunks.append(
                    {
                        "chunk": {
                            "text": window,
                            "meta": {"doc_items": [{"self_ref": ref}], "headings": []},
                        },
                        "search_text": window,
                        "search_tokens": len(tokens[start : start + 400]),
                        "origin": "native-item-recovery-v1",
                    }
                )
                recovered += 1
            covered.add(ref)
    return recovered


def convert_pdf(
    path: Path,
    *,
    artifacts: Path,
    tokenizer_dir: Path = DEFAULT_MODEL_DIR,
    language: str | None = None,
    usage_scope: str = "not-recorded",
) -> DoclingDocumentBundle:
    """Convert an arbitrary local text PDF with pinned, offline Docling components."""
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise DoclingDocumentError("provide a local PDF file")
    _check_local_models(artifacts)
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    from page_cite_rag.embedding import MODEL_FILES

    tokenizer_sha256 = MODEL_FILES["tokenizer.json"][1]
    if (
        tokenizer_dir.is_symlink()
        or not tokenizer_file.is_file()
        or _sha256_file(tokenizer_file) != tokenizer_sha256
    ):
        raise DoclingDocumentError("prepared local Qwen tokenizer is missing")
    source_sha256 = _sha256_file(path)
    try:
        versions = {name: version(name) for name in ("docling", "docling-core", "transformers")}
    except PackageNotFoundError as exc:
        raise DoclingDocumentError("install the optional Docling parser environment") from exc
    if versions != {"docling": "2.137.0", "docling-core": "2.101.1", "transformers": "5.19.0"}:
        raise DoclingDocumentError(
            "Docling parser dependency versions differ from the pinned profile"
        )

    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DOCLING_DEVICE": "cpu",
        "DOCLING_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ[key] = value
    from docling.chunking import HybridChunker
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    options = PdfPipelineOptions(
        artifacts_path=artifacts,
        do_ocr=False,
        do_table_structure=True,
        enable_remote_services=False,
        allow_external_plugins=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        accelerator_options=AcceleratorOptions(num_threads=2, device=AcceleratorDevice.CPU),
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )
    result = converter.convert(path)
    if result.status != ConversionStatus.SUCCESS:
        raise DoclingDocumentError("Docling could not completely parse this PDF")
    page_count = len(result.document.pages)
    if page_count < 1 or _sha256_file(path) != source_sha256:
        raise DoclingDocumentError("PDF changed during conversion or has no pages")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=496),
        merge_peers=True,
        repeat_table_header=True,
    )
    chunks = []
    for chunk in chunker.chunk(result.document):
        search_text = chunker.contextualize(chunk)
        token_count = len(tokenizer.encode(search_text, add_special_tokens=False))
        if token_count > 512:
            raise DoclingDocumentError("Docling search chunk exceeds 512 tokens")
        chunks.append(
            {
                "chunk": chunk.model_dump(mode="json", by_alias=True),
                "search_text": search_text,
                "search_tokens": token_count,
            }
        )
    document = result.document.export_to_dict()
    _recover_omitted_items(document, chunks, page_count=page_count, tokenizer=tokenizer)
    if not chunks:
        raise DoclingDocumentError("Docling found no readable chunks")
    parser_version = (
        f"docling-{versions['docling']};core-{versions['docling-core']};"
        f"hybrid-496;tok-{tokenizer_sha256[:16]};{ADAPTER_VERSION}"
    )
    return bundle_from_native(
        document,
        chunks,
        source_sha256=source_sha256,
        source_name=path.name,
        page_count=page_count,
        parser_version=parser_version,
        language=language,
        usage_scope=usage_scope,
    )


def load_bundle(store: SQLiteDocumentStore, document_id: str) -> DoclingDocumentBundle:
    record = store.get_document(document_id)
    if record is None or record.parser_name != "docling":
        raise DoclingDocumentError("the requested document is not a Docling source")
    raw = store.get_retrieval_manifest(document_id, SOURCE_KIND)
    if raw is None:
        raise StoreError("saved Docling native source is missing")
    payload = json.loads(raw)
    if payload.get("version") != _SOURCE_VERSION:
        raise StoreError("saved Docling source version is unsupported")
    bundle = bundle_from_native(
        payload["document"],
        payload["chunks"],
        source_sha256=record.source_sha256,
        source_name=record.source_name,
        page_count=record.page_count,
        parser_version=record.parser_version,
        language=record.language,
        usage_scope=record.usage_scope,
    )
    recorded_coverage = payload.get("coverage")
    if (
        bundle.metadata.document_id != record.document_id
        or not isinstance(recorded_coverage, dict)
        or any(bundle.coverage.get(key) != value for key, value in recorded_coverage.items())
    ):
        raise StoreError("saved Docling source identity or coverage changed")
    saved = store.get_blocks(document_id)
    if saved != bundle.source.get_blocks(document_id):
        raise StoreError("saved Docling blocks differ from the native source")
    return bundle
