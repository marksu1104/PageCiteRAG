"""Read-only Docling retrieval over contextualized chunks and native source blocks.

Retrieved chunk text is reading context. Every offered citation span still comes
from a saved native text item or table cell, with its original page regions kept
separately for read-back. This adapter does not modify SQLite schema or the parser.
"""

from __future__ import annotations

import hashlib
import json
import time
from bisect import bisect_right
from dataclasses import dataclass

import bm25s
import numpy as np

from page_cite_rag.docling_source import DoclingSourceError, read_native_item
from page_cite_rag.embedding import EmbeddingEncoder, normalized_vectors
from page_cite_rag.models import Block, ParsedUnit
from page_cite_rag.retrieval import SearchContextGroup, SearchHit, SearchResult
from page_cite_rag.tokenize import LexicalTokenizer

ADAPTER_VERSION = "docling-native-item-or-cell/v1"


@dataclass(frozen=True)
class DoclingDocumentIdentity:
    document_id: str
    source_sha256: str
    source_name: str
    page_count: int
    parser_name: str = "docling"


class DoclingSourceStore:
    """Satisfy the workflow's read-only source interface without false PDF offsets.

    The caller must first verify the capture manifest, source PDF SHA and candidate
    chunk artifact. A virtual unit is exactly one saved text item or table cell,
    *not* a PDF page. ``location_for`` carries its original Docling provenance.
    """

    def __init__(
        self,
        document: dict,
        chunks: list[dict],
        *,
        source_sha256: str,
        source_name: str,
        page_count: int,
    ) -> None:
        if (
            not isinstance(document, dict)
            or not isinstance(chunks, list)
            or not chunks
            or not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or not isinstance(source_name, str)
            or not source_name
            or not isinstance(page_count, int)
            or page_count < 1
        ):
            raise DoclingSourceError("invalid Docling comparison source")
        identity = json.dumps(
            [ADAPTER_VERSION, source_sha256, source_name, page_count, document, chunks],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        document_id = hashlib.sha256(identity).hexdigest()
        self.document = DoclingDocumentIdentity(
            document_id=document_id,
            source_sha256=source_sha256,
            source_name=source_name,
            page_count=page_count,
        )
        self.document_data = document
        self._chunks = chunks
        self._blocks: list[Block] = []
        self._by_id: dict[str, Block] = {}
        self._units: dict[int, ParsedUnit] = {}
        self._locations: dict[str, dict] = {}
        self._cell_headers: dict[str, tuple[bool, bool]] = {}
        self._ref_blocks: dict[str, tuple[str, ...]] = {}
        self.chunk_native_refs: list[tuple[str, ...]] = []
        self.chunk_block_ids: list[tuple[str, ...]] = []

        def add_block(text: str, *, ref: str, source: dict, cell: dict | None) -> str:
            ordinal = len(self._blocks) + 1
            block_id = hashlib.sha256(
                f"{document_id}\0{ref}\0{cell['cell_index'] if cell else 'item'}".encode()
            ).hexdigest()
            page_number = source["pages"][0] if len(source["pages"]) == 1 and cell is None else None
            starts = [0]
            for line in text.splitlines(keepends=True):
                starts.append(starts[-1] + len(line))
            block = Block(
                document_id=document_id,
                block_id=block_id,
                ordinal=ordinal,
                unit_index=ordinal,
                page_number=page_number,
                line_start=1,
                line_end=bisect_right(starts, len(text) - 1),
                char_start=0,
                char_end=len(text),
                text=text,
            )
            public_cell = (
                {
                    key: value
                    for key, value in cell.items()
                    if key not in {"row_header", "column_header"}
                }
                if cell is not None
                else None
            )
            self._blocks.append(block)
            self._by_id[block_id] = block
            self._units[ordinal] = ParsedUnit(
                unit_index=ordinal, page_number=page_number, text=text, status="ok"
            )
            self._locations[block_id] = {
                "native_ref": ref,
                "kind": "table_cell" if cell is not None else source["kind"],
                "pages": source["pages"],
                "regions": source["regions"],
                "pdf_character_offsets": "unavailable",
                "source_text_scope": (
                    "saved_native_cell" if cell is not None else "saved_native_item"
                ),
                "cell": public_cell,
            }
            if cell is not None:
                self._cell_headers[block_id] = (
                    cell["row_header"],
                    cell["column_header"],
                )
            return block_id

        for entry in chunks:
            refs = entry.get("chunk", {}).get("meta", {}).get("doc_items")
            if not isinstance(refs, list) or not refs:
                raise DoclingSourceError("candidate chunk lacks source refs")
            block_ids = []
            for item in refs:
                ref = item.get("self_ref") if isinstance(item, dict) else None
                if not isinstance(ref, str):
                    raise DoclingSourceError("candidate chunk has invalid native ref")
                if ref not in self._ref_blocks:
                    source = read_native_item(document, ref, page_count=page_count)
                    created = []
                    if source["saved_item_text"] is not None:
                        created.append(
                            add_block(source["saved_item_text"], ref=ref, source=source, cell=None)
                        )
                    elif source["table"] is not None:
                        for index, table_cell in enumerate(source["table"]["cells"]):
                            if table_cell["text"].strip():
                                cell = {
                                    "cell_index": index,
                                    "row_start": table_cell["row_start"],
                                    "row_end": table_cell["row_end"],
                                    "column_start": table_cell["column_start"],
                                    "column_end": table_cell["column_end"],
                                    "bbox": table_cell["bbox"],
                                    "cell_page_mapping": "unverified",
                                }
                                cell["row_header"] = table_cell["row_header"]
                                cell["column_header"] = table_cell["column_header"]
                                created.append(
                                    add_block(table_cell["text"], ref=ref, source=source, cell=cell)
                                )
                    self._ref_blocks[ref] = tuple(created)
                block_ids.extend(self._ref_blocks[ref])
            if not block_ids:
                raise DoclingSourceError("candidate chunk has no readable native source")
            self.chunk_native_refs.append(tuple(dict.fromkeys(item["self_ref"] for item in refs)))
            self.chunk_block_ids.append(tuple(dict.fromkeys(block_ids)))

    @property
    def document_id(self) -> str:
        return self.document.document_id

    def get_document(self, document_id: str) -> DoclingDocumentIdentity | None:
        return self.document if document_id == self.document_id else None

    def get_block(self, document_id: str, block_id: str) -> Block | None:
        return self._by_id.get(block_id) if document_id == self.document_id else None

    def get_unit(self, document_id: str, unit_index: int) -> ParsedUnit | None:
        return self._units.get(unit_index) if document_id == self.document_id else None

    def get_blocks(self, document_id: str) -> list[Block]:
        return list(self._blocks) if document_id == self.document_id else []

    def block_ids_for_ref(self, document_id: str, ref: str) -> tuple[str, ...]:
        return self._ref_blocks.get(ref, ()) if document_id == self.document_id else ()

    def context_refs_for_chunk(self, chunk_index: int) -> tuple[str, ...]:
        """Include native table captions that contextualize a chunk's saved rows."""
        if type(chunk_index) is not int or not 0 <= chunk_index < len(self.chunk_native_refs):
            raise ValueError("invalid Docling chunk index")
        refs = list(self.chunk_native_refs[chunk_index])
        for ref in self.chunk_native_refs[chunk_index]:
            source = read_native_item(self.document_data, ref, page_count=self.document.page_count)
            for caption_ref in source["caption_refs"]:
                if caption_ref not in refs:
                    refs.append(caption_ref)
        return tuple(refs)

    def contextualized_chunk_text(self, chunk_index: int) -> str:
        """Use Docling's chunk text and restore native table captions when absent."""
        if type(chunk_index) is not int or not 0 <= chunk_index < len(self._chunks):
            raise ValueError("invalid Docling chunk index")
        text = self._chunks[chunk_index]["search_text"]
        captions = []
        for ref in self.chunk_native_refs[chunk_index]:
            source = read_native_item(self.document_data, ref, page_count=self.document.page_count)
            for caption_ref in source["caption_refs"]:
                caption = read_native_item(
                    self.document_data, caption_ref, page_count=self.document.page_count
                )["saved_item_text"]
                if caption and caption not in text and caption not in captions:
                    captions.append(caption)
        return "\n".join((*captions, text))

    def context_block_ids_for_chunk(self, chunk_index: int) -> tuple[str, ...]:
        """Map serialized table rows to exact cells, row labels and column headers."""
        text = " ".join(self.contextualized_chunk_text(chunk_index).split()).casefold()
        selected: list[str] = []
        for ref in self.context_refs_for_chunk(chunk_index):
            block_ids = self.block_ids_for_ref(self.document_id, ref)
            if not ref.startswith("#/tables/"):
                selected.extend(block_ids)
                continue
            cells = {
                block_id: {
                    **self.location_for(self.document_id, block_id)["cell"],
                    "row_header": self._cell_headers[block_id][0],
                    "column_header": self._cell_headers[block_id][1],
                }
                for block_id in block_ids
            }
            matching = [
                block_id
                for block_id in block_ids
                if " ".join(self._by_id[block_id].text.split()).casefold() in text
            ]
            related = set(matching)
            for block_id in matching:
                cell = cells[block_id]
                for candidate_id, candidate in cells.items():
                    same_row = (
                        candidate["row_start"] < cell["row_end"]
                        and cell["row_start"] < candidate["row_end"]
                    )
                    same_column = (
                        candidate["column_start"] < cell["column_end"]
                        and cell["column_start"] < candidate["column_end"]
                    )
                    if same_row and (
                        candidate["row_header"]
                        or (cell["column_start"] > 0 and candidate["column_start"] == 0)
                    ):
                        related.add(candidate_id)
                    if cell["row_start"] > 0 and candidate["column_header"] and same_column:
                        related.add(candidate_id)
            selected.extend(block_id for block_id in block_ids if block_id in related)
        return tuple(dict.fromkeys(selected))

    def get_neighbor_blocks(
        self, document_id: str, ordinal: int, *, radius: int = 1
    ) -> list[Block]:
        if (
            not isinstance(ordinal, int)
            or ordinal < 1
            or not isinstance(radius, int)
            or not 0 <= radius <= 2
        ):
            raise ValueError("invalid Docling neighborhood")
        if document_id != self.document_id:
            return []
        return self._blocks[max(0, ordinal - radius - 1) : ordinal + radius]

    def location_for(self, document_id: str, block_id: str) -> dict | None:
        if document_id != self.document_id:
            return None
        location = self._locations.get(block_id)
        return json.loads(json.dumps(location)) if location is not None else None


class DoclingChunkRetriever:
    """Rank Docling chunks, then carry their text beside verifiable source blocks.

    The contextualized representation is not a citation. Native source blocks remain
    the only citeable text. The in-memory retrieval index has no silent fallback.
    """

    def __init__(
        self,
        store: DoclingSourceStore,
        chunks: list[dict],
        encoder: EmbeddingEncoder,
        *,
        candidates: int = 20,
        constant: int = 60,
        lexical_overrides: dict[str, str] | None = None,
    ) -> None:
        if (
            len(chunks) != len(store.chunk_block_ids)
            or type(candidates) is not int
            or candidates < 1
            or type(constant) is not int
            or constant < 1
        ):
            raise ValueError("invalid Docling retrieval source or RRF budget")
        self.store = store
        self.encoder = encoder
        self.candidates = candidates
        self.constant = constant
        self.lexical_overrides = dict(lexical_overrides or {})
        self.last_query_plan: dict | None = None
        self._tokenizer = LexicalTokenizer()
        search_texts = [entry.get("search_text") for entry in chunks]
        if any(not isinstance(text, str) or not text.strip() for text in search_texts):
            raise DoclingSourceError("candidate search chunk is empty")
        self._engine = bm25s.BM25(method="lucene", k1=1.5, b=0.75, backend="numpy")
        self._engine.index(
            [self._tokenizer.tokenize(text) for text in search_texts],
            create_empty_token=False,
            show_progress=False,
        )
        self._vectors = normalized_vectors(
            encoder.encode_documents(search_texts), len(search_texts), encoder.dimension
        )
        self._search_texts = search_texts
        identity = json.dumps(
            [store.document_id, search_texts, encoder.identity, candidates, constant],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.index_id = hashlib.sha256(identity).hexdigest()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        allowed_chunk_indices: tuple[int, ...] | None = None,
    ) -> SearchResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty")
        if type(top_k) is not int or not 1 <= top_k <= self.candidates:
            raise ValueError("top_k exceeds the candidate budget")
        allowed = tuple(range(len(self.store.chunk_block_ids)))
        if allowed_chunk_indices is not None:
            if (
                not allowed_chunk_indices
                or any(
                    type(index) is not int or not 0 <= index < len(self.store.chunk_block_ids)
                    for index in allowed_chunk_indices
                )
                or len(set(allowed_chunk_indices)) != len(allowed_chunk_indices)
            ):
                raise ValueError("allowed chunk scope must contain unique valid indexes")
            allowed = allowed_chunk_indices
        started = time.perf_counter()
        lexical_query = self.lexical_overrides.get(query, query)
        tokens = [
            token
            for token in self._tokenizer.tokenize(lexical_query)
            if token in self._engine.vocab_dict
        ]
        lexical_order = np.empty(0, dtype=np.int64)
        if tokens:
            scores = self._engine.get_scores(tokens)
            if scores.shape != (len(self.store.chunk_block_ids),) or not np.all(
                np.isfinite(scores)
            ):
                raise DoclingSourceError("candidate BM25 returned invalid scores")
            matches = np.flatnonzero(scores > 0)
            if allowed_chunk_indices is not None:
                matches = matches[np.isin(matches, allowed)]
            lexical_order = matches[np.argsort(-scores[matches], kind="stable")][: self.candidates]
        query_vector = normalized_vectors(
            self.encoder.encode_queries([query]), 1, self.encoder.dimension
        )[0]
        dense_scores = self._vectors @ query_vector
        if not np.all(np.isfinite(dense_scores)):
            raise DoclingSourceError("candidate dense search returned invalid scores")
        candidate_indexes = np.asarray(allowed, dtype=np.int64)
        dense_order = candidate_indexes[
            np.argsort(-dense_scores[candidate_indexes], kind="stable")[: self.candidates]
        ]
        fused: dict[int, float] = {}
        for order in (lexical_order, dense_order):
            for rank, index in enumerate(order, 1):
                position = int(index)
                fused[position] = fused.get(position, 0.0) + 1 / (self.constant + rank)
        query_terms = set(self._tokenizer.tokenize(query))
        blocks = {block.block_id: block for block in self.store.get_blocks(self.store.document_id)}
        ranked_chunks = sorted(fused, key=lambda index: (-fused[index], index))
        selected_chunk_indices = tuple(ranked_chunks[:top_k])
        chunk_contexts = {
            chunk_index: SearchContextGroup(
                chunk_index=chunk_index,
                text=self.store.contextualized_chunk_text(chunk_index),
                source_refs=self.store.context_refs_for_chunk(chunk_index),
                source_block_ids=self.store.context_block_ids_for_chunk(chunk_index),
            )
            for chunk_index in selected_chunk_indices
        }
        block_scores: dict[str, float] = {}
        block_chunk_ranks: dict[str, int] = {}
        for chunk_rank, chunk_index in enumerate(selected_chunk_indices, 1):
            for block_id in chunk_contexts[chunk_index].source_block_ids:
                block_scores[block_id] = max(block_scores.get(block_id, 0.0), fused[chunk_index])
                block_chunk_ranks.setdefault(block_id, chunk_rank)
        ordered = sorted(
            block_scores,
            key=lambda block_id: (
                block_chunk_ranks[block_id],
                -len(query_terms & set(self._tokenizer.tokenize(blocks[block_id].text))),
                blocks[block_id].ordinal,
                block_id,
            ),
        )
        context_groups = tuple(chunk_contexts[index] for index in selected_chunk_indices)
        self.last_query_plan = {
            "lexical_query": lexical_query,
            "dense_query": query,
            "lexical_chunks": len(lexical_order),
            "dense_chunks": len(dense_order),
            "fused_chunks": len(fused),
            "source_blocks": len(block_scores),
            "selected_context_chunks": list(selected_chunk_indices),
            "selected_context_chunk_count": len(context_groups),
            "selected_source_block_count": len(ordered),
            "scoped_chunk_count": len(allowed) if allowed_chunk_indices is not None else None,
            "context_serialization": "docling-chunk-plus-native-table-captions/v1",
            "version": "docling-chunk-top-k-context-to-native-source/v3",
        }
        return SearchResult(
            document_id=self.store.document_id,
            index_id=self.index_id,
            method="hybrid",
            query=query,
            hits=tuple(
                SearchHit(rank=rank, score=block_scores[block_id], block=blocks[block_id])
                for rank, block_id in enumerate(ordered, 1)
            ),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            context_groups=context_groups,
        )
