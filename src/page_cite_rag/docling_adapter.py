"""Read-only Docling retrieval over contextualized chunks and native source blocks.

Retrieved chunk text is reading context. Citations map to saved native text items
or structured whole tables; table rows are not guessed from serialized text. This
adapter does not modify SQLite schema or the parser.
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
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL
from page_cite_rag.tokenize import LexicalTokenizer

LEGACY_ADAPTER_VERSION = "docling-native-item-or-table/v2"
ADAPTER_VERSION = "docling-native-item-or-table/v3"


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
    chunk artifact. A virtual unit is one saved text item or structured whole table,
    *not* a PDF page. ``location_for`` carries Docling's item-level provenance.
    """

    def __init__(
        self,
        document: dict,
        chunks: list[dict],
        *,
        source_sha256: str,
        source_name: str,
        page_count: int,
        source_order: tuple[str, ...] = (),
        adapter_version: str = ADAPTER_VERSION,
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
        if adapter_version not in (LEGACY_ADAPTER_VERSION, ADAPTER_VERSION):
            raise DoclingSourceError("unknown native source adapter version")
        identity = json.dumps(
            [adapter_version, source_sha256, source_name, page_count, document, chunks],
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
        self._ref_blocks: dict[str, tuple[str, ...]] = {}
        self.chunk_native_refs: list[tuple[str, ...]] = []
        self.chunk_block_ids: list[tuple[str, ...]] = []

        def add_block(text: str, *, ref: str, source: dict) -> str:
            ordinal = len(self._blocks) + 1
            block_id = hashlib.sha256(f"{document_id}\0{ref}\0item".encode()).hexdigest()
            page_number = source["pages"][0] if len(source["pages"]) == 1 else None
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
            self._blocks.append(block)
            self._by_id[block_id] = block
            self._units[ordinal] = ParsedUnit(
                unit_index=ordinal, page_number=page_number, text=text, status="ok"
            )
            self._locations[block_id] = {
                "native_ref": ref,
                "source_type": source["label"],
                "kind": "table" if source["table"] is not None else source["kind"],
                "pages": source["pages"],
                "regions": source["regions"],
                "pdf_character_offsets": "unavailable",
                "source_text_scope": "saved_table_cells_with_coordinates"
                if source["table"] is not None
                else "saved_native_item",
                "cell": None,
                "table": (
                    {
                        "rows": source["table"]["rows"],
                        "columns": source["table"]["columns"],
                        "cell_page_mapping": source["table"]["cell_page_mapping"],
                    }
                    if source["table"] is not None
                    else None
                ),
            }
            return block_id

        chunk_refs: list[str] = []
        for entry in chunks:
            refs = entry.get("chunk", {}).get("meta", {}).get("doc_items")
            if not isinstance(refs, list) or not refs:
                raise DoclingSourceError("candidate chunk lacks source refs")
            current_refs = []
            for item in refs:
                ref = item.get("self_ref") if isinstance(item, dict) else None
                if not isinstance(ref, str):
                    raise DoclingSourceError("candidate chunk has invalid native ref")
                current_refs.append(ref)
                if ref not in chunk_refs:
                    chunk_refs.append(ref)
            self.chunk_native_refs.append(tuple(dict.fromkeys(current_refs)))

        caption_refs = []
        for ref in chunk_refs:
            source = read_native_item(document, ref, page_count=page_count)
            caption_refs.extend(source["caption_refs"])
        ordered_refs = list(dict.fromkeys((*source_order, *chunk_refs, *caption_refs)))
        for ref in ordered_refs:
            source = read_native_item(
                document,
                ref,
                page_count=page_count,
                include_enumerated_marker=adapter_version == ADAPTER_VERSION,
            )
            created = []
            if source["table"] is not None:
                rows, columns = source["table"]["rows"], source["table"]["columns"]
                lines = [f"Table ({rows} rows, {columns} columns)"]
                for cell in sorted(
                    source["table"]["cells"],
                    key=lambda value: (
                        value["row_start"],
                        value["column_start"],
                        value["row_end"],
                        value["column_end"],
                    ),
                ):
                    if cell["text"].strip():
                        lines.append(
                            f"[rows {cell['row_start']}–{cell['row_end']}; "
                            f"columns {cell['column_start']}–{cell['column_end']}] {cell['text']}"
                        )
                if len(lines) > 1:
                    created.append(add_block("\n".join(lines), ref=ref, source=source))
            elif source["saved_item_text"] is not None:
                created.append(add_block(source["saved_item_text"], ref=ref, source=source))
            self._ref_blocks[ref] = tuple(created)

        for refs in self.chunk_native_refs:
            block_ids = tuple(
                dict.fromkeys(
                    block_id
                    for ref in refs
                    for block_id in self.block_ids_for_ref(self.document_id, ref)
                )
            )
            if not block_ids:
                raise DoclingSourceError("candidate chunk has no readable native source")
            self.chunk_block_ids.append(block_ids)

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
        """Return honest item-level citations where Docling does not expose row IDs."""
        selected: list[str] = []
        for ref in self.context_refs_for_chunk(chunk_index):
            selected.extend(self.block_ids_for_ref(self.document_id, ref))
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
        candidates: int = FORMAL_RETRIEVAL.candidates_per_branch,
        constant: int = FORMAL_RETRIEVAL.rrf_constant,
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
        if type(top_k) is not int or not 1 <= top_k <= 2 * self.candidates:
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
            "context_serialization": "docling-chunk-plus-native-table-captions/v2",
            "version": "docling-chunk-top-k-context-to-native-source/v4",
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
