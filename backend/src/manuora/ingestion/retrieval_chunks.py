"""Assemble searchable chunks from Docling, normalized sections, and table repairs."""

import json
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer, ChunkingSerializerProvider, TripletTableSerializer,
)
from docling_core.transforms.serializer.common import create_ser_result

from manuora.ingestion.chunk_context import adapt_chunk, map_item_context
from manuora.ingestion.files import sha256_file
from manuora.ingestion.table_normalization import format_corrected_table_row
from manuora.ingestion.chunk_preservation import evaluate_chunk_preservation, source_inventory
from manuora.ingestion.failure_detection import configuration_text, detect_failures

ASSEMBLY_VERSION = "2"


class AutomaticTableSerializer(TripletTableSerializer):
    def serialize(self, *, item, doc_serializer, doc, **kwargs):
        text = configuration_text(item)
        # Keep captions and all unsupported table forms on Docling's normal path.
        if text is None or item.captions or item.self_ref in doc_serializer.get_excluded_refs(**kwargs):
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        return create_ser_result(text=text, span_source=item)


class CompleteSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        serializer = ChunkingDocSerializer(doc=doc, table_serializer=AutomaticTableSerializer())
        serializer.params = serializer.params.model_copy(update={"traverse_pictures": True})
        return serializer


def assemble_chunks(
    document: DoclingDocument, sections: list[dict], tables: list[dict],
    manifest, document_manifest, source_path: Path, parsed_path: Path, tokenizer,
) -> tuple[list[dict], dict]:
    """Replace corrected table refs once and retain neighboring Docling text items."""
    context_by_ref = map_item_context(document, sections)
    heading_refs_by_path: dict[tuple, list[str]] = {}
    for section in sections:
        if section.get("normalization_status") == "matched" and section.get("normalized_heading_path"):
            heading_refs_by_path.setdefault(tuple(section["normalized_heading_path"]), []).append(section["ref"])
    items_by_ref = {item.self_ref: item for item, _ in document.iterate_items(traverse_pictures=True)}
    table_by_ref = {table["table_ref"]: table for table in tables}
    if len(table_by_ref) != len(tables):
        raise ValueError("duplicate corrected table refs")
    source_hash = sha256_file(source_path)
    parsed_hash = sha256_file(parsed_path)
    for table in tables:
        if table["source_pdf_sha256"] != source_hash or table["parsed_sha256"] != parsed_hash:
            raise ValueError(f"stale corrected table: {table['table_ref']}")
        if table["table_ref"] not in items_by_ref:
            raise ValueError(f"corrected table ref is absent from parse: {table['table_ref']}")

    common = {
        "product_id": manifest.product_id,
        "product_version": manifest.version,
        "document_id": document_manifest.id,
        "document_title": document_manifest.title,
        "document_type": document_manifest.type,
        "language": manifest.language,
        "source_pdf": source_path.name,
        "source_pdf_sha256": source_hash,
        "parsed_sha256": parsed_hash,
    }
    output: list[dict] = []
    emitted_tables: set[str] = set()
    skipped = {"document_index": 0, "unresolved": 0}
    split_mixed_sections = 0
    preserved_mixed_prose_refs: set[str] = set()

    def add(text: str, headings: list[str], refs: list[str], pages: list[int],
            kind: str, origin_index: int, **extra) -> None:
        if not text.strip():
            return
        contextual = "\n".join([*headings, text.strip()])
        source_ids = set(refs)
        # Heading IDs describe context supplied alongside body text. Preserve
        # existing refs for downstream compatibility with the body references.
        for ref in refs:
            section_ref = context_by_ref.get(ref, {}).get("section_ref")
            if section_ref:
                source_ids.add(section_ref)
        for end in range(1, len(headings) + 1):
            source_ids.update(heading_refs_by_path.get(tuple(headings[:end]), []))
        if kind == "corrected_table_row":
            source_ids.add(f"{extra['table_ref']}/normalized_rows/{extra['table_row_index']}")
        output.append({
            **common,
            "chunk_id": f"{document_manifest.id}:{len(output) + 1:06d}",
            "text": text.strip(),
            "contextual_text": contextual,
            "headings": headings,
            "refs": refs,
            "source_block_ids": sorted(source_ids),
            "pages": sorted(set(pages)),
            "content_kind": kind,
            "origin_hybrid_index": origin_index,
            "context_tokens": tokenizer.count_tokens(contextual),
            **extra,
        })

    chunker = HybridChunker(tokenizer=tokenizer, serializer_provider=CompleteSerializerProvider())
    for index, raw in enumerate(chunker.chunk(document)):
        items = raw.meta.doc_items or []
        refs = [item.self_ref for item in items]
        pages = sorted({prov.page_no for item in items for prov in getattr(item, "prov", [])})
        labels = sorted({str(item.label) for item in items})
        adapted = adapt_chunk({
            "index": index, "text": raw.text,
            "contextual_text": chunker.contextualize(raw),
            "headings": raw.meta.headings or [], "refs": refs,
            "pages": pages, "labels": labels,
        }, context_by_ref)
        if "document_index" in labels:
            skipped["document_index"] += 1
            continue
        if adapted["context_status"] == "mixed_sections":
            split_mixed_sections += 1
            groups: list[tuple[list[str], list[tuple[object, str]]]] = []
            for position, chunk_item in enumerate(items):
                item = items_by_ref.get(chunk_item.self_ref)
                path = context_by_ref.get(chunk_item.self_ref, {}).get("normalized_heading_path", [])
                if not path and item is not None and str(item.label) == "section_header":
                    # Docling can place a heading in the same Hybrid chunk as
                    # body items from neighboring sections. Headers are not
                    # mapped by map_item_context, so attach one to the next
                    # mapped body item (or the preceding group at chunk end).
                    path = next((
                        context_by_ref.get(later.self_ref, {}).get("normalized_heading_path", [])
                        for later in items[position + 1:]
                        if context_by_ref.get(later.self_ref, {}).get("normalized_heading_path")
                    ), groups[-1][0] if groups else [])
                if not path or item is None:
                    raise ValueError(f"cannot split mixed-section item: {chunk_item.self_ref}")
                if isinstance(getattr(item, "text", None), str):
                    item_text = item.text
                elif hasattr(item, "export_to_markdown"):
                    item_text = configuration_text(item) if hasattr(item, "data") else None
                    item_text = item_text or item.export_to_markdown(doc=document)
                else:
                    # Pictures without extracted text cannot contribute searchable content.
                    continue
                if not item_text.strip():
                    continue
                if not groups or groups[-1][0] != path:
                    groups.append((path, []))
                groups[-1][1].append((item, item_text))
            for path, group in groups:
                add("\n".join(item_text for _, item_text in group), path,
                    [item.self_ref for item, _ in group],
                    [prov.page_no for item, _ in group for prov in getattr(item, "prov", [])],
                    "split_mixed_section", index,
                    labels=sorted({str(item.label) for item, _ in group}))
            continue
        if adapted["context_status"] != "normalized":
            skipped[adapted["context_status"]] += 1
            continue
        corrected_refs = set(refs) & table_by_ref.keys()
        if not corrected_refs:
            add(adapted["text"], adapted["headings"], refs, pages,
                "hybrid", index, labels=labels)
            continue

        # A Hybrid chunk can contain both a table and prose. Rebuild only that
        # chunk from its source items, leaving unrelated Hybrid chunks intact.
        pending: list = []

        def flush_prose() -> None:
            if not pending:
                return
            prose_refs = [item.self_ref for item in pending]
            prose_pages = [prov.page_no for item in pending for prov in getattr(item, "prov", [])]
            add("\n".join(item.text for item in pending), adapted["headings"],
                prose_refs, prose_pages, "prose_near_corrected_table", index,
                labels=sorted({str(item.label) for item in pending}))
            preserved_mixed_prose_refs.update(prose_refs)
            pending.clear()

        for ref in refs:
            if ref in corrected_refs:
                flush_prose()
                if ref in emitted_tables:
                    continue
                table = table_by_ref[ref]
                for row_index, row in enumerate(table["rows"]):
                    add(format_corrected_table_row(table, row),
                        [*table["section_path"], table["title"]],
                        [ref], [table["pdf_page"]], "corrected_table_row", index,
                        labels=["corrected_table"], table_ref=ref,
                        table_row_index=row_index,
                        raw_row_indices=row["raw_row_indices"],
                        review_status=table["review_status"])
                emitted_tables.add(ref)
            else:
                item = items_by_ref.get(ref)
                if item is None:
                    raise ValueError(f"missing non-table item: {ref}")
                if not isinstance(getattr(item, "text", None), str) or not item.text.strip():
                    raise ValueError(f"cannot preserve non-table item text: {ref}")
                pending.append(item)
        flush_prose()

    # Never silently drop eligible prose/code merely because the chunker or
    # heading mapper did not emit it. Reconcile the full inventory after chunking.
    represented = {ref for chunk in output for ref in chunk["source_block_ids"]}
    inventory, _ = source_inventory(document, tables)
    recovered_refs = []
    for block in inventory:
        ref = block["block_id"]
        item = items_by_ref.get(ref)
        if ref in represented or item is None or block["label"] in {"title", "section_header"}:
            continue
        value = getattr(item, "text", "")
        if not value.strip():
            continue
        path = context_by_ref.get(ref, {}).get("normalized_heading_path") or [
            "PDF page " + ", ".join(map(str, block["pages"]))]
        # Preserve exact source characters across bounded fallback pieces.
        while value:
            low, high = 1, len(value)
            while low < high:
                mid = (low + high + 1) // 2
                if tokenizer.count_tokens("\n".join([*path, value[:mid]])) <= tokenizer.max_tokens:
                    low = mid
                else:
                    high = mid - 1
            end = low
            if end < len(value):
                boundary = value.rfind(" ", 0, end + 1)
                if boundary > end // 2:
                    end = boundary + 1
            add(value[:end], path, [ref], block["pages"], "recovered_source_text", -1,
                context_fallback=not bool(context_by_ref.get(ref, {}).get("normalized_heading_path")))
            value = value[end:]
        recovered_refs.append(ref)

    if emitted_tables != set(table_by_ref):
        raise ValueError(f"corrected tables absent from Hybrid chunks: {set(table_by_ref) - emitted_tables}")
    expected_rows = sum(len(table["rows"]) for table in tables)
    actual_rows = sum(chunk["content_kind"] == "corrected_table_row" for chunk in output)
    if actual_rows != expected_rows:
        raise ValueError("corrected table row count changed during assembly")
    report = {
        "assembly_version": ASSEMBLY_VERSION,
        "document_id": document_manifest.id,
        "source_pdf_sha256": source_hash,
        "parsed_sha256": parsed_hash,
        "chunks": len(output),
        "corrected_tables": len(emitted_tables),
        "corrected_table_rows": actual_rows,
        "preserved_mixed_prose_refs": sorted(preserved_mixed_prose_refs),
        "split_mixed_section_chunks": split_mixed_sections,
        "automatically_recovered_text_refs": recovered_refs,
        "automatic_configuration_tables": [t.self_ref for t in document.tables
                                             if t.self_ref not in table_by_ref and configuration_text(t) is not None],
        "failure_detection": detect_failures(document, source_path=source_path,
                                               parsed_path=parsed_path, chunks=output),
        "skipped_hybrid_chunks": skipped,
        "over_token_budget": [chunk["chunk_id"] for chunk in output
                              if chunk["context_tokens"] > tokenizer.max_tokens],
        "max_context_tokens": max((chunk["context_tokens"] for chunk in output), default=0),
        "source_to_chunk_preservation": evaluate_chunk_preservation(
            document, output, tables, parsed_sha256=parsed_hash, source_pdf_sha256=source_hash,
        ),
    }
    return output, report


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
