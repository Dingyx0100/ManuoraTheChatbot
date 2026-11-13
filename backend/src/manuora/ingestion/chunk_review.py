"""Generate experimental Docling chunk reviews from a saved parse."""

import json
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

from docling.chunking import HierarchicalChunker, HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

from manuora.ingestion.chunk_context import adapt_chunk, map_item_context
from manuora.ingestion.files import resolve_output_dir, resolve_source_path, sha256_file
from manuora.ingestion.table_normalization import format_corrected_table_row
from manuora.knowledge.manifests import load_manifest


def review_chunks(document, chunker, tokenizer, output_path: Path) -> dict:
    """Save inspectable chunks and summarize their size and source coverage."""
    started = perf_counter()
    token_counts = []
    page_numbers = set()
    label_counts = {}
    with output_path.open("w", encoding="utf-8") as output:
        for index, chunk in enumerate(chunker.chunk(document)):
            items = chunk.meta.doc_items or []
            pages = sorted(
                {prov.page_no for item in items for prov in getattr(item, "prov", [])}
            )
            labels = sorted({str(item.label) for item in items})
            context = chunker.contextualize(chunk)
            tokens = tokenizer.count_tokens(context)
            token_counts.append(tokens)
            page_numbers.update(pages)
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            output.write(
                json.dumps(
                    {
                        "index": index,
                        "text": chunk.text,
                        "contextual_text": context,
                        "headings": chunk.meta.headings or [],
                        "refs": [item.self_ref for item in items],
                        "pages": pages,
                        "labels": labels,
                        "context_tokens": tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    ordered = sorted(token_counts)
    return {
        "chunks": len(ordered),
        "seconds": round(perf_counter() - started, 2),
        "output_bytes": output_path.stat().st_size,
        "pages_with_chunks": len(page_numbers),
        "pages_without_chunks": sorted(
            set(range(1, document.num_pages() + 1)) - page_numbers
        ),
        "median_context_tokens": median(ordered) if ordered else 0,
        "p95_context_tokens": ordered[int(0.95 * (len(ordered) - 1))] if ordered else 0,
        "max_context_tokens": ordered[-1] if ordered else 0,
        "label_counts": label_counts,
    }


def review_adapted_chunks(
    document, sections_path: Path, tokenizer, hybrid_path: Path, output_path: Path,
    corrected_table_refs: set[str] | None = None,
) -> dict:
    """Apply normalized heading metadata to saved Hybrid chunks for comparison."""
    started = perf_counter()
    with sections_path.open(encoding="utf-8") as source:
        contexts = map_item_context(document, (json.loads(line) for line in source))
    token_counts = []
    status_counts = {}
    callout_chunks = 0
    with hybrid_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            chunk = adapt_chunk(json.loads(line), contexts)
            chunk["superseded_table_refs"] = sorted(
                set(chunk["refs"]) & (corrected_table_refs or set())
            )
            chunk["context_tokens"] = tokenizer.count_tokens(chunk["contextual_text"])
            token_counts.append(chunk["context_tokens"])
            status = chunk["context_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            callout_chunks += bool(chunk["callout_starts"])
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    ordered = sorted(token_counts)
    return {
        "chunks": len(ordered),
        "seconds": round(perf_counter() - started, 2),
        "output_bytes": output_path.stat().st_size,
        "median_context_tokens": median(ordered) if ordered else 0,
        "p95_context_tokens": ordered[int(0.95 * (len(ordered) - 1))] if ordered else 0,
        "max_context_tokens": ordered[-1] if ordered else 0,
        "context_status_counts": status_counts,
        "callout_start_chunks": callout_chunks,
    }


def review_corrected_tables(tables_path: Path, tokenizer, output_path: Path) -> dict:
    """Expose one corrected table row per inspectable retrieval candidate."""
    started = perf_counter()
    counts = []
    with tables_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        index = 0
        for line in source:
            table = json.loads(line)
            headings = [*table["section_path"], table["title"]]
            for row in table["rows"]:
                body = format_corrected_table_row(table, row)
                contextual = "\n".join([*headings, body])
                tokens = tokenizer.count_tokens(contextual)
                counts.append(tokens)
                output.write(json.dumps({
                    "index": index,
                    "text": body,
                    "contextual_text": contextual,
                    "headings": headings,
                    "refs": [table["table_ref"]],
                    "pages": [table["pdf_page"]],
                    "labels": ["corrected_table"],
                    "context_tokens": tokens,
                    "context_status": "normalized_table",
                    "raw_row_indices": row["raw_row_indices"],
                    "review_status": table["review_status"],
                }, ensure_ascii=False) + "\n")
                index += 1
    return {
        "chunks": len(counts),
        "seconds": round(perf_counter() - started, 2),
        "output_bytes": output_path.stat().st_size,
        "max_context_tokens": max(counts, default=0),
    }


def refresh_chunk_review(
    manifest_path: Path, document_id: str | None = None,
    max_tokens: int = 512,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Path:
    """Refresh a document's debug chunk files without parsing its PDF again."""
    total_started = perf_counter()
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    manifest = load_manifest(manifest_path)
    documents = [
        doc for doc in manifest.documents if not document_id or doc.id == document_id
    ]
    if len(documents) != 1:
        raise ValueError("select exactly one document with document_id")
    source_path = resolve_source_path(manifest_path, documents[0])
    output_base = resolve_output_dir(source_path, documents[0])
    parsed_path = output_base / "parsed.json"
    sections_path = output_base / "normalized_sections.jsonl"
    tables_path = output_base / "normalized_tables.jsonl"
    if not parsed_path.is_file():
        raise FileNotFoundError(parsed_path)
    if not sections_path.is_file():
        raise FileNotFoundError(sections_path)

    started = perf_counter()
    document = DoclingDocument.load_from_json(parsed_path)
    load_seconds = round(perf_counter() - started, 2)
    started = perf_counter()
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        tokenizer_name, max_tokens=max_tokens, local_files_only=True
    )
    tokenizer_load_seconds = round(perf_counter() - started, 2)
    output_dir = output_base / "debug" / "docling_chunk_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, chunker in (
        ("hierarchical", HierarchicalChunker()),
        ("hybrid", HybridChunker(tokenizer=tokenizer)),
    ):
        results[name] = review_chunks(
            document, chunker, tokenizer, output_dir / f"{name}.jsonl"
        )
    results["adapted_hybrid"] = review_adapted_chunks(
        document,
        sections_path,
        tokenizer,
        output_dir / "hybrid.jsonl",
        output_dir / "adapted_hybrid.jsonl",
        {json.loads(line)["table_ref"] for line in tables_path.read_text(encoding="utf-8").splitlines()}
        if tables_path.is_file() else set(),
    )
    if tables_path.is_file():
        results["corrected_tables"] = review_corrected_tables(
            tables_path, tokenizer, output_dir / "corrected_tables.jsonl"
        )

    summary = {
        "document_id": documents[0].id,
        "source_pdf": source_path.name,
        "source_sha256": sha256_file(source_path),
        "docling_version": version("docling"),
        "docling_core_version": version("docling-core"),
        "tokenizer": tokenizer_name,
        "max_tokens": max_tokens,
        "parsed_load_seconds": load_seconds,
        "tokenizer_load_seconds": tokenizer_load_seconds,
        "total_wall_seconds": round(perf_counter() - total_started, 2),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return output_dir
