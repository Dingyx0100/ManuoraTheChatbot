"""Build retrieval candidate chunks from a saved parse and reviewed corrections."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from manuora.ingestion.files import (  # noqa: E402
    document_metadata_sha256, resolve_output_dir, resolve_source_path, sha256_file,
)
from manuora.ingestion.retrieval_chunks import assemble_chunks, read_jsonl  # noqa: E402
from manuora.knowledge.manifests import load_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--document-id")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    manifest = load_manifest(args.manifest)
    documents = [doc for doc in manifest.documents
                 if not args.document_id or doc.id == args.document_id]
    if len(documents) != 1:
        parser.error("select exactly one document with --document-id")
    source_path = resolve_source_path(args.manifest, documents[0])
    output_dir = resolve_output_dir(source_path, documents[0])
    parsed_path = output_dir / "parsed.json"
    sections_path = output_dir / "normalized_sections.jsonl"
    tables_path = output_dir / "normalized_tables.jsonl"
    if not parsed_path.is_file() or not sections_path.is_file():
        parser.error("saved parse and normalized sections are required")

    started = perf_counter()
    document = DoclingDocument.load_from_json(parsed_path)
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        args.tokenizer, max_tokens=args.max_tokens, local_files_only=True
    )
    chunks, report = assemble_chunks(
        document, read_jsonl(sections_path), read_jsonl(tables_path),
        manifest, documents[0], source_path, parsed_path, tokenizer,
    )
    report.update({
        "docling_version": version("docling"),
        "docling_core_version": version("docling-core"),
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "assembly_wall_seconds": round(perf_counter() - started, 2),
        "normalized_sections_sha256": sha256_file(sections_path),
        "normalized_tables_sha256": sha256_file(tables_path) if tables_path.is_file() else None,
        "document_metadata_sha256": document_metadata_sha256(manifest, documents[0]),
    })
    output_path = output_dir / "retrieval_chunks.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    report["retrieval_chunks_sha256"] = sha256_file(output_path)
    report_path = output_dir / "retrieval_chunk_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"source_to_chunk_preservation", "failure_detection"}},
                     indent=2, ensure_ascii=False))
    print(f"Structural diagnostics: {report['failure_detection']['counts']}")
    preservation = report["source_to_chunk_preservation"]
    print(f"Source-to-chunk preservation: {preservation['status']}; "
          f"content recovery={preservation['chunk_content_recovery']}; "
          f"missing blocks={len(preservation['missing_block_ids'])}")
    print(f"Chunks: {output_path}")


if __name__ == "__main__":
    main()
