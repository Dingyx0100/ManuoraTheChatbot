"""Classify parsed PDF regions for future text and visual indexing."""

import argparse
import json
from pathlib import Path

from manuora.ingestion.files import (
    document_metadata_sha256, resolve_output_dir, resolve_source_path,
)
from manuora.ingestion.visual_regions import inventory_regions
from manuora.knowledge.manifests import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--pages", help="Comma-separated physical PDF pages for inspection")
    parser.add_argument("--output-dir", type=Path,
                        help="Required with --pages; keeps partial results out of production outputs")
    args = parser.parse_args()
    if bool(args.pages) != bool(args.output_dir):
        parser.error("--pages and --output-dir must be supplied together")
    manifest = load_manifest(args.manifest)
    matches = [item for item in manifest.documents if item.id == args.document_id]
    if len(matches) != 1:
        parser.error(f"unknown document ID: {args.document_id}")
    document = matches[0]
    source = resolve_source_path(args.manifest, document)
    output_dir = resolve_output_dir(source, document)
    parsed = output_dir / "parsed.json"
    if not parsed.is_file():
        parser.error(f"parse this PDF first: {parsed}")
    try:
        pages = {int(value.strip()) for value in args.pages.split(",")} if args.pages else None
    except ValueError:
        parser.error("--pages must be comma-separated physical PDF page numbers")
    destination = args.output_dir if args.output_dir else output_dir
    report = inventory_regions(source, parsed, destination / "visual_regions.jsonl",
                               document.id, selected_pages=pages,
                               sections_path=output_dir / "normalized_sections.jsonl",
                               product_id=manifest.product_id,
                               product_version=manifest.version,
                               metadata_sha256=document_metadata_sha256(manifest, document))
    (destination / "visual_region_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{document.id}: {report['region_count']} regions on "
          f"{report['inventoried_pages']} pages; routes={report['routes']}; "
          f"{report['wall_seconds']}s -> {destination}")


if __name__ == "__main__":
    main()
