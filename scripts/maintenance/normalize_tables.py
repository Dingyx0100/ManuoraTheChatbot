"""Apply reviewed table corrections to an existing parsed document."""

import argparse
from pathlib import Path

from manuora.ingestion.files import resolve_output_dir, resolve_source_path
from manuora.ingestion.table_normalization import export_normalized_tables
from manuora.knowledge.manifests import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--document-id")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    documents = [doc for doc in manifest.documents if not args.document_id or doc.id == args.document_id]
    if len(documents) != 1:
        parser.error("select exactly one document with --document-id")
    pdf_path = resolve_source_path(args.manifest, documents[0])
    directory = resolve_output_dir(pdf_path, documents[0])
    records = export_normalized_tables(
        directory / "parsed.json",
        pdf_path,
        directory / "normalized_sections.jsonl",
        directory / "table_corrections.yaml",
    )
    print(f"Wrote {len(records)} corrected table(s): {directory / 'normalized_tables.jsonl'}")


if __name__ == "__main__":
    main()
