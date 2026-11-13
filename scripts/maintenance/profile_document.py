"""Profile a PDF before conversion and save an advisory parsing plan."""

import argparse
from pathlib import Path
from time import perf_counter

from manuora.ingestion.files import resolve_output_dir, resolve_source_path
from manuora.ingestion.parsing_planner import create_parsing_plan
from manuora.ingestion.profiler import profile_document
from manuora.knowledge.manifests import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--input", type=Path)
    parser.add_argument("--document-id")
    parser.add_argument("--output", type=Path, help="Default: profile.json beside the PDF")
    args = parser.parse_args()
    if args.manifest:
        manifest = load_manifest(args.manifest)
        documents = [
            item for item in manifest.documents
            if not args.document_id or item.id == args.document_id
        ]
        if len(documents) != 1:
            parser.error("select exactly one document with --document-id")
        document = documents[0]
        path = resolve_source_path(args.manifest, document)
        output_dir = resolve_output_dir(path, document)
        metadata = dict(
            product_id=manifest.product_id,
            product_version=manifest.version,
            document_id=document.id,
        )
    else:
        path = args.input
        output_dir = path.parent
        metadata = {}
    started = perf_counter()
    profile = profile_document(path, **metadata)
    profile_seconds = perf_counter() - started
    plan = create_parsing_plan(profile)
    output = args.output or output_dir / "profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
    plan_path = output.parent / "parsing_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Profile: {output}")
    print(f"Plan: {plan_path} ({plan.status}; OCR recommendation: {plan.ocr_mode})")
    print(
        f"Pages: {profile.page_count}; blank: {sum(p.likely_blank for p in profile.page_profiles)}; "
        f"OCR candidates: {len(plan.ocr_pages)}; TOC: {profile.toc_pages}; "
        f"time: {profile_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
