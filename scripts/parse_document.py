"""Parse one document from a Manuora source manifest."""

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from manuora.ingestion.docling_parser import parse_pdf
from manuora.ingestion.files import resolve_output_dir, resolve_source_path, sha256_file
from manuora.ingestion.ocr_enrichment import apply_ocr_plan
from manuora.ingestion.parsing_planner import create_parsing_plan
from manuora.ingestion.profile_schemas import DocumentProfile, ParsingPlan
from manuora.ingestion.profiler import profile_document
from manuora.ingestion.quality import evaluate_parsing_quality
from manuora.ingestion.review import export_review
from manuora.ingestion.table_normalization import export_normalized_tables
from manuora.knowledge.manifests import load_manifest


PRODUCT_FILES = (
    "profile.json",
    "parsing_plan.json",
    "parsed.json",
    "parsed.confidence.json",
    "ocr_execution.json",
    "normalized_sections.jsonl",
    "normalized_tables.jsonl",
    "quality_report.json",
)
DEBUG_FILES = (
    "parsed_ocr_off.json",
    "pages.jsonl",
    "pages.md",
    "sections.jsonl",
    "callouts.jsonl",
    "toc_matches.jsonl",
    "parse_summary.json",
)


def record_processing_run(
    source_path: Path,
    output_dir: Path,
    source_sha256: str,
    profile_seconds: float | None,
    parse_seconds: float | None,
    ocr_seconds: float | None,
    review_seconds: float,
    quality_seconds: float | None,
) -> Path:
    """Append debug-run timings and current file sizes."""
    output_paths = [output_dir / name for name in PRODUCT_FILES]
    output_paths.extend(output_dir / "debug" / name for name in DEBUG_FILES)
    output_bytes = {
        path.relative_to(output_dir).as_posix(): path.stat().st_size
        for path in output_paths
        if path.is_file()
    }
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "review_only" if parse_seconds is None else "parse_and_review",
        "debug": True,
        "source_sha256": source_sha256,
        "source_bytes": source_path.stat().st_size,
        "docling_version": version("docling"),
        "docling_core_version": version("docling-core"),
        "docling_parse_version": version("docling-parse"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "parse_wall_seconds": parse_seconds,
        "ocr_wall_seconds": ocr_seconds,
        "profile_wall_seconds": profile_seconds,
        "review_wall_seconds": review_seconds,
        "quality_wall_seconds": quality_seconds,
        "output_bytes": output_bytes,
        "total_output_bytes": sum(output_bytes.values()),
        "gpu_usage": None,
        "gpu_usage_note": "Not measured by this command",
    }
    metrics_path = output_dir / "debug" / "processing_runs.jsonl"
    with metrics_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and inspect one source PDF")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--document-id")
    parser.add_argument("--trusted-outline", type=Path,
                        help="Optional independently derived, reviewed TOC JSON for quality evaluation")
    parser.add_argument("--trust-pdf-bookmarks", action="store_true",
                        help="Declare PDF bookmarks verified; enable the below-70-percent outline failure gate")
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Regenerate normalized sections from an existing parsed.json",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also export inspection files and record processing metrics under debug/",
    )
    parser.add_argument(
        "--apply-ocr",
        action="store_true",
        help="With --review-only, execute the saved plan's OCR pages on parsed.json",
    )
    parser.add_argument(
        "--ocr-pages",
        help="Comma-separated subset of planned pages for an OCR trial (marks execution partial)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip planned OCR on a fresh parse (marks execution partial)",
    )
    args = parser.parse_args()
    if args.apply_ocr and not args.review_only:
        parser.error("--apply-ocr is only needed with --review-only")
    if args.no_ocr and args.apply_ocr:
        parser.error("--no-ocr conflicts with --apply-ocr")
    if args.ocr_pages and args.no_ocr:
        parser.error("--ocr-pages conflicts with --no-ocr")
    if args.ocr_pages and args.review_only and not args.apply_ocr:
        parser.error("--ocr-pages requires --apply-ocr with --review-only")
    try:
        selected_ocr_pages = (
            [int(page.strip()) for page in args.ocr_pages.split(",")]
            if args.ocr_pages else None
        )
    except ValueError:
        parser.error("--ocr-pages must be comma-separated page numbers")

    manifest = load_manifest(args.manifest)
    if args.document_id:
        documents = [doc for doc in manifest.documents if doc.id == args.document_id]
    else:
        documents = manifest.documents
    if len(documents) != 1:
        parser.error("select exactly one document with --document-id")

    source_path = resolve_source_path(args.manifest, documents[0])
    output_dir = resolve_output_dir(source_path, documents[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = output_dir / "parsed.json"
    source_sha256 = sha256_file(source_path)
    print(f"Source SHA-256: {source_sha256}")
    profile_path = output_dir / "profile.json"
    profile = None
    plan = None
    profile_seconds = None
    parse_seconds = None
    ocr_seconds = None
    if not args.review_only:
        started = perf_counter()
        profile = profile_document(
            source_path,
            product_id=manifest.product_id,
            product_version=manifest.version,
            document_id=documents[0].id,
        )
        profile_seconds = perf_counter() - started
        profile_path.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
        plan = create_parsing_plan(profile)
        (output_dir / "parsing_plan.json").write_text(
            plan.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        print(f"Advisory parsing plan: OCR {plan.ocr_mode} on {len(plan.ocr_pages)} pages")
        started = perf_counter()
        parse_pdf(source_path, parsed_path)
        parse_seconds = perf_counter() - started
    elif not parsed_path.is_file():
        parser.error(f"parsed file does not exist: {parsed_path}")
    elif profile_path.is_file():
        profile = DocumentProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
        if profile.source_sha256 != source_sha256:
            parser.error("profile source hash differs from the current PDF; reprofile it")
        plan_path = output_dir / "parsing_plan.json"
        if plan_path.is_file():
            plan = ParsingPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    if not args.review_only or args.apply_ocr:
        if plan is None:
            parser.error("OCR execution requires a parsing plan")
        execution_path = output_dir / "ocr_execution.json"
        previous_execution = (
            json.loads(execution_path.read_text(encoding="utf-8"))
            if args.review_only and execution_path.is_file() else None
        )
        if args.debug:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            shutil.copyfile(parsed_path, debug_dir / "parsed_ocr_off.json")
        started = perf_counter()
        execution = apply_ocr_plan(
            source_path,
            parsed_path,
            plan,
            selected_pages=[] if args.no_ocr else selected_ocr_pages,
            previous_execution=previous_execution,
            language=manifest.language,
        )
        ocr_seconds = perf_counter() - started
        execution_path.write_text(
            json.dumps(execution, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"OCR: {execution['accepted_page_count']}/{len(execution['selected_pages'])} "
            f"processed pages accepted; {execution['added_line_count']} OCR lines total "
            f"({len(execution['newly_processed_pages'])} pages processed this run)"
        )
    started = perf_counter()
    summary = export_review(
        parsed_path, debug=args.debug,
        outline_pdf=source_path,
    )
    review_seconds = perf_counter() - started
    print(f"Review: {summary}")
    annotations_path = output_dir / "table_corrections.yaml"
    if annotations_path.is_file():
        tables = export_normalized_tables(
            parsed_path, source_path,
            output_dir / "normalized_sections.jsonl", annotations_path,
        )
        print(f"Corrected tables: {len(tables)}")
    quality_seconds = None
    if profile is not None:
        started = perf_counter()
        quality = evaluate_parsing_quality(
            profile,
            parsed_path,
            output_dir / "normalized_sections.jsonl",
            output_dir / "debug" / "docling_chunk_review" / "summary.json",
            output_dir / "ocr_execution.json",
            source_path=source_path,
            trusted_outline_path=args.trusted_outline,
            trust_pdf_bookmarks=args.trust_pdf_bookmarks,
        )
        (output_dir / "quality_report.json").write_text(
            quality.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        quality_seconds = perf_counter() - started
        print(f"Quality: {quality.status}; {quality.reasons}")
    if args.debug:
        metrics_path = record_processing_run(
            source_path, output_dir, source_sha256, profile_seconds, parse_seconds, ocr_seconds,
            review_seconds, quality_seconds,
        )
        print(f"Processing metrics: {metrics_path}")


if __name__ == "__main__":
    main()
