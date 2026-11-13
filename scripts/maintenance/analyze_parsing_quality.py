"""Compare a saved Docling parse with the PDF's pre-parse profile."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from functools import lru_cache

from manuora.ingestion.files import resolve_output_dir, resolve_source_path, sha256_file
from manuora.ingestion.profile_schemas import DocumentProfile, TextCoverageSettings
from manuora.ingestion.quality import evaluate_parsing_quality
from manuora.knowledge.manifests import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--manifest", type=Path)
    scope.add_argument("--batch", action="store_true", help="Evaluate all saved parses under --source-root")
    parser.add_argument("--source-root", type=Path, default=Path("data/sources"))
    parser.add_argument("--report-dir", type=Path, default=Path("data/evaluation/parsing_quality_latest"))
    parser.add_argument("--trial-reassembly", action="store_true",
                        help="Batch only: compare deterministic assembly fixes; save candidates under report-dir, leaving live chunks intact")
    parser.add_argument("--document-id")
    parser.add_argument("--trusted-outline", type=Path,
                        help="Reviewed TOC JSON derived independently from native PDF text; otherwise use bookmarks")
    parser.add_argument("--trust-pdf-bookmarks", action="store_true",
                        help="Declare PDF bookmarks verified; enable the below-70-percent outline failure gate")
    parser.add_argument("--stage", choices=("parsing", "chunks", "all"), default="parsing")
    parser.add_argument("--chunk-exclusions", type=Path,
                        help="Optional JSON object mapping source block IDs to intentional exclusion reasons")
    defaults = TextCoverageSettings()
    parser.add_argument("--coverage-fail-below", type=float, default=defaults.fail_below)
    parser.add_argument("--coverage-warn-below", type=float, default=defaults.warn_below)
    parser.add_argument("--coverage-tokenizer", default=defaults.tokenizer_id)
    parser.add_argument("--coverage-tokenizer-revision", default=defaults.tokenizer_revision)
    args = parser.parse_args()
    if args.trial_reassembly and (not args.batch or args.stage != "all"):
        parser.error("--trial-reassembly requires --batch --stage all")
    if args.batch:
        if args.document_id or args.trusted_outline or args.chunk_exclusions or args.trust_pdf_bookmarks:
            parser.error("batch mode does not apply a single document's selection, exclusions or trust declaration to the corpus")
        evaluate_batch(args)
    else:
        evaluate_document(args)


def evaluate_document(args):
    manifest = load_manifest(args.manifest)
    documents = [
        item for item in manifest.documents
        if not args.document_id or item.id == args.document_id
    ]
    if len(documents) != 1:
        raise ValueError("select exactly one document with --document-id")
    source = resolve_source_path(args.manifest, documents[0])
    directory = resolve_output_dir(source, documents[0])
    if args.stage in {"chunks", "all"}:
        from docling_core.types.doc import DoclingDocument
        from manuora.ingestion.chunk_preservation import evaluate_chunk_preservation

        started = perf_counter()
        parsed = directory / "parsed.json"
        chunks_path = directory / "retrieval_chunks.jsonl"
        tables_path = directory / "normalized_tables.jsonl"
        report_path = directory / "retrieval_chunk_report.json"
        assembly = json.loads(report_path.read_text(encoding="utf-8"))
        parsed_hash, source_hash = sha256_file(parsed), sha256_file(source)
        if assembly.get("parsed_sha256") != parsed_hash or assembly.get("source_pdf_sha256") != source_hash:
            raise ValueError("chunk assembly belongs to a different parse/source; assemble chunks again")
        chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        tables = ([json.loads(line) for line in tables_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                  if tables_path.is_file() else [])
        exclusions = (json.loads(args.chunk_exclusions.read_text(encoding="utf-8"))
                      if args.chunk_exclusions else None)
        metric = evaluate_chunk_preservation(
            DoclingDocument.load_from_json(parsed), chunks, tables,
            parsed_sha256=parsed_hash, source_pdf_sha256=source_hash, exclusions=exclusions,
        )
        metric["evaluation_with_io_wall_seconds"] = round(perf_counter() - started, 3)
        metric["normalized_tables_sha256"] = sha256_file(tables_path) if tables_path.is_file() else None
        assembly["source_to_chunk_preservation"] = metric
        assembly["retrieval_chunks_sha256"] = sha256_file(chunks_path)
        report_path.write_text(json.dumps(assembly, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Chunk preservation: {report_path} ({metric['status']}); recovery={metric['chunk_content_recovery']}")
    if args.stage == "chunks":
        return
    profile = DocumentProfile.model_validate_json(
        (directory / "profile.json").read_text(encoding="utf-8")
    )
    if profile.source_sha256 != sha256_file(source):
        raise ValueError("profile source hash differs from the current PDF; reprofile it")
    report = evaluate_parsing_quality(
        profile,
        directory / "parsed.json",
        directory / "normalized_sections.jsonl",
        directory / "debug" / "docling_chunk_review" / "summary.json",
        directory / "ocr_execution.json",
        source_path=source,
        trusted_outline_path=args.trusted_outline,
        trust_pdf_bookmarks=args.trust_pdf_bookmarks,
        text_coverage_settings=TextCoverageSettings(
            fail_below=args.coverage_fail_below, warn_below=args.coverage_warn_below,
            tokenizer_id=args.coverage_tokenizer, tokenizer_revision=args.coverage_tokenizer_revision,
        ),
    )
    output = directory / "quality_report.json"
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Quality: {output} ({report.status})")
    for reason in report.reasons:
        print(f"- {reason}")


def evaluate_batch(args):
    """Reuse the single-document evaluator and checkpoint compact run results."""
    began = perf_counter()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(), stage=args.stage,
                   source_root=str(args.source_root.resolve()), documents=[], skipped=[], errors=[])
    pending = []
    for path in sorted(args.source_root.rglob("manifest.y*ml")):
        try:
            manifest = load_manifest(path)
            for document in manifest.documents:
                source = resolve_source_path(path, document)
                directory = resolve_output_dir(source, document)
                if not (directory / "parsed.json").is_file():
                    summary["skipped"].append(dict(document_id=document.id, reason="no saved parse"))
                    continue
                pending.append((path, manifest, document, source, directory))
        except Exception as error:
            summary["errors"].append(dict(manifest=str(path), error=f"{type(error).__name__}: {error}"))
    summary["selected_documents"] = len(pending)
    for index, (path, manifest, document, source, directory) in enumerate(pending, 1):
        started = perf_counter()
        print(f"\n[{index}/{len(pending)}] {document.id}", flush=True)
        row = dict(document_id=document.id, product_id=manifest.product_id, product_version=manifest.version,
                   source_pdf=str(source), output_dir=str(directory), manifest=str(path))
        old = directory / "quality_report.json"
        if old.is_file():
            previous = json.loads(old.read_text(encoding="utf-8"))
            row["previous_parsing_status"] = previous.get("status")
        try:
            evaluate_document(argparse.Namespace(**(vars(args) | {"manifest": path, "document_id": document.id})))
            quality = json.loads((directory / "quality_report.json").read_text(encoding="utf-8"))
            assembly = json.loads((directory / "retrieval_chunk_report.json").read_text(encoding="utf-8"))
            retention = assembly.get("source_to_chunk_preservation", {})
            row.update(
                parsing_status=quality["status"], pages=quality["page_count"], chunks=assembly["chunks"],
                native_character_coverage=quality.get("native_character_coverage_ratio"),
                native_token_coverage=quality.get("native_token_coverage_ratio"),
                native_failed_pages=quality.get("text_coverage_failed_pages", []),
                native_warning_pages=quality.get("text_coverage_warning_pages", []),
                heading_resolution=quality.get("heading_resolution_rate"),
                outline_heading_coverage=quality.get("outline_heading_coverage"),
                outline_trusted=quality.get("outline_quality", {}).get("reference_trusted"),
                section_specificity=quality.get("section_specificity"),
                specificity_assessment_coverage=quality.get("section_specificity_details", {}).get("assessment_coverage"),
                preservation_status=retention.get("status"), chunk_content_recovery=retention.get("chunk_content_recovery"),
                block_reference_recall=retention.get("block_reference_recall"),
                missing_blocks=len(retention.get("missing_block_ids", [])),
                partial_blocks=len(retention.get("partially_recovered_block_ids", [])),
                excessive_duplication_ratio=retention.get("excessive_duplication_ratio"),
                eligible_blocks=retention.get("eligible_blocks"),
                legacy_provenance_chunks=len(retention.get("legacy_provenance_chunk_ids", [])),
                reasons=quality["reasons"],
                detection_counts=(quality.get("failure_detection") or {}).get("counts", {}),
                detection_wall_seconds=(quality.get("failure_detection") or {}).get("wall_seconds"),
            )
            if args.trial_reassembly:
                row["reassembly_trial"] = trial_reassembly(args, manifest, document, source, directory, assembly)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            summary["errors"].append(dict(document_id=document.id, error=row["error"]))
            print(row["error"], flush=True)
        row["wall_seconds"] = round(perf_counter() - started, 3)
        summary["documents"].append(row)
        summary["wall_seconds"] = round(perf_counter() - began, 3)
        (args.report_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Batch results: {args.report_dir / 'results.json'}", flush=True)


@lru_cache(maxsize=4)
def trial_tokenizer(name, limit):
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    return HuggingFaceTokenizer.from_pretrained(name, max_tokens=limit, local_files_only=True)


def trial_reassembly(args, manifest, document, source, directory, assembly):
    from docling_core.types.doc import DoclingDocument
    from manuora.ingestion.retrieval_chunks import assemble_chunks, read_jsonl
    from manuora.ingestion.failure_detection import detect_failures

    began = perf_counter()
    parsed = directory / "parsed.json"
    doc = DoclingDocument.load_from_json(parsed)
    baseline = read_jsonl(directory / "retrieval_chunks.jsonl")
    before = detect_failures(doc, source_path=source, parsed_path=parsed, chunks=baseline)
    tokenizer = trial_tokenizer(assembly["tokenizer"], assembly["max_tokens"])
    chunks, report = assemble_chunks(doc, read_jsonl(directory / "normalized_sections.jsonl"),
                                    read_jsonl(directory / "normalized_tables.jsonl"),
                                    manifest, document, source, parsed, tokenizer)
    target = args.report_dir / "candidates" / document.id
    target.mkdir(parents=True, exist_ok=True)
    (target / "retrieval_chunks.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    report.update(tokenizer=assembly["tokenizer"], max_tokens=assembly["max_tokens"],
                  trial_only=True, detection_before=before)
    (target / "retrieval_chunk_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preservation = report["source_to_chunk_preservation"]
    print(f"Trial: {document.id}: recovery {preservation['chunk_content_recovery']:.4f}; "
          f"{len(report['automatically_recovered_text_refs'])} fallback blocks", flush=True)
    return dict(candidate_dir=str(target), chunks=len(chunks),
                preservation_status=preservation["status"], recovery=preservation["chunk_content_recovery"],
                missing_blocks=len(preservation["missing_block_ids"]),
                excessive_duplication_ratio=preservation["excessive_duplication_ratio"],
                legacy_chunks=len(preservation["legacy_provenance_chunk_ids"]),
                recovered_text_blocks=len(report["automatically_recovered_text_refs"]),
                configuration_tables=len(report["automatic_configuration_tables"]),
                detection_before=before["counts"], detection_after=report["failure_detection"]["counts"],
                wall_seconds=round(perf_counter() - began, 3))


if __name__ == "__main__":
    main()
