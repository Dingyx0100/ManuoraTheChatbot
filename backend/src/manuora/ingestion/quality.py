"""Post-parse quality signals and a conservative indexing gate."""

import json
import re
from pathlib import Path

from docling_core.types.doc import DoclingDocument, TableItem

from manuora.ingestion.chunk_context import map_item_context
from manuora.ingestion.files import sha256_file
from manuora.ingestion.profile_schemas import (
    DocumentProfile,
    PageQuality,
    ParsingQualityReport,
    TextCoverageSettings,
)
from manuora.ingestion.text_coverage import measure_text_coverage
from manuora.ingestion.outline_quality import evaluate_outline_quality
from manuora.ingestion.failure_detection import detect_failures


def evaluate_parsing_quality(
    profile: DocumentProfile, parsed_path: Path, sections_path: Path,
    chunk_summary_path: Path | None = None,
    ocr_execution_path: Path | None = None,
    *, source_path: Path | None = None,
    text_coverage_settings: TextCoverageSettings | None = None,
    trusted_outline_path: Path | None = None,
    trust_pdf_bookmarks: bool = False,
) -> ParsingQualityReport:
    """Compare native-PDF signals with Docling output; do not grade table fidelity."""
    document = DoclingDocument.load_from_json(parsed_path)
    if document.num_pages() != profile.page_count:
        raise ValueError("profile and parsed document have different page counts")
    retention = measure_text_coverage(
        profile, document, source_path, text_coverage_settings or TextCoverageSettings(),
    )
    retention_pages = retention.pop("pages")
    sections = [json.loads(line) for line in sections_path.open(encoding="utf-8")]
    parsed_chars = {page: 0 for page in range(1, profile.page_count + 1)}
    text_items = 0
    orphan_items = 0
    duplicates = 0
    table_count = 0
    previous_text = None
    for item, _ in document.iterate_items(traverse_pictures=True):
        if str(item.label) == "picture":
            continue
        table_count += isinstance(item, TableItem)
        # Quality signals only need the text. Rendering every table as Markdown
        # repeats an expensive conversion already done for the review/chunks.
        value = (
            " ".join(cell.text for cell in item.data.table_cells if cell.text)
            if isinstance(item, TableItem) else getattr(item, "text", "")
        )
        if not value or not value.strip():
            continue
        text_items += 1
        pages = {source.page_no for source in getattr(item, "prov", [])}
        if not pages:
            orphan_items += 1
        for page in pages:
            if page in parsed_chars:
                parsed_chars[page] += sum(not char.isspace() for char in value)
        compact = re.sub(r"\s+", " ", value).strip().casefold()
        if len(compact) >= 20 and compact == previous_text:
            duplicates += 1
        previous_text = compact

    page_quality = [
        PageQuality(
            page_number=page.page_number,
            native_body_text_chars=page.body_text_char_count,
            parsed_text_chars=parsed_chars[page.page_number],
            likely_ocr_needed=page.likely_ocr_needed,
            parsed_text_present=(retention_pages[page.page_number]["parsed_coverage_chars"] > 0
                                 if retention_pages[page.page_number]["parsed_coverage_chars"] is not None
                                 else None),
            **retention_pages[page.page_number],
        )
        for page in profile.page_profiles
    ]
    ocr_candidates = [page for page in page_quality if page.likely_ocr_needed]
    ocr_uncovered = [page.page_number for page in ocr_candidates if page.parsed_text_present is False]
    matched = sum(section.get("normalization_status") == "matched" for section in sections)
    heading_rate = matched / len(sections) if sections else None
    contexts = map_item_context(document, sections)
    outline_metrics = evaluate_outline_quality(
        document, contexts, source_sha256=profile.source_sha256,
        source_path=source_path, trusted_outline_path=trusted_outline_path,
        trust_pdf_bookmarks=trust_pdf_bookmarks,
    )
    item_context_rate = (
        sum(bool(context["normalized_heading_path"]) for context in contexts.values()) / len(contexts)
        if contexts else None
    )
    outline_based = any(section.get("normalization_source") == "pdf_outline"
                        for section in sections)
    chunk_rate = None
    if (
        chunk_summary_path is not None
        and chunk_summary_path.is_file()
        and chunk_summary_path.stat().st_mtime >= sections_path.stat().st_mtime
    ):
        summary = json.loads(chunk_summary_path.read_text(encoding="utf-8"))
        if summary.get("source_sha256") == profile.source_sha256:
            adapted = summary.get("results", {}).get("adapted_hybrid", {})
            chunks = adapted.get("chunks", 0)
            statuses = adapted.get("context_status_counts", {})
            chunk_rate = statuses.get("normalized", 0) / chunks if chunks else None

    ocr_execution = None
    if ocr_execution_path is not None and ocr_execution_path.is_file():
        candidate = json.loads(ocr_execution_path.read_text(encoding="utf-8"))
        if (
            candidate.get("source_sha256") == profile.source_sha256
            and candidate.get("parsed_sha256") == sha256_file(parsed_path)
        ):
            ocr_execution = candidate

    status = "PASS"
    reasons = []
    def flag(severity: str, message: str) -> None:
        nonlocal status
        reasons.append(message)
        if severity == "FAIL" or status == "PASS":
            status = severity

    if outline_based and item_context_rate is not None and item_context_rate < 0.1:
        flag("FAIL", "PDF outline leaves nearly all body items without section context")
    elif outline_based and item_context_rate is not None and item_context_rate < 0.8:
        flag("WARN", "PDF outline leaves some body items without section context")
    elif outline_based:
        pass
    elif len(sections) >= 10 and matched == 0:
        flag("FAIL", "No detected section heading has a trusted normalized path")
    elif len(sections) >= 10 and heading_rate is not None and heading_rate < 0.1:
        flag("FAIL", "Fewer than one tenth of detected headings have trusted paths")
    elif heading_rate is not None and heading_rate < 0.25:
        flag("WARN", "Fewer than one quarter of detected headings have trusted paths")
    if outline_based and len(sections) >= 20 and heading_rate is not None and heading_rate < 0.4:
        flag("WARN", "PDF outline matches few body headings; chunk section paths may be coarse")
    if outline_metrics["outline_quality"]["status"] == "FAIL":
        flag("FAIL", "Trusted outline heading coverage is below 70%")
    elif outline_metrics["outline_quality"]["status"] == "WARN":
        flag("WARN", "Independent outline entries are missing or ambiguous in parsed content")
    if outline_metrics["section_specificity_details"]["status"] == "WARN":
        flag("WARN", "Body section paths are coarse, unassigned or on a different branch from the independent outline")
    if retention["text_coverage_failed_pages"]:
        flag("FAIL", "Native text character/token count retention is below the failure threshold on pages "
             + str(retention["text_coverage_failed_pages"]))
    if retention["text_coverage_warning_pages"]:
        flag("WARN", "Native text character/token count retention is below the warning threshold on pages "
             + str(retention["text_coverage_warning_pages"]))
    if retention["text_coverage_unmeasured_pages"]:
        flag("WARN", "Native text count retention is not fully measurable on pages "
             + str(retention["text_coverage_unmeasured_pages"]))
    uncovered_ratio = len(ocr_uncovered) / profile.page_count if profile.page_count else 0
    if uncovered_ratio >= 0.2:
        flag("FAIL", "At least 20% of pages are uncovered OCR candidates")
    elif ocr_uncovered:
        flag("WARN", "OCR-candidate pages have no searchable parsed text")
    if any(page.parsed_text_present is None for page in ocr_candidates):
        flag("WARN", "OCR-candidate text presence cannot be assigned reliably to pages")
    if ocr_candidates and ocr_execution is None:
        flag("WARN", "OCR execution record is missing or does not match parsed.json")
    if ocr_execution is not None and not ocr_execution["complete"]:
        flag("WARN", "Only part of the planned OCR page set was processed")
    if chunk_rate is not None and chunk_rate < 0.1:
        flag("FAIL", "Fewer than one tenth of adapted Hybrid chunks have resolved context")
    elif chunk_rate is not None and chunk_rate < 0.8:
        flag("WARN", "Fewer than 80% of adapted Hybrid chunks have resolved context")
    if not sections:
        flag("WARN", "No section headings were detected")
    if table_count:
        flag("WARN", "Table fidelity has not been verified")
    detection = detect_failures(document, source_path=source_path, parsed_path=parsed_path)
    if detection["findings"]:
        flag("WARN", "Automatic structural checks found potential extraction/propagation problems")

    return ParsingQualityReport(
        schema_version=4,
        source_sha256=profile.source_sha256,
        parsed_sha256=sha256_file(parsed_path),
        status=status,
        retrieval_ready=status == "PASS",
        page_count=profile.page_count,
        **retention,
        ocr_candidate_pages=len(ocr_candidates),
        ocr_candidate_uncovered_pages=ocr_uncovered,
        ocr_execution_complete=ocr_execution["complete"] if ocr_execution is not None else None,
        ocr_accepted_pages=ocr_execution["accepted_page_count"] if ocr_execution is not None else None,
        section_count=len(sections),
        matched_section_count=matched,
        heading_resolution_rate=round(heading_rate, 4) if heading_rate is not None else None,
        **outline_metrics,
        body_item_context_rate=round(item_context_rate, 4) if item_context_rate is not None else None,
        orphan_block_ratio=round(orphan_items / text_items, 4) if text_items else None,
        duplicate_text_ratio=round(duplicates / text_items, 4) if text_items else None,
        chunk_context_resolution_rate=round(chunk_rate, 4) if chunk_rate is not None else None,
        parsed_table_count=table_count,
        reasons=reasons,
        page_quality=page_quality,
        failure_detection=detection,
    )
