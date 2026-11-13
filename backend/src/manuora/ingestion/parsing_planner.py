"""Turn measured PDF traits into an advisory, reproducible parsing plan."""

from manuora.ingestion.profile_schemas import DocumentProfile, ParsingPlan


def create_parsing_plan(profile: DocumentProfile) -> ParsingPlan:
    """Recommend strategies without silently changing the Docling run."""
    ocr_pages = [page.page_number for page in profile.page_profiles if page.likely_ocr_needed]
    toc_pages = [page for page in profile.page_profiles if page.likely_toc]
    number_title_page_rows = sum(page.toc_numbered_rows for page in toc_pages)
    if not toc_pages:
        toc_mode = "ignore"
    elif number_title_page_rows >= 3:
        toc_mode = "multi_column"
    else:
        toc_mode = "standard"
    headings = ["parser_labels"]
    if toc_mode == "standard":
        headings.append("toc_section_numbers")
    if profile.numbered_heading_candidate_count >= 5:
        headings.append("numbered_headings")
    warnings = ["Heading and TOC recommendations are advisory; parse_document executes planned OCR separately."]
    if ocr_pages:
        warnings.append("OCR output is admitted only after a per-picture readability check.")
    if toc_mode == "multi_column":
        warnings.append("Number/title/page TOC rows need a dedicated matcher before using them as hierarchy anchors.")
    return ParsingPlan(
        source_sha256=profile.source_sha256,
        ocr_mode="selective" if ocr_pages else "off",
        ocr_pages=ocr_pages,
        toc_mode=toc_mode,
        heading_strategy=headings,
        warnings=warnings,
    )
