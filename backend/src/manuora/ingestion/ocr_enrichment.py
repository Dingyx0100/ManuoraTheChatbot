"""Recover searchable text from selected PDF pictures with a conservative gate."""

import re
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from docling.datamodel.pipeline_options import OcrMode, RapidOcrOptions
from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument, ImageRefMode

from manuora.ingestion.docling_parser import create_pdf_converter
from manuora.ingestion.files import sha256_file
from manuora.ingestion.profile_schemas import ParsingPlan


def contiguous_ranges(pages: list[int]) -> list[tuple[int, int]]:
    """Group selected one-based pages for Docling's contiguous page_range API."""
    ranges = []
    for page in sorted(set(pages)):
        if not ranges or page > ranges[-1][1] + 1:
            ranges.append((page, page))
        else:
            ranges[-1] = (ranges[-1][0], page)
    return ranges


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def picture_match(source_picture, base_pictures):
    """Find the same picture in the full parse by page and bounding box."""
    if not source_picture.prov:
        return None
    source = source_picture.prov[0]
    candidates = []
    for picture in base_pictures:
        if not picture.prov or picture.prov[0].page_no != source.page_no:
            continue
        box = picture.prov[0].bbox
        other = source.bbox
        if box.coord_origin != other.coord_origin:
            continue
        distance = sum(abs(getattr(box, edge) - getattr(other, edge)) for edge in ("l", "t", "r", "b"))
        candidates.append((distance, picture))
    if not candidates:
        return None
    distance, picture = min(candidates, key=lambda pair: pair[0])
    return picture if distance <= 8 else None


def assess_ocr_lines(lines: list) -> tuple[list, dict]:
    """Keep readable lines and reject pictures made mostly of isolated marks."""
    readable = []
    for item in lines:
        value = item.text.strip()
        nonspace = sum(not char.isspace() for char in value)
        alphabetic = sum(char.isalpha() for char in value)
        if nonspace >= 4 and alphabetic >= 4 and alphabetic / nonspace >= 0.25:
            readable.append(item)
    informative = [
        item for item in readable
        if len(re.findall(r"[A-Za-z]{2,}", item.text)) >= 2
    ]
    informative_chars = sum(len(item.text.strip()) for item in informative)
    accepted = len(informative) >= 4 and informative_chars >= 100
    return (readable if accepted else []), {
        "ocr_line_count": len(lines),
        "readable_line_count": len(readable),
        "informative_line_count": len(informative),
        "informative_chars": informative_chars,
        "accepted": accepted,
        "reason": "readable_screenshot_text" if accepted else "insufficient_readable_text",
    }


def enrich_document_page(base: DoclingDocument, ocr: DoclingDocument, page: int) -> dict:
    """Move accepted OCR lines from picture children into chunkable body siblings."""
    native = Counter(
        compact(item.text) for item in base.texts
        if any(prov.page_no == page for prov in item.prov) and item.text.strip()
    )
    source_pictures = {picture.self_ref: picture for picture in ocr.pictures}
    base_pictures = [picture for picture in base.pictures if any(prov.page_no == page for prov in picture.prov)]
    grouped = defaultdict(list)
    for item in ocr.texts:
        if (
            item.content_layer == ContentLayer.BODY
            and item.parent is not None
            and item.parent.cref in source_pictures
            and any(prov.page_no == page for prov in item.prov)
        ):
            key = compact(item.text)
            if native[key]:
                native[key] -= 1
            elif key:
                grouped[item.parent.cref].append(item)

    picture_results = []
    added_refs = []
    for picture_ref, lines in grouped.items():
        source_picture = source_pictures[picture_ref]
        anchor = picture_match(source_picture, base_pictures)
        accepted_lines, assessment = assess_ocr_lines(lines)
        if anchor is None:
            assessment.update(accepted=False, reason="no_matching_picture")
            accepted_lines = []
        else:
            sibling = anchor
            for item in accepted_lines:
                sibling = base.insert_text(
                    sibling=sibling,
                    label=DocItemLabel.TEXT,
                    text=item.text,
                    orig=item.orig,
                    prov=item.prov[0].model_copy(deep=True),
                    content_layer=ContentLayer.BODY,
                )
                added_refs.append(sibling.self_ref)
        picture_results.append({
            "source_picture_ref": picture_ref,
            "anchor_ref": anchor.self_ref if anchor else None,
            "added_text_refs": added_refs[-len(accepted_lines):] if accepted_lines else [],
            **assessment,
        })
    return {
        "page_number": page,
        "picture_results": picture_results,
        "added_text_refs": added_refs,
        "added_line_count": len(added_refs),
        "accepted": bool(added_refs),
    }


def apply_ocr_plan(
    source_path: Path,
    parsed_path: Path,
    plan: ParsingPlan,
    selected_pages: list[int] | None = None,
    previous_execution: dict | None = None,
    language: str = "en",
) -> dict:
    """Execute selected OCR passes and replace parsed.json only after validation."""
    source_sha256 = sha256_file(source_path)
    if plan.source_sha256 != source_sha256:
        raise ValueError("parsing plan source hash differs from PDF")
    if plan.ocr_mode == "full":
        raise NotImplementedError("full-document OCR is not implemented by the selective enrichment step")
    planned = sorted(set(plan.ocr_pages)) if plan.ocr_mode == "selective" else []
    requested = planned if selected_pages is None else sorted(set(selected_pages))
    if not set(requested) <= set(planned):
        raise ValueError("selected OCR pages must be listed in the parsing plan")
    if requested and language.lower().split("-", 1)[0] != "en":
        raise ValueError("the current OCR engine and readability gate support English manuals only")

    started = perf_counter()
    base = DoclingDocument.load_from_json(parsed_path)
    if any(page < 1 or page > base.num_pages() for page in requested):
        raise ValueError("OCR page is outside the parsed document")
    prior_results = []
    if previous_execution is not None:
        if previous_execution.get("source_sha256") != source_sha256:
            raise ValueError("previous OCR execution source hash differs from PDF")
        if previous_execution.get("planned_pages") != planned:
            raise ValueError("previous OCR execution used a different page plan")
        prior_hash = previous_execution.get("parsed_sha256")
        if prior_hash is not None and prior_hash != sha256_file(parsed_path):
            raise ValueError("parsed document changed since the previous OCR execution")
        prior_results = previous_execution.get("page_results", [])
        for result in prior_results:
            for ref in result["added_text_refs"]:
                index = int(ref.rsplit("/", 1)[1])
                if index >= len(base.texts) or base.texts[index].self_ref != ref:
                    raise ValueError("previous OCR execution refs are absent from parsed document")
    previously_processed = {result["page_number"] for result in prior_results}
    to_process = sorted(set(requested) - previously_processed)
    new_results = []
    if to_process:
        options = RapidOcrOptions(
            lang=["iso:en"],
            mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS,
            backend="torch",
        )
        converter = create_pdf_converter(options)
        for first, last in contiguous_ranges(to_process):
            conversion = converter.convert(source_path, page_range=(first, last))
            ocr_document = conversion.document
            for page in range(first, last + 1):
                new_results.append(enrich_document_page(base, ocr_document, page))

    added_now = sum(result["added_line_count"] for result in new_results)
    if added_now:
        temporary = parsed_path.with_name("parsed.ocr_pending.json")
        base.save_as_json(temporary, image_mode=ImageRefMode.PLACEHOLDER)
        checked = DoclingDocument.load_from_json(temporary)
        if checked.num_pages() != base.num_pages():
            raise ValueError("OCR enrichment changed the document page count")
        if not all(
            any(prov.page_no == result["page_number"] for prov in checked.texts[int(ref.rsplit("/", 1)[1])].prov)
            for result in new_results for ref in result["added_text_refs"]
        ):
            raise ValueError("OCR text lost its page provenance")
        temporary.replace(parsed_path)

    results = sorted([*prior_results, *new_results], key=lambda result: result["page_number"])
    processed = [result["page_number"] for result in results]
    return {
        "source_sha256": source_sha256,
        "docling_version": version("docling"),
        "rapidocr_version": version("rapidocr"),
        "ocr_engine": "RapidOCR torch CPU",
        "ocr_mode": "pdf_aware_layout_regions",
        "ocr_language": "iso:en",
        "planned_pages": planned,
        "requested_pages": requested,
        "newly_processed_pages": to_process,
        "selected_pages": processed,
        "complete": processed == planned,
        "added_line_count": sum(result["added_line_count"] for result in results),
        "accepted_page_count": sum(result["accepted"] for result in results),
        "wall_seconds": round(perf_counter() - started, 3),
        "parsed_sha256": sha256_file(parsed_path),
        "page_results": results,
    }
