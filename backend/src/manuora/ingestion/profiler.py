"""Cheap PDF diagnostics before Docling conversion.

Signals here are candidates, not confirmed semantic labels. In particular,
native PDF text can include running headers while the body is a screenshot.
"""

import re
from importlib.metadata import version
from pathlib import Path

import pypdfium2 as pdfium

from manuora.ingestion.files import sha256_file
from manuora.ingestion.profile_schemas import (
    DocumentProfile,
    PageProfile,
    ProfilingThresholds,
)


NUMBERED_LINE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+\S")
TOC_NUMBERED_ROW = re.compile(r"^\s*\d+(?:\.\d+)*\s+.+\s+\d+\s*$")
DOT_LEADER = re.compile(r"(?:\.\s*){4,}\d+\s*$")


def _char_count(value: str) -> int:
    return sum(not character.isspace() for character in value)


def _clipped_box(bounds: tuple[float, ...], width: float, height: float) -> tuple[float, ...] | None:
    left, bottom, right, top = bounds
    box = (max(0.0, left), max(0.0, bottom), min(width, right), min(height, top))
    return box if box[2] > box[0] and box[3] > box[1] else None


def calculate_image_area_ratio(
    boxes: list[tuple[float, ...]], page_width: float, page_height: float
) -> float:
    """Measure the union of image rectangles, avoiding overlap double counts."""
    if page_width <= 0 or page_height <= 0:
        return 0.0
    clipped = [box for raw in boxes if (box := _clipped_box(raw, page_width, page_height))]
    edges = sorted({coordinate for box in clipped for coordinate in (box[0], box[2])})
    area = 0.0
    for left, right in zip(edges, edges[1:]):
        intervals = sorted(
            (box[1], box[3]) for box in clipped if box[0] < right and box[2] > left
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for bottom, top in intervals[1:]:
                if bottom > end:
                    covered += end - start
                    start, end = bottom, top
                else:
                    end = max(end, top)
            covered += end - start
        area += (right - left) * covered
    return min(1.0, area / (page_width * page_height))


def is_likely_blank_page(
    body_text_chars: int, image_ratio: float, drawing_ratio: float,
    thresholds: ProfilingThresholds,
) -> bool:
    return (
        body_text_chars < 10
        and image_ratio < thresholds.blank_image_ratio
        and drawing_ratio < thresholds.blank_drawing_ratio
    )


def is_image_heavy_page(image_ratio: float, thresholds: ProfilingThresholds) -> bool:
    return image_ratio >= thresholds.image_heavy_ratio


def needs_ocr(
    *, body_text_chars: int, image_ratio: float, likely_blank: bool,
    thresholds: ProfilingThresholds,
) -> bool:
    """Flag low-body-text raster pages; this does not execute OCR."""
    return (
        not likely_blank
        and body_text_chars < thresholds.min_body_text_chars
        and image_ratio >= thresholds.ocr_image_ratio
    )


def estimate_column_count(
    text_objects: list[tuple[str, tuple[float, ...]]], page_width: float,
    page_height: float, margin_ratio: float,
) -> int | None:
    """Find repeated side-by-side text bands; tables may still mimic columns."""
    body = [
        box
        for text, box in text_objects
        if _char_count(text) >= 15
        and page_height * margin_ratio < (box[1] + box[3]) / 2 < page_height * (1 - margin_ratio)
        and page_width * 0.18 < box[2] - box[0] < page_width * 0.47
    ]
    if not text_objects:
        return None
    left = [box for box in body if box[2] < page_width * 0.51]
    right = [box for box in body if box[0] > page_width * 0.49]
    if len(left) >= 4 and len(right) >= 4:
        paired = sum(
            any(
                a[2] < b[0] and min(a[3], b[3]) > max(a[1], b[1])
                for b in right
            )
            for a in left
        )
        if paired >= 3:
            return 2
    return 1


def detect_toc_page(lines: list[str]) -> tuple[bool, int]:
    heading = any(line.strip().casefold() in {"contents", "table of contents"} for line in lines)
    numbered_rows = sum(bool(TOC_NUMBERED_ROW.match(line)) for line in lines)
    leaders = sum(bool(DOT_LEADER.search(line)) for line in lines)
    isolated_numbers = sum(bool(re.fullmatch(r"\d+(?:\.\d+)+", line)) for line in lines)
    return heading or (numbered_rows >= 4 and leaders >= 3) or isolated_numbers >= 8, numbered_rows


def extract_page_statistics(page, page_number: int, thresholds: ProfilingThresholds) -> PageProfile:
    width, height = page.get_size()
    text_page = page.get_textpage()
    native_text = text_page.get_text_range()
    text_objects = []
    image_boxes = []
    drawing_boxes = []
    for obj in page.get_objects(textpage=text_page):
        box = _clipped_box(obj.get_bounds(), width, height)
        if box is None:
            continue
        if isinstance(obj, pdfium.PdfTextObj):
            text_objects.append((obj.extract(), box))
        elif isinstance(obj, pdfium.PdfImage):
            image_boxes.append(box)
        else:
            drawing_boxes.append(box)
    body_chars = sum(
        _char_count(text)
        for text, box in text_objects
        if height * thresholds.body_margin_ratio
        < (box[1] + box[3]) / 2
        < height * (1 - thresholds.body_margin_ratio)
    )
    image_ratio = calculate_image_area_ratio(image_boxes, width, height)
    drawing_ratio = calculate_image_area_ratio(drawing_boxes, width, height)
    blank = is_likely_blank_page(body_chars, image_ratio, drawing_ratio, thresholds)
    lines = [line.strip() for line in native_text.splitlines() if line.strip()]
    toc, numbered_rows = detect_toc_page(lines)
    numbered_lines = sum(
        bool(NUMBERED_LINE.match(line)) and len(line) <= 160 for line in lines
    )
    text_page.close()
    return PageProfile(
        page_number=page_number,
        text_char_count=_char_count(native_text),
        body_text_char_count=body_chars,
        text_object_count=len(text_objects),
        image_count=len(image_boxes),
        image_area_ratio=round(image_ratio, 4),
        drawing_count=len(drawing_boxes),
        drawing_area_ratio=round(drawing_ratio, 4),
        column_count=estimate_column_count(
            text_objects, width, height, thresholds.body_margin_ratio
        ),
        toc_numbered_rows=numbered_rows,
        numbered_line_count=numbered_lines,
        likely_blank=blank,
        likely_image_heavy=is_image_heavy_page(image_ratio, thresholds),
        likely_ocr_needed=needs_ocr(
            body_text_chars=body_chars, image_ratio=image_ratio,
            likely_blank=blank, thresholds=thresholds,
        ),
        likely_toc=toc,
    )


def profile_document(
    path: Path, *, product_id: str | None = None,
    product_version: str | None = None, document_id: str | None = None,
    thresholds: ProfilingThresholds | None = None,
) -> DocumentProfile:
    """Profile a PDF without invoking Docling or OCR."""
    thresholds = thresholds or ProfilingThresholds()
    document = pdfium.PdfDocument(path)
    try:
        pages = []
        for index in range(len(document)):
            page = document[index]
            try:
                pages.append(extract_page_statistics(page, index + 1, thresholds))
            finally:
                page.close()
    finally:
        document.close()
    count = len(pages)
    ratio = lambda predicate: round(sum(predicate(page) for page in pages) / count, 4) if count else 0.0
    toc_pages = [page.page_number for page in pages if page.likely_toc]
    body_pages = [page for page in pages if not page.likely_toc]
    numbered = sum(page.numbered_line_count for page in body_pages)
    warnings = []
    ocr_count = sum(page.likely_ocr_needed for page in pages)
    if ocr_count:
        warnings.append(f"{ocr_count} pages have little body text and substantial raster coverage")
    if toc_pages:
        warnings.append(f"TOC-like layout on pages {toc_pages}")
    if numbered:
        warnings.append("Numbered lines are candidates; they are not confirmed headings")
    return DocumentProfile(
        source_sha256=sha256_file(path),
        source_bytes=path.stat().st_size,
        pdf_engine=f"pypdfium2 {version('pypdfium2')}",
        file_name=path.name,
        product_id=product_id,
        product_version=product_version,
        document_id=document_id,
        page_count=count,
        native_text_page_ratio=ratio(
            lambda page: page.body_text_char_count >= thresholds.min_body_text_chars
        ),
        image_heavy_page_ratio=ratio(lambda page: page.likely_image_heavy),
        blank_page_ratio=ratio(lambda page: page.likely_blank),
        ocr_candidate_page_ratio=ratio(lambda page: page.likely_ocr_needed),
        multi_column_page_ratio=ratio(lambda page: page.column_count == 2),
        toc_detected=bool(toc_pages),
        toc_pages=toc_pages,
        numbered_heading_candidate_count=numbered,
        numbered_heading_candidate_page_ratio=(
            round(sum(page.numbered_line_count > 0 for page in body_pages) / len(body_pages), 4)
            if body_pages else 0.0
        ),
        thresholds=thresholds,
        page_profiles=pages,
        warnings=warnings,
    )
