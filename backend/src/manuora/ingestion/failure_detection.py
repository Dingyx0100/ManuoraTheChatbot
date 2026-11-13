"""Model-free pre-embedding diagnostics. Findings are suspicions, not accuracy scores."""

import json
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import pypdfium2 as pdfium
from pypdfium2 import raw as pdfium_c

from manuora.ingestion.files import sha256_file

DETECTOR_VERSION = "1"
WRAPPED_IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*-\s+[A-Za-z][A-Za-z0-9_]*(?:-\s+[A-Za-z][A-Za-z0-9_]*)*")


def configuration_header(table):
    """Recognize a complete, unambiguous header from cells, not a table spec."""
    if table.data.num_cols != 3:
        return None
    for row in range(min(4, table.data.num_rows)):
        cells = [c for c in table.data.table_cells if c.start_row_offset_idx == row]
        by_col = {c.start_col_offset_idx: c for c in cells if c.col_span == c.row_span == 1}
        if len(by_col) == 3 and [by_col[i].text.strip().casefold() for i in range(3)] == ["variable", "default", "meaning"]:
            return row
    return None


def wrapped_identifier_candidate(text, configuration=False):
    if not WRAPPED_IDENTIFIER.search(text):
        return False
    joined = re.sub(r"-\s+", "", text.strip())
    # Ordinary prose in the first column also wraps. Require a single lexical
    # identifier and code-like casing, unless its header explicitly says Variable.
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", joined) and
                (configuration or "_" in joined or re.search(r"[a-z][A-Z]", joined)))


def configuration_text(table):
    """Serialize recognized columns explicitly; preserve uncertain identifier spelling.

    Continuation rows without a key remain explicit fragments. No inferred values,
    blanket dehyphenation, or cross-row reassignment is performed.
    """
    header = configuration_header(table)
    if header is None:
        return None
    lines = []
    banners = list(dict.fromkeys(c.text.strip() for c in table.data.table_cells
                                if c.start_row_offset_idx < header and c.text.strip()))
    lines.extend(banners)
    lines.append("Columns: Variable; Default; Meaning.")
    for row in range(header + 1, table.data.num_rows):
        cells = [c for c in table.data.table_cells if c.start_row_offset_idx == row]
        if any(c.col_span != 1 or c.row_span != 1 for c in cells):
            return None  # Leave complex merged structures to the existing serializer.
        by_col = {c.start_col_offset_idx: c.text.strip() for c in cells}
        if len(by_col) != len(cells):
            return None
        if any(by_col.values()):
            lines.append("; ".join(f"{label}: {by_col[col]}" for col, label in enumerate(
                ("Variable", "Default", "Meaning")) if by_col.get(col)))
    return "\n".join(lines)


def _bounds(box, height):
    if str(box.coord_origin).lower().endswith("topleft"):
        return box.l, height - box.b, box.r, height - box.t
    return box.l, box.b, box.r, box.t


def read_confidence(parsed_path: Path, source_hash: str, parsed_hash: str):
    path = parsed_path.with_suffix(".confidence.json")
    if not path.is_file():
        return {"status": "unavailable", "reason": "conversion confidence was not saved"}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("source_sha256") != source_hash or data.get("parsed_sha256") != parsed_hash:
        return {"status": "stale", "reason": "confidence belongs to another source/parse"}
    return {"status": "available", **data}


def detect_failures(document, *, source_path: Path | None = None,
                    parsed_path: Path | None = None, chunks: list[dict] | None = None,
                    confidence: dict | None = None):
    """Inspect saved structure and native geometry without OCR or rendering.

    All tables receive cheap checks. Native PDF geometry is fetched only for
    candidate merged rows or empty columns; high Docling confidence never skips
    these checks. Missing confidence is unknown, not a quality failure.
    """
    started = perf_counter()
    findings = []

    def flag(code, ref, pages, evidence, severity="WARN"):
        findings.append(dict(code=code, block_id=ref, pages=sorted(pages),
                             severity=severity, evidence=evidence))

    if confidence is None and source_path and parsed_path:
        confidence = read_confidence(parsed_path, sha256_file(source_path), sha256_file(parsed_path))
    confidence = confidence or {"status": "unavailable"}
    if confidence.get("status") == "available":
        for page, scores in confidence.get("scores", {}).get("pages", {}).items():
            if scores.get("low_grade") in {"poor", "fair"}:
                flag("low_docling_confidence", None, [int(page)], scores)

    nodes = {i.self_ref: i for i in [*document.texts, *document.tables, *document.pictures, *document.groups]}
    linked = None if chunks is None else {ref for c in chunks for ref in c.get("source_block_ids", c.get("refs", []))}
    nested = []
    for item in document.texts:
        if item.content_layer.value != "body" or not item.text.strip():
            continue
        parent, seen, inside_picture, inside_table = item.parent, set(), False, False
        while parent and parent.cref not in seen:
            seen.add(parent.cref)
            inside_picture |= parent.cref.startswith("#/pictures/")
            inside_table |= parent.cref.startswith("#/tables/")
            node = nodes.get(parent.cref)
            parent = node.parent if node else None
        if inside_picture and not inside_table:
            nested.append(item.self_ref)
            if linked is None or item.self_ref not in linked:
                flag("nested_text_at_risk" if linked is None else "nested_text_missing_from_chunks",
                     item.self_ref, {p.page_no for p in item.prov},
                     {"label": str(item.label), "excerpt": item.text[:180]},
                     "WARN" if linked is None else "FAIL")

    pdf = pdfium.PdfDocument(source_path) if source_path else None
    native_pages = set()
    try:
        for table in document.tables:
            pages = {p.page_no for p in table.prov}
            ref = table.self_ref
            cells = table.data.table_cells
            header = configuration_header(table)
            if header is not None and header > 0:
                flag("table_banner_before_column_headers", ref, pages, {"header_row": header})
            wrapped = [c.text for c in cells if c.start_col_offset_idx == 0
                       and wrapped_identifier_candidate(c.text, header is not None)]
            if wrapped:
                flag("wrapped_identifier_candidate", ref, pages, {"examples": wrapped[:5], "count": len(wrapped)})
            footer = [c.text for c in cells if re.match(r"^(Chapter|Page)\s+\d+\.?$", c.text.strip(), re.I)
                      and c.start_row_offset_idx >= table.data.num_rows - 2]
            if footer:
                flag("table_footer_contamination", ref, pages, {"cells": footer})
            if len(pages) != 1:
                flag("multipage_table_attribution" if pages else "table_missing_provenance",
                     ref, pages, {"page_count": len(pages)})
                continue
            page_no = next(iter(pages))
            height = document.pages[page_no].size.height
            box = _bounds(table.prov[0].bbox, height)
            if box[1] < 0 or box[3] > height:
                flag("table_outside_page", ref, pages, {"bounds": box, "page_height": height})
            headers = [c for c in cells if c.start_row_offset_idx == 0 and c.col_span == 1 and c.bbox]
            empty_cols = [c for c in headers if c.text.strip() and not any(
                x.text.strip() and x.start_row_offset_idx > 0
                and x.start_col_offset_idx <= c.start_col_offset_idx < x.end_col_offset_idx for x in cells)]
            two_row = table.data.num_rows == 2 and table.data.num_cols >= 2
            if not pdf or not (empty_cols or two_row):
                continue
            native_pages.add(page_no)
            page = pdf[page_no - 1]
            try:
                if two_row:
                    data_cells = [c for c in cells if c.start_row_offset_idx == 1 and c.bbox and c.col_span == 1]
                    textpage = page.get_textpage()
                    try:
                        line_counts = []
                        for cell in sorted(data_cells, key=lambda c: c.start_col_offset_idx)[:2]:
                            lines = textpage.get_text_bounded(*_bounds(cell.bbox, height)).splitlines()
                            line_counts.append(sum(bool(line.strip()) for line in lines))
                        if len(line_counts) == 2 and min(line_counts) >= 3:
                            flag("possible_collapsed_table_rows", ref, pages,
                                 {"parsed_data_rows": 1, "native_lines_first_two_cells": line_counts})
                    finally:
                        textpage.close()
                if empty_cols:
                    images = [obj.get_bounds() for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])]
                    for cell in empty_cols:
                        cb = _bounds(cell.bbox, height)
                        following = [h.bbox.l for h in headers if h.start_col_offset_idx > cell.start_col_offset_idx]
                        right = min(following) if following else box[2]
                        contained = [b for b in images if cb[0] - 3 <= (b[0] + b[2]) / 2 < right
                                     and box[1] <= (b[1] + b[3]) / 2 < cb[1]
                                     and b[2] - b[0] < 80 and b[3] - b[1] < 80]
                        if contained:
                            flag("unrepresented_table_symbols", ref, pages,
                                 {"column": cell.text, "embedded_images": len(contained)})
            finally:
                page.close()
    finally:
        if pdf:
            pdf.close()
    return dict(detector_version=DETECTOR_VERSION, confidence=confidence,
                tables_checked=len(document.tables), nested_text_blocks=len(nested),
                native_geometry_pages=len(native_pages), findings=findings,
                counts=dict(Counter(f["code"] for f in findings)),
                status="FAIL" if any(f["severity"] == "FAIL" for f in findings) else "WARN" if findings else "PASS",
                wall_seconds=round(perf_counter() - started, 4),
                limitations=["Structural warnings are heuristic, not calibrated error probabilities.",
                             "Vector-only symbols and plausible wrong cell associations can be missed.",
                             "A table outside the crop or text absent from the PDF cannot be reconstructed here."])
