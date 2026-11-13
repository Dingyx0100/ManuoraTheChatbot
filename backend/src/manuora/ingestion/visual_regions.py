"""Inventory PDF visual regions and choose text, visual, or both retrieval routes.

This is a deterministic, inspectable first-pass classifier. Coordinates are
PDF points in bottom-left origin. It does not render, embed, or alter chunks.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import pypdfium2 as pdfium

from manuora.ingestion.files import sha256_file


SCHEMA_VERSION = 2
CLASSIFIER_VERSION = "2"
SMALL_REGION_AREA = 0.012
LARGE_PICTURE_AREA = 0.025
CONTEXT_GAP_POINTS = 28.0


def pdf_box(raw: dict, height: float) -> tuple[float, float, float, float] | None:
    """Convert a Docling box to PDF bottom-left coordinates."""
    if not raw:
        return None
    if raw.get("coord_origin") == "TOPLEFT":
        box = (raw["l"], height - raw["b"], raw["r"], height - raw["t"])
    else:
        box = (raw["l"], raw["b"], raw["r"], raw["t"])
    return tuple(map(float, box)) if box[2] > box[0] and box[3] > box[1] else None


def overlap(a: tuple, b: tuple) -> float:
    """Intersection as a fraction of the smaller rectangle."""
    area = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return area / smaller if smaller > 0 else 0.0


def clipped(box: tuple, width: float, height: float) -> tuple | None:
    result = (max(0.0, box[0]), max(0.0, box[1]),
              min(width, box[2]), min(height, box[3]))
    return result if result[2] > result[0] and result[3] > result[1] else None


def expanded(box: tuple, width: float, height: float, margin: float = 4.0) -> tuple:
    return clipped((box[0]-margin, box[1]-margin, box[2]+margin, box[3]+margin),
                   width, height) or box


def compact(box: tuple) -> list[float]:
    return [round(value, 2) for value in box]


def text_context(box: tuple, text_items: list[tuple],
                 *, limit: int = 220) -> tuple[str, tuple | None]:
    """Find nearby labels and a same-line box suitable for an icon crop."""
    candidates = []
    for text_box, value, label in text_items:
        if label in {"page_header", "page_footer"} or not value.strip():
            continue
        vertical_gap = max(0, text_box[1]-box[3], box[1]-text_box[3])
        horizontal_gap = max(0, text_box[0]-box[2], box[0]-text_box[2])
        if vertical_gap <= CONTEXT_GAP_POINTS and horizontal_gap <= 90:
            candidates.append((vertical_gap * 2 + horizontal_gap, vertical_gap,
                               value.strip(), text_box))
    candidates.sort(key=lambda item: item[0])
    context = " | ".join(value for _, _, value, _ in candidates[:2])[:limit]
    same_line_box = next((item[3] for item in candidates if item[1] <= 6), None)
    return context, same_line_box


def with_label(box: tuple, label_box: tuple | None,
               width: float, height: float) -> tuple:
    if label_box is None:
        return box
    union = (min(box[0], label_box[0]), min(box[1], label_box[1]),
             max(box[2], label_box[2]), max(box[3], label_box[3]))
    return expanded(union, width, height, 4)


def table_rows(table: dict, table_box: tuple, height: float) -> list[dict]:
    """Derive row bands from Docling cell positions, including empty icon cells."""
    data = table.get("data") or {}
    rows = int(data.get("num_rows") or 0)
    if rows <= 0:
        return []
    centers: dict[int, list[float]] = defaultdict(list)
    values: dict[int, list[str]] = defaultdict(list)
    for cell in data.get("table_cells", []):
        start = int(cell.get("start_row_offset_idx") or 0)
        end = min(rows, int(cell.get("end_row_offset_idx") or start + 1))
        value = (cell.get("text") or "").strip()
        for row in range(start, end):
            if value:
                values[row].append(value)
            box = pdf_box(cell.get("bbox"), height)
            if box and end == start + 1:
                centers[row].append((box[1] + box[3]) / 2)
    if not centers:
        return []
    observed = {row: sum(ys)/len(ys) for row, ys in centers.items()}
    first, last = min(observed), max(observed)
    step = (observed[first]-observed[last]) / max(1, last-first)
    row_y = [observed.get(row, observed[first] - (row-first)*step) for row in range(rows)]
    boundaries = [table_box[3]] + [(a+b)/2 for a, b in zip(row_y, row_y[1:])] + [table_box[1]]
    output = []
    for row in range(rows):
        top, bottom = boundaries[row], boundaries[row+1]
        if top <= bottom:
            continue
        output.append({"row": row, "bbox": (table_box[0], bottom, table_box[2], top),
                       "text": " | ".join(dict.fromkeys(values[row]))[:500]})
    return output


def make_region(document_id: str, page: int, suffix: str, kind: str, box: tuple,
                route: str, reason: str, context: str, refs: list[str],
                image_count: int, width: float, height: float,
                parent: str | None = None, content_box: tuple | None = None) -> dict:
    source_box = content_box or box
    area = (source_box[2]-source_box[0])*(source_box[3]-source_box[1])/(width*height)
    visual_priority = ("none" if route == "text" else "high" if kind in
                       {"table", "icon_table_row"} or area >= LARGE_PICTURE_AREA else
                       "medium" if context and area >= 0.001 else "low")
    return {
        "schema_version": SCHEMA_VERSION, "region_id": f"{document_id}:p{page:04d}:{suffix}",
        "document_id": document_id, "page": page, "kind": kind,
        "bbox_pdf": compact(box), "content_bbox_pdf": compact(source_box),
        "page_size_pdf": [round(width, 2), round(height, 2)],
        "route": route, "visual_priority": visual_priority,
        "reason": reason, "context_text": context,
        "source_refs": refs, "parent_region_id": parent,
        "signals": {"area_ratio": round(area, 5), "pdf_image_count": image_count},
    }


def load_section_anchors(path: Path | None, parsed: dict) -> list[dict]:
    if path is None or not path.is_file():
        return []
    anchors = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if not item.get("normalized_heading_path"):
                continue
            if item.get("anchor_y") is None and item.get("ref", "").startswith("#/texts/"):
                index = int(item["ref"].rsplit("/", 1)[-1])
                text = parsed["texts"][index]
                prov = next((p for p in text.get("prov", [])
                             if p.get("page_no") == item["page_no"]), None)
                if prov:
                    page = parsed["pages"][str(item["page_no"])]
                    box = pdf_box(prov.get("bbox"), page["size"]["height"])
                    if box:
                        item["anchor_y"] = box[3]
            if item.get("anchor_y") is not None:
                anchors.append(item)
    return sorted(anchors, key=lambda item: (item["page_no"], -item["anchor_y"]))


def section_for(page: int, box: tuple, anchors: list[dict], keys: list[tuple]) -> dict | None:
    index = bisect_right(keys, (page, -box[3])) - 1
    if index < 0:
        return None
    item = anchors[index]
    return {"ref": item["ref"], "number": item.get("section_number"),
            "heading_path": item["normalized_heading_path"]}


def inventory_regions(source: Path, parsed_path: Path, output_path: Path,
                      document_id: str, selected_pages: set[int] | None = None,
                      sections_path: Path | None = None,
                      product_id: str | None = None,
                      product_version: str | None = None,
                      metadata_sha256: str | None = None) -> dict:
    """Write one JSONL record per table, icon row, picture, or uncovered image."""
    started = perf_counter()
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    source_sha = sha256_file(source)
    parsed_sha = sha256_file(parsed_path)
    anchors = load_section_anchors(sections_path, parsed)
    anchor_keys = [(item["page_no"], -item["anchor_y"]) for item in anchors]
    pdf = pdfium.PdfDocument(source)
    page_count = len(pdf)
    if len(parsed.get("pages", {})) != page_count:
        raise ValueError("parsed page count does not match source PDF")
    pages = set(range(1, page_count+1)) if selected_pages is None else set(selected_pages)
    if not pages or min(pages) < 1 or max(pages) > page_count:
        raise ValueError("selected pages must exist in the source PDF")
    by_page = {name: defaultdict(list) for name in ("tables", "pictures", "texts")}
    for name in by_page:
        for item in parsed.get(name, []):
            if item.get("content_layer") == "furniture":
                continue
            for prov in item.get("prov", []):
                page = prov.get("page_no")
                if page in pages:
                    by_page[name][page].append((item, prov))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    by_kind = Counter()
    by_priority = Counter()
    page_summaries = []
    with output_path.open("w", encoding="utf-8") as target:
        for page_no in sorted(pages):
            page = pdf[page_no-1]
            width, height = page.get_size()
            images = []
            for object_no, obj in enumerate(page.get_objects()):
                if isinstance(obj, pdfium.PdfImage):
                    box = clipped(obj.get_bounds(), width, height)
                    if box:
                        images.append((object_no, box))
            texts = []
            for item, prov in by_page["texts"][page_no]:
                box = pdf_box(prov.get("bbox"), height)
                if box:
                    texts.append((box, item.get("text") or "", item.get("label") or ""))
            regions = []
            covered = []
            for table, prov in by_page["tables"][page_no]:
                box = pdf_box(prov.get("bbox"), height)
                if not box:
                    continue
                ref = table.get("self_ref", "")
                in_table = [(i, b) for i, b in images if overlap(b, box) >= 0.65]
                is_toc = table.get("label") == "document_index"
                route = "both" if in_table and not is_toc else "text"
                reason = "image-bearing table" if route == "both" else "text table or document index"
                rows = table_rows(table, box, height)
                preview = " | ".join(row["text"] for row in rows[:3])[:400]
                table_id = f"{document_id}:p{page_no:04d}:table-{ref.rsplit('/', 1)[-1]}"
                regions.append(make_region(document_id, page_no,
                    f"table-{ref.rsplit('/', 1)[-1]}", "table", box, route,
                    reason, preview, [ref], len(in_table), width, height))
                if in_table and not is_toc:
                    headers = rows[0]["text"] if rows else ""
                    for row in rows:
                        row_images = [(i, b) for i, b in in_table
                                      if overlap(b, row["bbox"]) >= 0.65]
                        if not row_images:
                            continue
                        context = (headers + " | " + row["text"]).strip(" |")[:500]
                        regions.append(make_region(document_id, page_no,
                            f"table-{ref.rsplit('/', 1)[-1]}-row-{row['row']}",
                            "icon_table_row", expanded(row["bbox"], width, height, 3),
                            "both", "PDF image within labeled table row", context,
                            [ref], len(row_images), width, height, table_id))
                covered.append(box)
            for picture, prov in by_page["pictures"][page_no]:
                box = pdf_box(prov.get("bbox"), height)
                if not box:
                    continue
                if any(overlap(box, table_box) >= 0.7 for table_box in covered):
                    continue  # The table or its row already represents this icon.
                ref = picture.get("self_ref", "")
                area = (box[2]-box[0])*(box[3]-box[1])/(width*height)
                context, label_box = text_context(box, texts)
                route = "both" if area < SMALL_REGION_AREA and context else "visual"
                reason = ("small picture with nearby text" if route == "both" else
                          "large picture/drawing" if area >= LARGE_PICTURE_AREA else
                          "picture without reliable text context")
                regions.append(make_region(document_id, page_no,
                    f"picture-{ref.rsplit('/', 1)[-1]}", "picture",
                    with_label(box, label_box, width, height)
                    if area < SMALL_REGION_AREA else box, route,
                    reason, context, [ref],
                    sum(overlap(image_box, box) >= 0.65 for _, image_box in images),
                    width, height, content_box=box))
                covered.append(box)
            for object_no, box in images:
                if any(overlap(box, visual_box) >= 0.7 for visual_box in covered):
                    continue
                context, label_box = text_context(box, texts)
                route = "both" if context else "visual"
                area = (box[2]-box[0])*(box[3]-box[1])/(width*height)
                regions.append(make_region(document_id, page_no,
                    f"pdf-image-{object_no}", "pdf_image",
                    with_label(box, label_box, width, height)
                    if area < SMALL_REGION_AREA else box, route,
                    "unmatched PDF image" + (" with nearby text" if context else ""),
                    context, [f"pdf-object:{object_no}"], 1, width, height,
                    content_box=box))
            for region in regions:
                region["product_id"] = product_id
                region["product_version"] = product_version
                region["source_pdf"] = source.name
                region["source_pdf_sha256"] = source_sha
                region["section"] = section_for(page_no, tuple(region["bbox_pdf"]),
                                                anchors, anchor_keys)
                target.write(json.dumps(region, ensure_ascii=False) + "\n")
                counts[region["route"]] += 1
                by_kind[region["kind"]] += 1
                by_priority[region["visual_priority"]] += 1
            page_summaries.append({"page": page_no, "pdf_images": len(images),
                                   "regions": len(regions),
                                   "visual_regions": sum(r["route"] != "text" for r in regions)})
            page.close()
    pdf.close()
    return {
        "schema_version": SCHEMA_VERSION, "classifier_version": CLASSIFIER_VERSION,
        "document_id": document_id, "source_sha256": source_sha,
        "document_metadata_sha256": metadata_sha256,
        "parsed_sha256": parsed_sha,
        "normalized_sections_sha256": sha256_file(sections_path)
        if sections_path is not None and sections_path.is_file() else None,
        "source_pages": page_count,
        "inventoried_pages": len(pages), "region_count": sum(counts.values()),
        "visual_priorities": dict(sorted(by_priority.items())),
        "thresholds": {"small_region_area": SMALL_REGION_AREA,
                       "large_picture_area": LARGE_PICTURE_AREA,
                       "medium_priority_area": 0.001,
                       "context_gap_points": CONTEXT_GAP_POINTS},
        "routes": dict(sorted(counts.items())), "kinds": dict(sorted(by_kind.items())),
        "pages_with_visual_regions": sum(x["visual_regions"] > 0 for x in page_summaries),
        "page_summaries": page_summaries,
        "wall_seconds": round(perf_counter()-started, 2),
    }
