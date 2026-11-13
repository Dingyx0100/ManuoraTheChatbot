"""Conservative, model-free reconstruction proposals; never mutate the saved parse.

Coordinates in proposals are PDF points with a top-left origin. A recovered status
means the structural checks passed, not that an independent accuracy audit passed.
"""

import re
import unicodedata
from collections import Counter

VERSION = "2"
CROSS_PAGE_VERSION = "1"


def bbox(box, height):
    if box.get("coord_origin") == "BOTTOMLEFT":
        return (box["l"], height - box["t"], box["r"], height - box["b"])
    return (box["l"], box["t"], box["r"], box["b"])


def inside(inner, outer, tolerance=2):
    return (inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance
            and inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def compact(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if not c.isspace())


def fidelity(original, recovered):
    """Character occurrence matching tolerates whitespace, never value substitution.

    This is a preservation guard, not proof of correct row/cell association.
    """
    a, b = Counter(compact(original)), Counter(compact(recovered))
    shared = sum((a & b).values())
    return dict(recall=shared / sum(a.values()) if a else float(not b),
                precision=shared / sum(b.values()) if b else float(not a))


def marker_kind(text):
    # Icon fonts sometimes decode to letters (e.g. one glyph + "Warning").
    # Permit a single separated glyph, still requiring the entire label and a
    # validated enclosure downstream; never match "Note that ..." prose.
    found = re.fullmatch(r"(?:[^\w]*|.\s+)(note|warning|caution|important|tip|attention|danger)\s*:?\s*",
                         text.strip(), flags=re.I)
    return found.group(1).lower() if found else None


def recover_table(page, table):
    result = dict(kind="table", source_block_id=table["self_ref"], status="unresolved",
                  method="ruled_grid", rows=None, checks={})
    if len(table.get("prov", [])) != 1:
        return {**result, "reason": "missing_or_multiple_page_provenance"}
    box = bbox(table["prov"][0]["bbox"], page.height)
    result["bbox"] = box
    if not inside(box, (0, 0, page.width, page.height), 0):
        return {**result, "reason": "source_region_outside_page"}
    pad = (max(0, box[0] - 4), max(0, box[1] - 4),
           min(page.width, box[2] + 4), min(page.height, box[3] + 4))
    tables = page.crop(pad).find_tables({"snap_tolerance": 2, "join_tolerance": 2,
                                        "intersection_tolerance": 3})
    candidates = [t for t in tables if inside(t.bbox, box, 5)
                  and len(t.columns) == table["data"]["num_cols"]]
    if len(candidates) != 1:
        return {**result, "reason": "no_unique_complete_ruled_grid",
                "grid_candidates": len(candidates)}
    found = candidates[0]
    rows = found.extract(x_tolerance=1, y_tolerance=2)
    result.update(rows=rows, bbox=found.bbox,
                  cell_bboxes=[list(row.cells) for row in found.rows])
    if any(value is None for row in rows for value in row):
        return {**result, "reason": "spanning_cells_require_separate_validation"}
    cells = table["data"]["table_cells"]
    if any(c.get("col_span", 1) > 1 or c.get("row_span", 1) > 1 for c in cells):
        return {**result, "reason": "existing_spanning_cells_preserved"}
    checks = []
    for column in range(table["data"]["num_cols"]):
        old = " ".join(c["text"] for c in cells if c["start_col_offset_idx"] == column)
        new = " ".join(row[column] or "" for row in rows)
        checks.append({**fidelity(old, new), "sequence_preserved": compact(old) == compact(new)})
    result["checks"]["column_content_preservation"] = checks
    # Do not propose a replacement if even one non-whitespace character differs.
    if any(c["recall"] < 1 or c["precision"] < 1 or not c["sequence_preserved"] for c in checks):
        return {**result, "reason": "column_content_changed"}
    if any(inside((i["x0"], i["top"], i["x1"], i["bottom"]), found.bbox, 0)
           for i in page.images):
        return {**result, "reason": "embedded_symbols_need_preservation"}
    old_rows = table["data"]["num_rows"]
    if len(rows) <= old_rows:
        return {**result, "status": "unchanged", "reason": "no_additional_grid_rows"}
    result.update(status="recovered", reason="more_rows_with_exact_column_content",
                  before_rows=old_rows, after_rows=len(rows))
    return result


def recover_callout(page, marker, texts):
    kind = marker_kind(marker.get("text", ""))
    result = dict(kind="callout", source_block_id=marker["self_ref"],
                  callout_type=kind, status="unresolved", body_block_ids=[], checks={})
    if not kind:
        return {**result, "status": "unchanged", "reason": "not_a_callout_marker"}
    if len(marker.get("prov", [])) != 1:
        return {**result, "reason": "missing_or_multiple_page_provenance"}
    mbox = bbox(marker["prov"][0]["bbox"], page.height)
    result["bbox"] = mbox
    candidates = []
    for shape in [*page.rects, *page.curves]:
        box = (shape["x0"], shape["top"], shape["x1"], shape["bottom"])
        # Rounded boxes are closed filled curves. Ignore tiny icons, header-only
        # fills, page backgrounds and open decorative paths.
        path = shape.get("path", [])
        closed = (shape.get("object_type") == "rect" or any(segment[0] == "h" for segment in path)
                  or bool(path and len(path[0]) > 1 and len(path[-1]) > 1
                          and max(abs(a - b) for a, b in zip(path[0][1], path[-1][-1])) < 1))
        if not closed or not (shape.get("fill") or shape.get("stroke")):
            continue
        if (inside(mbox, box) and box[2] - box[0] > 100
                and 15 < box[3] - mbox[3] < page.height * .65
                and mbox[1] - box[1] < 30
                and box[1] > 15 and box[3] < page.height - 15):
            if not any(max(abs(a - b) for a, b in zip(box, c)) < 2 for c in candidates):
                candidates.append(box)
    if len(candidates) != 1:
        return {**result, "reason": "no_unique_enclosing_shape", "box_candidates": len(candidates)}
    box = candidates[0]
    result.update(bbox=box, method="enclosing_pdf_shape")
    if any(f['bottom_open'] and inside(mbox,f['bbox'])
           and max(abs(a-b) for a,b in zip(box,f['bbox'])) < 2 for f in _filled_frames(page)):
        return {**result, "reason": "open_bottom_requires_continuation"}
    members, crossing = [], []
    for text in texts:
        if text["self_ref"] == marker["self_ref"] or text.get("content_layer") != "body":
            continue
        for prov in text.get("prov", []):
            if prov["page_no"] != page.page_number:
                continue
            tb = bbox(prov["bbox"], page.height)
            if inside(tb, box):
                if len(text["prov"]) != 1:
                    crossing.append(text["self_ref"])
                else:
                    members.append((tb, text))
            elif max(tb[0], box[0]) < min(tb[2], box[2]) - 2 and max(tb[1], box[1]) < min(tb[3], box[3]) - 2:
                crossing.append(text["self_ref"])
    if crossing:
        return {**result, "reason": "block_crosses_boundary", "crossing_block_ids": crossing}
    members.sort(key=lambda value: (round(value[0][1], 1), value[0][0]))
    if not members or any(marker_kind(t.get("text", "")) for _, t in members):
        return {**result, "reason": "empty_body_or_multiple_markers"}
    def body_text(text):
        value, marker = text.get("text", ""), text.get("marker", "")
        return f"{marker} {value}" if marker and not value.lstrip().startswith(marker) else value
    body = "\n".join(body_text(t) for _, t in members)
    # Compare the body independently: icon fonts in the title can appear as
    # '(cid:174)' in PDFMiner. Do not interpret that extraction artifact as prose.
    native = page.crop((box[0], mbox[3] + 1, box[2], box[3])).extract_text(x_tolerance=1, y_tolerance=2) or ""
    letters = lambda s: "".join(c for c in unicodedata.normalize("NFKC", s).casefold() if c.isalnum())
    check = fidelity(letters(native), letters(body))
    check["sequence_preserved"] = letters(native) == letters(body)
    result["checks"]["native_content_preservation"] = check
    if not check["sequence_preserved"]:
        return {**result, "reason": "native_and_parsed_content_disagree", "body_text": body}
    result.update(status="recovered", reason="enclosure_and_content_checks_passed",
                  marker_text=marker["text"], body_text=body,
                  body_block_ids=[t["self_ref"] for _, t in members],
                  embedded_image_count=sum(inside((i["x0"], i["top"], i["x1"], i["bottom"]), box)
                                           for i in page.images),
                  semantic_scope="not_inferred")
    return result


def _filled_frames(page):
    """Paired PDF fills expose a border, including its open/closed edges.

    Some PDF generators draw a darker outer shape then a lighter inset body.
    At a page break, both fills end at the same y coordinate (no border there).
    This deliberately abstains on layouts without this independent edge evidence.
    """
    shapes = [s for s in [*page.rects, *page.curves]
              if s.get("fill") and s["x1"] - s["x0"] > 100
              and s["bottom"] - s["top"] > 10]
    frames = []
    for outer in shapes:
        ob = (outer["x0"], outer["top"], outer["x1"], outer["bottom"])
        for inner in shapes:
            ib = (inner["x0"], inner["top"], inner["x1"], inner["bottom"])
            inset = (ib[0] - ob[0], ib[1] - ob[1], ob[2] - ib[2], ob[3] - ib[3])
            if not (.2 <= inset[0] <= 2 and .2 <= inset[2] <= 2
                    and -.1 <= inset[1] <= 2 and -.1 <= inset[3] <= 2):
                continue
            oc, ic = outer.get("non_stroking_color"), inner.get("non_stroking_color")
            if oc is None or ic is None or oc == ic:
                continue
            frame = dict(bbox=ob, border_color=oc, body_color=ic,
                         top_open=abs(inset[1]) <= .15,
                         bottom_open=abs(inset[3]) <= .15,
                         insets=inset)
            if not any(max(abs(a-b) for a,b in zip(ob, f["bbox"])) < .1
                       and oc == f["border_color"] and ic == f["body_color"] for f in frames):
                frames.append(frame)
    return frames


def _colors_match(a, b):
    if isinstance(a, (int, float)):
        a = [a]
    if isinstance(b, (int, float)):
        b = [b]
    return len(a) == len(b) and max(abs(x-y) for x,y in zip(a, b)) <= .02


def _lexical(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def recover_cross_page_callout(page, next_page, marker, texts):
    """Link two adjacent open-edge frames, retaining source character spans.

    No model, source mutation or inferred semantic warning scope. Partial text
    blocks are allowed only with valid provenance spans and exact native lexical
    agreement. A three-page continuation remains unresolved in this first pilot.
    """
    result = dict(kind="callout", source_block_id=marker["self_ref"],
                  callout_type=marker_kind(marker.get("text", "")), status="unresolved",
                  method="cross_page_pdf_frames", cross_page_version=CROSS_PAGE_VERSION,
                  segments=[], body_block_ids=[], source_spans=[], checks={},
                  semantic_scope="not_inferred")
    if not result["callout_type"] or len(marker.get("prov", [])) != 1:
        return {**result, "reason": "not_a_single_page_callout_marker"}
    if marker["prov"][0]["page_no"] != page.page_number:
        return {**result, "reason": "marker_page_mismatch"}
    if next_page is None:
        return {**result, "reason": "next_page_unavailable"}
    if next_page.page_number != page.page_number + 1 or next_page.pdf is not page.pdf:
        return {**result, "reason": "pages_not_adjacent_in_same_pdf"}
    mb = bbox(marker["prov"][0]["bbox"], page.height)
    start = [f for f in _filled_frames(page) if inside(mb, f["bbox"])
             and f["bottom_open"] and not f["top_open"]
             and mb[1] - f["bbox"][1] < 30
             and f["bbox"][1] > page.height * .65 and f["bbox"][3] > page.height * .8]
    if len(start) != 1:
        return {**result, "reason": "no_unique_open_bottom_frame", "start_candidates": len(start)}
    first = start[0]
    end = [f for f in _filled_frames(next_page) if f["top_open"] and not f["bottom_open"]
           and 15 < f["bbox"][1] < next_page.height * .2
           and max(abs(first["bbox"][i] - f["bbox"][i]) for i in (0,2)) <= 2
           and _colors_match(first["border_color"], f["border_color"])
           and _colors_match(first["body_color"], f["body_color"])]
    if len(end) != 1:
        return {**result, "reason": "no_unique_matching_open_top_frame", "end_candidates": len(end)}
    last = end[0]
    segments = [dict(page=pg.page_number, bbox=f["bbox"], frame=f)
                for pg,f in [(page,first),(next_page,last)]]
    result["segments"] = segments
    selected, checks = [], []
    for pg, segment in zip((page, next_page), segments):
        region = segment["bbox"]
        pieces = []
        for text in texts:
            if text.get("content_layer") != "body" or text["self_ref"] == marker["self_ref"]:
                continue
            for prov in text.get("prov", []):
                if prov["page_no"] != pg.page_number:
                    continue
                tb = bbox(prov["bbox"], pg.height)
                # A separate heading/body block between marker and continuation
                # is contradictory evidence, even if the candidate frame matches.
                between = (tb[1] > first["bbox"][3]+2 if pg is page else tb[3] < last["bbox"][1]-2)
                horizontal_overlap = max(tb[0], region[0]) < min(tb[2], region[2])-2
                if between and horizontal_overlap:
                    return {**result, "reason": "intervening_body_block", "conflict_block_id": text["self_ref"]}
                intersects = (horizontal_overlap and max(tb[1],region[1]) < min(tb[3],region[3])-2)
                if not inside(tb, region):
                    if intersects:
                        return {**result, "reason": "body_box_crosses_segment", "conflict_block_id": text["self_ref"]}
                    continue
                if marker_kind(text.get("text", "")):
                    return {**result, "reason": "competing_callout_marker", "conflict_block_id": text["self_ref"]}
                span = prov.get("charspan")
                value = text.get("text", "")
                if (not span or len(span) != 2 or any(not isinstance(x, int) for x in span)
                        or not 0 <= span[0] < span[1] <= len(value)):
                    return {**result, "reason": "invalid_source_charspan", "conflict_block_id": text["self_ref"]}
                fragment = value[span[0]:span[1]]
                list_marker = text.get("marker", "")
                if span[0] == 0 and list_marker and not fragment.lstrip().startswith(list_marker):
                    fragment = list_marker + " " + fragment
                pieces.append(dict(block_id=text["self_ref"], page=pg.page_number,
                                   charspan=span, bbox=tb, text=fragment))
        pieces.sort(key=lambda p:(round(p["bbox"][1],1),p["bbox"][0]))
        native_box = (region[0], mb[3]+1 if pg is page else region[1],region[2],region[3])
        native = pg.crop(native_box).extract_text(x_tolerance=1,y_tolerance=2) or ""
        extracted = "\n".join(p["text"] for p in pieces)
        check = dict(page=pg.page_number, sequence_preserved=_lexical(native)==_lexical(extracted),
                     **fidelity(_lexical(native),_lexical(extracted)))
        checks.append(check)
        selected.extend(pieces)
        if not check["sequence_preserved"]:
            return {**result, "reason": "native_and_selected_spans_disagree", "checks": {"pages": checks}}
        segment["body_text"] = extracted
    if not segments[1]["body_text"].strip():
        return {**result, "reason": "empty_continuation"}
    by_ref = {}
    for piece in selected:
        spans = by_ref.setdefault(piece["block_id"], [])
        if any(max(s[0],piece["charspan"][0]) < min(s[1],piece["charspan"][1]) for s in spans):
            return {**result, "reason": "overlapping_source_spans"}
        spans.append(piece["charspan"])
    node_by_ref = {t["self_ref"]:t for t in texts}
    partial = []
    for ref,spans in by_ref.items():
        covered = set(i for lo,hi in spans for i in range(lo,hi))
        if any(not c.isspace() and i not in covered for i,c in enumerate(node_by_ref[ref]["text"])):
            partial.append(ref)
    result.update(status="recovered", reason="adjacent_open_edges_and_source_spans_verified",
                  body_block_ids=list(by_ref), source_spans=selected, partial_block_ids=partial,
                  body_text="\n".join(s["body_text"] for s in segments if s["body_text"].strip()),
                  checks=dict(pages=checks, non_overlapping_source_spans=True,
                              matching_width_and_colors=True, complementary_open_edges=True))
    return result
