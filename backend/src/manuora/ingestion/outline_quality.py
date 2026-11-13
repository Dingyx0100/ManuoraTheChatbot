"""Independent outline recall and spatially grounded section specificity.

The reference comes from PDF destinations or a reviewed native-text TOC, never
from normalized_sections.jsonl. Section assignments are the system under test.
"""

import ctypes
import re
import unicodedata
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Literal

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from pydantic import BaseModel, ConfigDict, Field

from manuora.ingestion.chunk_preservation import source_inventory
from manuora.ingestion.files import sha256_file


OUTLINE_COVERAGE_FAIL_BELOW = 0.70


class OutlineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_id: str
    title: str
    parent_id: str | None = None
    page_no: int | None = None  # Physical, one-based PDF page.
    y_top: float | None = Field(default=None, allow_inf_nan=False)
    exclusion_reason: str | None = None


class TrustedTOC(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    source_pdf_sha256: str
    reference_type: Literal["reviewed_native_toc"]
    independent_of_heading_detector: Literal[True]
    reviewed: Literal[True]
    provenance: str = Field(min_length=1)
    entries: list[OutlineEntry]


def _key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", value)
    return re.sub(r"[\W_]+", " ", value).strip()


def _bookmarks(path: Path) -> list[OutlineEntry]:
    pdf = pdfium.PdfDocument(path)
    entries, ancestors, heights = [], [], {}
    try:
        for index, bookmark in enumerate(pdf.get_toc() or []):
            ancestors = ancestors[:bookmark.level]
            dest = bookmark.get_dest()
            page_index = dest.get_index() if dest is not None else None
            page_no = page_index + 1 if page_index is not None else None
            y_top = None
            if page_no is not None and 1 <= page_no <= len(pdf):
                if page_no not in heights:
                    page = pdf[page_no - 1]
                    try:
                        heights[page_no] = page.get_height()
                    finally:
                        page.close()
                # GetView's zero can mean an unspecified XYZ coordinate. Read
                # the explicit presence flag; page-only destinations stay so.
                has_x, has_y, has_zoom = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
                x, y, zoom = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
                ok = pdfium_raw.FPDFDest_GetLocationInPage(
                    dest, ctypes.byref(has_x), ctypes.byref(has_y), ctypes.byref(has_zoom),
                    ctypes.byref(x), ctypes.byref(y), ctypes.byref(zoom),
                )
                if ok and has_y.value:
                    y_top = heights[page_no] - y.value
            entry_id = f"bookmark:{index}"
            entries.append(OutlineEntry(entry_id=entry_id, title=bookmark.get_title(),
                                        parent_id=ancestors[-1] if ancestors else None,
                                        page_no=page_no, y_top=y_top))
            ancestors.append(entry_id)
    finally:
        pdf.close()
    return entries


def _reference(source_path, trusted_outline_path, source_hash, trust_pdf_bookmarks):
    if source_path is not None and sha256_file(source_path) != source_hash:
        raise ValueError("Outline source hash differs from profile")
    if trusted_outline_path is not None:
        reference = TrustedTOC.model_validate_json(trusted_outline_path.read_text(encoding="utf-8"))
        if reference.source_pdf_sha256 != source_hash:
            raise ValueError("Trusted TOC source hash differs from current PDF/profile")
        return reference.entries, dict(reference_type=reference.reference_type,
            reference_sha256=sha256_file(trusted_outline_path), provenance=reference.provenance,
            reference_trusted=True, trust_basis="independently reviewed native TOC")
    if source_path is not None:
        return _bookmarks(source_path), dict(reference_type="pdf_bookmarks", reference_sha256=source_hash,
            reference_trusted=trust_pdf_bookmarks,
            trust_basis="explicit bookmark trust declaration" if trust_pdf_bookmarks else "bookmarks not verified")
    return [], dict(reference_type="unavailable", reference_sha256=None,
                    reference_trusted=False, trust_basis="no reference")


def evaluate_outline_quality(document, contexts: dict, *, source_sha256: str,
                             source_path: Path | None = None,
                             trusted_outline_path: Path | None = None,
                             trust_pdf_bookmarks: bool = False) -> dict:
    started = perf_counter()
    entries, provenance = _reference(source_path, trusted_outline_path, source_sha256, trust_pdf_bookmarks)
    by_id = {entry.entry_id: entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("Duplicate outline entry ID")
    paths = {}
    for entry in entries:
        path, seen, current = [], set(), entry
        while current is not None:
            if current.entry_id in seen:
                raise ValueError("Cycle in outline hierarchy")
            seen.add(current.entry_id)
            path.append(current.title)
            if current.parent_id is not None and current.parent_id not in by_id:
                raise ValueError("Unknown parent in outline hierarchy")
            current = by_id.get(current.parent_id)
        paths[entry.entry_id] = list(reversed(path))

    eligible, excluded, destinations = [], [], set()
    for entry in entries:
        reason = entry.exclusion_reason
        if not reason and not _key(entry.title):
            reason = "empty title"
        if not reason and entry.page_no not in document.pages:
            reason = "no valid local PDF destination"
        signature = (_key(entry.title), entry.page_no, entry.y_top, entry.parent_id)
        if not reason and signature in destinations:
            reason = "duplicate title/destination under the same parent"
        if reason:
            excluded.append(dict(entry_id=entry.entry_id, title=entry.title, reason=reason))
        else:
            destinations.add(signature)
            eligible.append(entry)

    blocks, block_exclusions = source_inventory(document, [])
    allowed = {block["block_id"] for block in blocks}
    # Search raw saved text, including ordinary text and picture children.
    # Require a complete title line/block, not an incidental substring or a
    # normalized heading injected from the very outline being evaluated.
    lines_by_key = defaultdict(list)
    for item in document.texts:
        if item.self_ref not in allowed:
            continue
        for prov in item.prov:
            if prov.page_no not in document.pages:
                continue
            value = item.text
            if len({span.page_no for span in item.prov}) > 1:
                start, end = prov.charspan
                if not 0 <= start < end <= len(value) or any(
                    other.page_no != prov.page_no and max(start, other.charspan[0]) < min(end, other.charspan[1])
                    for other in item.prov
                ):
                    continue  # Cannot assign this title text to a page reliably.
                value = value[start:end]
            top = prov.bbox.to_top_left_origin(document.pages[prov.page_no].size.height).t
            values = set(value.splitlines() + [" ".join(value.split())])
            for value in values:
                key = _key(value)
                if key:
                    lines_by_key[prov.page_no, key].append(dict(ref=item.self_ref, y_top=top,
                                                               label=str(item.label)))
    matches, used = [], set()
    for entry in eligible:
        candidates = {candidate["ref"]: candidate for candidate in lines_by_key[entry.page_no, _key(entry.title)]}
        available = [candidate for ref, candidate in candidates.items() if (ref, _key(entry.title)) not in used]
        chosen, method = None, "title_on_destination_page"
        if len(available) == 1:
            chosen = available[0]
        elif len(available) > 1 and entry.y_top is not None:
            available.sort(key=lambda candidate: abs(candidate["y_top"] - entry.y_top))
            distance = abs(available[0]["y_top"] - entry.y_top)
            gap = abs(available[1]["y_top"] - entry.y_top) - distance
            if distance <= 40 and gap >= 10:
                chosen, method = available[0], "title_page_position"
        if chosen:
            used.add((chosen["ref"], _key(entry.title)))
        matches.append(dict(entry_id=entry.entry_id, title=entry.title, page_no=entry.page_no,
                            parent_id=entry.parent_id, y_top=entry.y_top, heading_path=paths[entry.entry_id],
                            status="found" if chosen else ("ambiguous" if candidates else "missing"),
                            matched_ref=chosen["ref"] if chosen else None,
                            parsed_label=chosen["label"] if chosen else None,
                            match_method=method if chosen else None))

    # Independent section intervals. Shared/unknown same-page boundaries must
    # not become a fabricated deep expected section.
    at_position, on_page = defaultdict(list), defaultdict(list)
    invalid_boundary_entries = []
    for entry in eligible:
        height = document.pages[entry.page_no].size.height
        if entry.y_top is not None and not 0 <= entry.y_top <= height:
            invalid_boundary_entries.append(dict(entry_id=entry.entry_id, page_no=entry.page_no,
                                                 reason="destination coordinate outside page"))
            y_top = 0.0
        else:
            y_top = entry.y_top if entry.y_top is not None else 0.0
        at_position[entry.page_no, y_top].append(entry)
        on_page[entry.page_no].append(entry)
    positions = sorted(at_position)
    ambiguous_pages = {page for page, nodes in on_page.items()
                       if len(nodes) > 1 and any(node.y_top is None for node in nodes)}
    ambiguous_pages.update(row["page_no"] for row in invalid_boundary_entries)

    def expected_at(page, y_top):
        if page in ambiguous_pages:
            return None, "ambiguous same-page outline boundaries"
        index = bisect_right(positions, (page, y_top)) - 1
        if index < 0:
            return None, "before first eligible outline destination"
        group = at_position[positions[index]]
        if len(group) != 1:
            return None, "shared outline destination"
        return group[0], None

    nodes = {item.self_ref: item for item in [*document.texts, *document.tables]}
    histogram = defaultdict(lambda: dict(blocks=0, tokens=0))
    issues, unassessed = [], []
    total = assigned = top_level = assessed = coarse = wrong = missing = 0
    weighted_specificity = 0.0
    reference_title_refs = {row["matched_ref"] for row in matches
                            if row["status"] == "found"
                            and _key(nodes[row["matched_ref"]].text) == _key(row["title"])}
    body_exclusions = list(block_exclusions)
    for block in blocks:
        if block["label"] in {"section_header", "title"} or block["block_id"] in reference_title_refs:
            body_exclusions.append(dict(block_id=block["block_id"], reason="heading/title excluded from body-specificity denominator"))
            continue
        ref, weight = block["block_id"], len(block["tokens"])
        actual = contexts.get(ref, {}).get("normalized_heading_path", [])
        total += weight
        if actual:
            assigned += weight
            top_level += weight if len(actual) == 1 else 0
            histogram[tuple(actual)]["blocks"] += 1
            histogram[tuple(actual)]["tokens"] += weight
        expected, reason = None, "no independent outline"
        if eligible:
            prov = nodes[ref].prov
            expected_ids = set()
            reason = "missing page provenance"
            for span in prov:
                if span.page_no not in document.pages:
                    reason = "invalid page provenance"
                    break
                box = span.bbox.to_top_left_origin(document.pages[span.page_no].size.height)
                start, reason = expected_at(span.page_no, min(box.t, box.b))
                end, _ = expected_at(span.page_no, max(min(box.t, box.b), max(box.t, box.b) - 1e-6))
                if start is None or end is None or start.entry_id != end.entry_id:
                    reason = reason or "block crosses an outline boundary"
                    break
                expected_ids.add(start.entry_id)
            else:
                if len(expected_ids) == 1:
                    expected = by_id[next(iter(expected_ids))]
                    reason = None
                elif expected_ids:
                    reason = "block spans several outline sections"
        if expected is None:
            unassessed.append(dict(block_id=ref, tokens=weight, reason=reason))
            continue
        assessed += weight
        wanted = paths[expected.entry_id]
        a, b = [_key(title) for title in actual], [_key(title) for title in wanted]
        if not a:
            score, kind = 0.0, "unassigned"
            missing += weight
        elif a[:len(b)] == b:
            score, kind = 1.0, "specific"
        elif b[:len(a)] == a:
            score, kind = len(a) / len(b), "ancestor_only"
            coarse += weight
        else:
            score, kind = 0.0, "wrong_branch"
            wrong += weight
        weighted_specificity += score * weight
        if kind != "specific":
            issues.append(dict(block_id=ref, pages=block["pages"], tokens=weight, status=kind,
                               assigned_path=actual, expected_path=wanted, expected_entry_id=expected.entry_id))

    found = sum(row["status"] == "found" for row in matches)
    coverage = found / len(eligible) if eligible else None
    outline_status = (
        "UNAVAILABLE" if coverage is None else
        "FAIL" if provenance["reference_trusted"] and coverage < OUTLINE_COVERAGE_FAIL_BELOW else
        "WARN" if coverage < 1 else "PASS"
    )
    dominant = max((value["tokens"] for value in histogram.values()), default=0)
    return dict(
        outline_heading_coverage=coverage,
        section_specificity=weighted_specificity / assessed if assessed else None,
        outline_quality=dict(
            metric_version="2", **provenance, source_pdf_sha256=source_sha256,
            status=outline_status,
            fail_below=OUTLINE_COVERAGE_FAIL_BELOW, warn_below=1.0,
            fail_gate_enabled=bool(eligible) and provenance["reference_trusted"],
            eligible_entries=len(eligible), found_entries=found, excluded_entries=excluded, entries=matches,
            reference_max_depth=max((len(paths[entry.entry_id]) for entry in eligible), default=0),
            matching_policy="exact_normalized_title_on_destination_page; raw text only; one-to-one",
        ),
        section_specificity_details=dict(
            metric_version="1", scope="parsed_body_blocks_and_their_current_section_assignments",
            status=("WARN" if issues else "PASS") if assessed else "UNMEASURED",
            eligible_tokens=total, assessed_tokens=assessed,
            assessment_coverage=assessed / total if total else None,
            assigned_token_ratio=assigned / total if total else None,
            top_level_only_token_ratio=top_level / total if total else None,
            dominant_section_token_ratio=dominant / total if total else None,
            distinct_assigned_sections=len(histogram),
            ancestor_only_token_ratio=coarse / assessed if assessed else None,
            wrong_branch_token_ratio=wrong / assessed if assessed else None,
            unassigned_token_ratio=missing / assessed if assessed else None,
            assignments=[dict(path=list(path), **value) for path, value in histogram.items()],
            issues=issues, unassessed_blocks=unassessed,
            excluded_blocks=body_exclusions, invalid_boundary_entries=invalid_boundary_entries,
            boundary_policy="PDF top-left coordinates; entire block must fall in one independent interval",
        ),
        outline_evaluation_wall_seconds=round(perf_counter() - started, 6),
    )
