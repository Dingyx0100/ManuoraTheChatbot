"""Attach normalized section context to Docling chunks without rechunking."""

import re
from bisect import bisect_right
from typing import Iterable

from docling_core.types.doc import DoclingDocument, SectionHeaderItem


CALLOUT_MARKER = re.compile(r"\b(warning|note|caution)$", re.IGNORECASE)


def map_item_context(document: DoclingDocument, sections: Iterable[dict]) -> dict[str, dict]:
    """Map body items to normalized headings, using PDF positions when available.

    Warning/note markers are tracked as callout starts, never as section parents.
    """
    sections = list(sections)
    if any(section.get("normalization_source") == "pdf_outline" for section in sections):
        return _map_outline_item_context(document, sections)
    by_ref = {section["ref"]: section for section in sections}
    path: list[str] = []
    section_ref = None
    local_heading = None
    pending_callout = None
    by_item_ref = {}

    for item, _ in document.iterate_items(traverse_pictures=True):
        if isinstance(item, SectionHeaderItem):
            marker = (
                CALLOUT_MARKER.search(item.text.strip())
                if len(item.text.strip()) < 20
                else None
            )
            if marker:
                pending_callout = {"kind": marker.group(1).lower(), "ref": item.self_ref}
                continue

            pending_callout = None
            section = by_ref.get(item.self_ref)
            if section is None:
                continue
            if section["normalization_status"] == "matched":
                path = section["normalized_heading_path"].copy()
                section_ref = item.self_ref
                local_heading = None
            elif section["role"] == "local_heading":
                path = section["normalized_heading_path"].copy()
                local_heading = section["heading"]
            else:
                path = []
                section_ref = None
                local_heading = section["heading"]
            continue

        untexted_picture = str(item.label) == "picture" and not getattr(item, "text", None)
        by_item_ref[item.self_ref] = {
            "normalized_heading_path": path.copy(),
            "section_ref": section_ref,
            "local_heading": local_heading,
            "callout_start": None if untexted_picture else pending_callout,
        }
        if not untexted_picture:
            pending_callout = None

    return by_item_ref


def _map_outline_item_context(document: DoclingDocument, sections: list[dict]) -> dict[str, dict]:
    """Assign content by physical page position, independent of parser item order."""
    anchors = sorted(
        (
            ((section["page_no"], -section["anchor_y"]), section)
            for section in sections
            if section["normalization_status"] == "matched"
            and section.get("page_no") is not None
            and section.get("anchor_y") is not None
        ),
        key=lambda entry: entry[0],
    )
    positions = [position for position, _ in anchors]
    by_item_ref = {}
    pending_callout = None
    for item, _ in document.iterate_items(traverse_pictures=True):
        if isinstance(item, SectionHeaderItem):
            marker = (
                CALLOUT_MARKER.search(item.text.strip())
                if len(item.text.strip()) < 20 else None
            )
            pending_callout = (
                {"kind": marker.group(1).lower(), "ref": item.self_ref}
                if marker else None
            )
            continue
        provenance = getattr(item, "prov", [])
        if provenance:
            first = provenance[0]
            index = bisect_right(positions, (first.page_no, -first.bbox.t)) - 1
            section = anchors[index][1] if index >= 0 else None
        else:
            section = None
        untexted_picture = str(item.label) == "picture" and not getattr(item, "text", None)
        by_item_ref[item.self_ref] = {
            "normalized_heading_path": (
                section["normalized_heading_path"].copy() if section else []
            ),
            "section_ref": section["ref"] if section else None,
            "local_heading": None,
            "callout_start": None if untexted_picture else pending_callout,
        }
        if not untexted_picture:
            pending_callout = None
    return by_item_ref


def adapt_chunk(record: dict, context_by_ref: dict[str, dict]) -> dict:
    """Replace a chunk's raw heading prefix when all source items agree."""
    contexts = [context_by_ref[ref] for ref in record["refs"] if ref in context_by_ref]
    paths = {tuple(context["normalized_heading_path"]) for context in contexts}
    adapted = {**record, "raw_headings": record["headings"]}
    adapted.pop("context_tokens", None)  # The caller must recount after changing the prefix.
    adapted["callout_starts"] = [
        context["callout_start"]
        for context in contexts
        if context["callout_start"] is not None
    ]

    if not contexts or len(paths) != 1 or not next(iter(paths)):
        adapted["context_status"] = "mixed_sections" if len(paths) > 1 else "unresolved"
        adapted["normalized_heading_path"] = []
        return adapted

    path = list(next(iter(paths)))
    local_headings = {context["local_heading"] for context in contexts}
    local_heading = next(iter(local_headings)) if len(local_headings) == 1 else None
    headings = [*path, *([local_heading] if local_heading else [])]
    adapted.update(
        context_status="normalized",
        normalized_heading_path=path,
        section_ref=contexts[0]["section_ref"],
        local_heading=local_heading,
        headings=headings,
        contextual_text="\n".join([*headings, record["text"]]),
    )
    return adapted
