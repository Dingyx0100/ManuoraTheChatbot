"""Make a Docling parse easy to inspect before chunking."""

import json
import re
from collections import Counter
from pathlib import Path

from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TableItem

from manuora.ingestion.chunk_context import map_item_context
from manuora.ingestion.pdf_outline import normalize_from_pdf_outline, read_pdf_outline
from manuora.ingestion.sections import normalize_sections


def export_review(
    parsed_path: Path, debug: bool = False, outline_pdf: Path | None = None
) -> dict[str, int | str]:
    """Write canonical sections; optionally export inspection views."""
    document = DoclingDocument.load_from_json(parsed_path)
    pages = {number: [] for number in range(1, document.num_pages() + 1)}
    sections = []
    callouts = []
    heading_path: list[str] = []
    labels: Counter[str] = Counter()

    for item, _ in document.iterate_items():
        label = str(item.label)
        callout_match = (
            re.search(r"\b(warning|note|caution)$", item.text, re.IGNORECASE)
            if isinstance(item, SectionHeaderItem) and len(item.text) < 20
            else None
        )
        if callout_match:
            label = f"{callout_match.group(1).lower()}_marker"
        labels[label] += 1
        page_numbers = sorted({provenance.page_no for provenance in getattr(item, "prov", [])})
        item_text = (
            item.export_to_markdown(doc=document)
            if isinstance(item, TableItem)
            else getattr(item, "text", "")
        )
        block = {"ref": item.self_ref, "label": label, "text": item_text}
        for page_number in page_numbers:
            pages[page_number].append(block)

        if callout_match:
            callouts.append(
                {
                    "ref": item.self_ref,
                    "page_no": page_numbers[0] if page_numbers else None,
                    "kind": callout_match.group(1).lower(),
                }
            )
        elif isinstance(item, SectionHeaderItem):
            level = max(item.level, 1)
            heading_path = heading_path[: level - 1]
            heading_path.append(item.text)
            sections.append(
                {
                    "ref": item.self_ref,
                    "page_no": page_numbers[0] if page_numbers else None,
                    "level": level,
                    "heading": item.text,
                    "provisional_heading_path": heading_path.copy(),
                    "anchor_y": item.prov[0].bbox.t if item.prov else None,
                }
            )

    if outline_pdf is not None:
        nodes, toc_numbers = read_pdf_outline(outline_pdf)
        body_sections = normalize_sections(sections, pages)
        outline_sections = (
            normalize_from_pdf_outline(sections, nodes, toc_numbers) if nodes else []
        )
        coverage = lambda records: sum(
            bool(context["normalized_heading_path"])
            for context in map_item_context(document, records).values()
        )
        outline_matches = sum(item["normalization_status"] == "matched"
                              for item in outline_sections)
        normalized_sections = (
            outline_sections if outline_matches >= 3
            and coverage(outline_sections) >= coverage(body_sections)
            else body_sections
        )
    else:
        normalized_sections = normalize_sections(sections, pages)
    output_dir = parsed_path.parent
    with (output_dir / "normalized_sections.jsonl").open("w", encoding="utf-8") as output:
        for section in normalized_sections:
            output.write(json.dumps(section, ensure_ascii=False) + "\n")

    summary = {
        "pages": len(pages),
        "sections": len(sections),
        "normalization_source": (
            normalized_sections[0]["normalization_source"]
            if normalized_sections
            else "none"
        ),
        "toc_matches": sum(
            section["role"] == "toc_section"
            or any("toc" in source for source in section.get("match_evidence", []))
            for section in normalized_sections
        ),
        "outline_matches": sum(
            section["role"] == "outline_section" for section in normalized_sections
        ),
        "outline_nodes": len(nodes) if outline_pdf is not None else 0,
        "reordered_chapter_titles": sum(
            section["order_corrected"] for section in normalized_sections
        ),
        "unmatched_section_paths": sum(
            section["normalization_status"] != "matched"
            for section in normalized_sections
        ),
        "callout_markers": len(callouts),
        **dict(labels),
    }
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        with (debug_dir / "pages.jsonl").open("w", encoding="utf-8") as output:
            for page_number, blocks in pages.items():
                output.write(
                    json.dumps({"page_no": page_number, "blocks": blocks}, ensure_ascii=False)
                    + "\n"
                )
        with (debug_dir / "sections.jsonl").open("w", encoding="utf-8") as output:
            for section in sections:
                output.write(json.dumps(section, ensure_ascii=False) + "\n")
        with (debug_dir / "callouts.jsonl").open("w", encoding="utf-8") as output:
            for callout in callouts:
                output.write(json.dumps(callout, ensure_ascii=False) + "\n")
        with (debug_dir / "toc_matches.jsonl").open("w", encoding="utf-8") as output:
            for section in normalized_sections:
                if section["role"] == "toc_section" or any(
                    "toc" in source for source in section.get("match_evidence", [])
                ):
                    output.write(json.dumps(section, ensure_ascii=False) + "\n")
        with (debug_dir / "pages.md").open("w", encoding="utf-8") as output:
            for page_number in pages:
                output.write(f"# PDF page {page_number}\n\n")
                output.write(document.export_to_markdown(page_no=page_number))
                output.write("\n\n")
        (debug_dir / "parse_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return summary
