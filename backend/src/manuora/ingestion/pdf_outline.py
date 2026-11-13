"""Join PDF bookmarks to parsed headings without trusting Docling heading order."""

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pypdfium2


LEADING_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s+")
TOC_LINE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s+(\d+)\s*$")


def title_key(value: str) -> str:
    value = LEADING_NUMBER.sub("", value)
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\W_]+", " ", value).strip()


def read_pdf_outline(pdf_path: Path) -> tuple[list[dict], dict[str, str]]:
    """Read bookmark hierarchy and numbered TOC lines from the source PDF."""
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    nodes = []
    ancestors: list[int] = []
    for bookmark in pdf.get_toc() or []:
        destination = bookmark.get_dest()
        if destination is None or bookmark.level > len(ancestors):
            continue
        ancestors = ancestors[: bookmark.level]
        view = destination.get_view()
        node = {
            "index": len(nodes),
            "title": bookmark.get_title(),
            "level": bookmark.level,
            "parent_index": ancestors[-1] if ancestors else None,
            "page_no": destination.get_index() + 1,
            "y": view[1][1] if view and len(view[1]) > 1 else None,
        }
        nodes.append(node)
        ancestors.append(node["index"])

    toc_numbers: dict[str, str] = {}
    conflicting: set[str] = set()
    first_body_page = nodes[0]["page_no"] if nodes else 1
    for page_no in range(1, min(first_body_page, 12)):
        text = pdf[page_no - 1].get_textpage().get_text_range()
        for line in text.splitlines():
            match = TOC_LINE.match(line)
            if not match:
                continue
            number, title, _ = match.groups()
            title = re.sub(r"(?:\s*\.){2,}\s*$", "", title).strip()
            key = title_key(title)
            if key in toc_numbers and toc_numbers[key] != number:
                conflicting.add(key)
            else:
                toc_numbers[key] = number
    for key in conflicting:
        toc_numbers.pop(key, None)
    return nodes, toc_numbers


def normalize_from_pdf_outline(sections: list[dict], nodes: list[dict], toc_numbers: dict[str, str]) -> list[dict]:
    """Use page, title and position to attach parsed headings to outline nodes."""
    by_title_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for section in sections:
        if section["page_no"] is not None:
            by_title_page[(title_key(section["heading"]), section["page_no"])].append(section)

    used_refs: set[str] = set()
    matched: dict[int, dict] = {}
    for node in nodes:
        key = title_key(node["title"])
        page = node["page_no"]
        candidates = []
        repeated_on_page = False
        match_kind = None
        for kind, possibilities in (
            ("title_page", by_title_page.get((key, page), [])),
            ("joined_words", [s for (k, p), values in by_title_page.items()
                              if p == page and k.replace(" ", "") == key.replace(" ", "")
                              for s in values]),
            ("next_page", by_title_page.get((key, page + 1), [])),
        ):
            candidates = [s for s in possibilities if s["ref"] not in used_refs]
            if candidates:
                match_kind = kind
                repeated_on_page = kind != "next_page" and len(possibilities) > 1
                break
        if not candidates:
            continue
        if len(candidates) > 1 or repeated_on_page:
            if match_kind == "next_page" or node["y"] is None:
                continue
            candidates.sort(key=lambda s: abs(node["y"] - s["anchor_y"]))
            first = abs(node["y"] - candidates[0]["anchor_y"])
            second = abs(node["y"] - candidates[1]["anchor_y"]) if len(candidates) > 1 else None
            if first > 25 or (second is not None and second - first < 10):
                continue
            match_kind = "position"
        section = candidates[0]
        used_refs.add(section["ref"])
        matched[node["index"]] = {"section": section, "match_kind": match_kind}

    titles = Counter((title_key(node["title"]), node["level"]) for node in nodes)
    numbers: dict[int, str] = {}
    number_sources: dict[int, str] = {}
    for node in nodes:
        index = node["index"]
        match = matched.get(index)
        body_number = LEADING_NUMBER.match(match["section"]["heading"]) if match else None
        toc_number = toc_numbers.get(title_key(node["title"]))
        if toc_number and (
            titles[(title_key(node["title"]), node["level"])] != 1
            or toc_number.count(".") != node["level"]
        ):
            toc_number = None
        if body_number and body_number.group(1).count(".") == node["level"]:
            number = body_number.group(1)
            if toc_number and number != toc_number:
                continue
            numbers[index] = number
            number_sources[index] = "body_heading+toc" if toc_number else "body_heading"
        elif toc_number:
            numbers[index] = toc_number
            number_sources[index] = "toc"

    # A chapter omitted from the printed TOC can still inherit an unambiguous
    # chapter number from its numbered descendants.
    for node in nodes:
        if node["level"] != 0 or node["index"] in numbers:
            continue
        descendants = [
            value.split(".")[0]
            for other, value in numbers.items()
            if other > node["index"] and _is_descendant(nodes, other, node["index"])
        ]
        if descendants and len(set(descendants)) == 1:
            numbers[node["index"]] = descendants[0]
            number_sources[node["index"]] = "descendant_numbering"

    paths: dict[int, list[str]] = {}
    for node in nodes:
        index = node["index"]
        parent = node["parent_index"]
        label = f"{numbers[index]} {node['title']}" if index in numbers else node["title"]
        paths[index] = [*(paths[parent] if parent is not None else []), label]

    by_ref = {match["section"]["ref"]: (node, match)
              for node in nodes if (match := matched.get(node["index"]))}
    normalized = []
    for section in sections:
        entry = {**section, "order_corrected": False, "normalization_source": "pdf_outline"}
        if section["ref"] in by_ref:
            node, match = by_ref[section["ref"]]
            index = node["index"]
            entry.update(
                role="outline_section",
                section_number=numbers.get(index),
                normalized_heading_path=paths[index],
                normalization_status="matched",
                outline_index=index,
                outline_parent_index=node["parent_index"],
                outline_destination={"page_no": node["page_no"], "y": node["y"]},
                match_confidence=(
                    "medium" if match["match_kind"] in {"joined_words", "next_page"}
                    else "high"
                ),
                match_evidence=["pdf_outline", match["match_kind"],
                                *([number_sources[index]] if index in number_sources else [])],
            )
        else:
            entry.update(
                role="noncanonical_heading",
                section_number=None,
                normalized_heading_path=[],
                normalization_status="noncanonical",
            )
        normalized.append(entry)
    return normalized


def _is_descendant(nodes: list[dict], child: int, parent: int) -> bool:
    current = nodes[child]["parent_index"]
    while current is not None:
        if current == parent:
            return True
        current = nodes[current]["parent_index"]
    return False
