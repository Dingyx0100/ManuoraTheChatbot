"""Build reviewable section paths from Docling headings and PDF page blocks."""

import re
import unicodedata
from collections import Counter, defaultdict


NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+")
CHAPTER_NUMBER = re.compile(r"^\d{1,2}$")
PART_HEADING = re.compile(r"^Part\s+[IVXLCDM]+\b", re.IGNORECASE)
TOC_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+(?:\.\d+)+)\s*\|$")


def _section_number(heading: str) -> tuple[int, ...] | None:
    match = NUMBERED_HEADING.match(heading)
    return tuple(map(int, match.group(1).split("."))) if match else None


def _title_key(title: str) -> str:
    """Ignore case and punctuation when joining a TOC title to a body heading."""
    folded = unicodedata.normalize("NFKC", title).casefold()
    return re.sub(r"[\W_]+", " ", folded).strip()


def _numbered_toc(
    sections: list[dict], pages: dict[int, list[dict]]
) -> dict[str, dict] | None:
    """Return trusted TOC matches, or None if the outline cannot be joined safely."""
    rows = []
    previous_page = None
    for page_no, blocks in sorted(pages.items()):
        has_toc_start = any(
            block.get("label") == "document_index"
            or (
                block.get("label") == "section_header"
                and block.get("text", "").strip().casefold()
                in {"contents", "table of contents"}
            )
            for block in blocks
        )
        table_continuation = (
            bool(blocks)
            and all(block.get("label") in {"table", "picture"} for block in blocks)
            and any(block.get("label") == "table" for block in blocks)
        )
        if previous_page is None:
            if not has_toc_start:
                continue
        elif page_no != previous_page + 1 or not (has_toc_start or table_continuation):
            break
        previous_page = page_no
        for block in blocks:
            if block.get("label") not in {"document_index", "table"}:
                continue
            for line in block["text"].splitlines():
                if match := TOC_ROW.fullmatch(line.strip()):
                    rows.append((match.group(1).strip(), match.group(2)))
    if len(rows) < 3 or len({number for _, number in rows}) != len(rows):
        return None

    body_by_title: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, section in enumerate(sections):
        body_by_title[_title_key(section["heading"])].append((index, section))

    matches = []
    for title, number in rows:
        candidates = body_by_title[_title_key(title)]
        if len(candidates) != 1:
            return None
        matches.append((title, number, *candidates[0]))
    if any(left[2] >= right[2] for left, right in zip(matches, matches[1:])):
        return None

    title_by_number = {number: title for title, number in rows}
    if any(
        number.rsplit(".", 1)[0] not in title_by_number
        for _, number in rows
        if number.count(".") > 1
    ):
        return None

    matched_by_ref = {}
    for title, number, _, section in matches:
        parts = number.split(".")
        path = [
            f"{prefix} {title_by_number[prefix]}"
            for depth in range(2, len(parts) + 1)
            if (prefix := ".".join(parts[:depth])) in title_by_number
        ]
        matched_by_ref[section["ref"]] = {"section_number": number, "path": path}
    return matched_by_ref


def _normalize_from_toc(sections: list[dict], matches: dict[str, dict]) -> list[dict]:
    """Use matched TOC entries as section anchors; keep other headings noncanonical."""
    normalized = []
    current_path: list[str] = []
    for section in sections:
        entry = {**section, "order_corrected": False, "normalization_source": "numbered_toc"}
        if match := matches.get(section["ref"]):
            current_path = match["path"]
            entry.update(
                role="toc_section",
                section_number=match["section_number"],
                normalized_heading_path=current_path.copy(),
                normalization_status="matched",
            )
        else:
            entry.update(
                role="local_heading",
                section_number=None,
                normalized_heading_path=current_path.copy(),
                normalization_status=(
                    "inherited_toc_context" if current_path else "outside_toc"
                ),
            )
        normalized.append(entry)
    return normalized


def normalize_sections(sections: list[dict], pages: dict[int, list[dict]]) -> list[dict]:
    """Use a trusted numbered TOC, or fall back to numbered body headings.

    TOC titles must each match one body heading in order and have complete
    numeric parents. Otherwise, a chapter title is paired only when its page
    has an unambiguous N.1 heading or a standalone N block. The original
    Docling document and page-block order are never changed.
    """
    if toc_matches := _numbered_toc(sections, pages):
        return _normalize_from_toc(sections, toc_matches)

    first_subsections: dict[int, set[int]] = defaultdict(set)
    numbered_headings: dict[tuple[int, ...], str] = {}
    number_counts: Counter[tuple[int, ...]] = Counter()
    for section in sections:
        number = _section_number(section["heading"])
        if number is None:
            continue
        number_counts[number] += 1
        numbered_headings[number] = section["heading"]
        if len(number) == 2 and number[1] == 1:
            first_subsections[section["page_no"]].add(number[0])

    page_markers = {
        page_no: {
            int(block["text"].strip())
            for block in blocks
            if CHAPTER_NUMBER.fullmatch(block["text"].strip())
        }
        for page_no, blocks in pages.items()
    }

    chapters: dict[int, dict] = {}
    title_numbers: dict[str, int] = {}
    conflicting_chapters: set[int] = set()
    current_part: str | None = None
    for section in sections:
        heading = section["heading"]
        if section["level"] == 1 and PART_HEADING.match(heading):
            current_part = heading
            continue
        if (section["level"] not in {1, 2} or _section_number(heading) is not None
                or CHAPTER_NUMBER.fullmatch(heading)):
            continue
        page_no = section["page_no"]
        subsection_numbers = first_subsections.get(page_no, set())
        priority = 2
        # Some manuals put an unnumbered level-one chapter title on the page
        # before its first numbered subsection.
        if not subsection_numbers and section["level"] == 1:
            subsection_numbers = first_subsections.get(page_no + 1, set())
            priority = 1
        marker_numbers = page_markers.get(page_no, set())
        if len(subsection_numbers) == 1:
            candidates = subsection_numbers
            if marker_numbers and not candidates.issubset(marker_numbers):
                continue
        else:
            candidates = marker_numbers if not subsection_numbers else set()
        if len(candidates) != 1:
            continue
        chapter_number = next(iter(candidates))
        if chapter_number in chapters:
            previous = chapters[chapter_number]
            if priority < previous["priority"]:
                continue
            if priority == previous["priority"]:
                conflicting_chapters.add(chapter_number)
                continue
            title_numbers.pop(previous["ref"], None)
            conflicting_chapters.discard(chapter_number)
        title_numbers[section["ref"]] = chapter_number
        chapters[chapter_number] = {
            "ref": section["ref"],
            "page_no": page_no,
            "priority": priority,
            "path": [
                *([current_part] if current_part else []),
                f"{chapter_number} {heading}",
            ],
        }

    for chapter_number in conflicting_chapters:
        title_numbers.pop(chapters[chapter_number]["ref"], None)
        chapters.pop(chapter_number)
    unique_numbered_headings = {
        number: heading
        for number, heading in numbered_headings.items()
        if number_counts[number] == 1
    }

    normalized = []
    current_path: list[str] = []
    for section in sections:
        entry = {
            **section,
            "order_corrected": False,
            "normalization_source": "body_numbering",
        }
        heading = section["heading"]
        number = _section_number(heading)
        if section["level"] == 1 and PART_HEADING.match(heading):
            entry.update(
                role="part",
                section_number=None,
                normalized_heading_path=[heading],
                normalization_status="matched",
            )
        elif section["ref"] in title_numbers:
            chapter_number = title_numbers[section["ref"]]
            chapter = chapters[chapter_number]
            entry.update(
                role="chapter_title",
                section_number=str(chapter_number),
                normalized_heading_path=chapter["path"],
                normalization_status="matched",
            )
        elif (
            section["level"] == 1
            and CHAPTER_NUMBER.fullmatch(heading)
            and int(heading) in chapters
        ):
            entry.update(
                role="chapter_number_marker",
                section_number=heading,
                normalized_heading_path=chapters[int(heading)]["path"],
                normalization_status="matched",
            )
        elif number is not None and number[0] in chapters:
            parent_headings = [
                unique_numbered_headings.get(number[:depth])
                for depth in range(2, len(number))
            ]
            entry.update(
                role="numbered_section",
                section_number=".".join(map(str, number)),
                normalized_heading_path=[
                    *chapters[number[0]]["path"],
                    *(parent for parent in parent_headings if parent is not None),
                    heading,
                ],
                normalization_status=(
                    "matched" if all(parent_headings) else "missing_numbered_parent"
                ),
            )
        else:
            if (number is None and current_path and not heading.isupper()
                    and heading.strip().casefold() not in {"contents", "table of contents"}):
                entry.update(
                    role="local_heading",
                    section_number=None,
                    normalized_heading_path=current_path.copy(),
                    normalization_status="inherited_body_context",
                )
            else:
                entry.update(
                    role="unmatched",
                    section_number=None,
                    normalized_heading_path=section["provisional_heading_path"],
                    normalization_status="unmatched",
                )
        if entry["normalization_status"] == "matched":
            current_path = entry["normalized_heading_path"].copy()
        elif entry["role"] == "unmatched":
            current_path = []
        normalized.append(entry)

    # Docling sometimes emits the first subsection before the chapter banner on
    # the same page. Move only the section record, leaving page blocks untouched.
    for chapter_number, chapter in chapters.items():
        title_index = next(
            index
            for index, item in enumerate(normalized)
            if item["ref"] == chapter["ref"]
        )
        earlier_children = [
            index
            for index, item in enumerate(normalized[:title_index])
            if item["page_no"] == chapter["page_no"]
            and (number := _section_number(item["heading"])) is not None
            and number[0] == chapter_number
        ]
        if earlier_children:
            title = normalized.pop(title_index)
            title["order_corrected"] = True
            normalized.insert(earlier_children[0], title)

    return normalized
