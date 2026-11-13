"""Regression examples for chapter headings emitted in the wrong order."""

from manuora.ingestion.sections import normalize_sections


def test_numbered_children_attach_to_their_chapters() -> None:
    def section(ref: str, page: int, level: int, heading: str) -> dict:
        return {
            "ref": ref,
            "page_no": page,
            "level": level,
            "heading": heading,
            "provisional_heading_path": [heading],
        }

    raw = [
        section("part_ii", 27, 1, "Part II - Workflows"),
        section("old_chapter", 28, 2, "Import and manage patient data"),
        section("six_one", 33, 3, "6.1 About this workflow"),
        section("chapter_six", 33, 2, "Create an initial plan"),
        section("six_three", 34, 3, "6.3 Planning inputs"),
        section("six_three_one", 35, 4, "6.3.1 Record the external beam contribution"),
        section("part_iv", 69, 1, "Part IV Appendices"),
        section("twelve", 88, 1, "12"),
        section("chapter_twelve", 88, 2, "Calculation definitions"),
        section("chapter_thirteen", 93, 2, "Regulatory and safety information"),
        section("thirteen_one", 93, 3, "13.1 Regulatory classification"),
    ]
    pages = {
        33: [{"text": "6"}],
        34: [],
        35: [],
        88: [{"text": "12"}],
        93: [{"text": "13"}],
    }

    result = normalize_sections(raw, pages)
    by_ref = {item["ref"]: item for item in result}

    assert [item["ref"] for item in result].index("chapter_six") < [
        item["ref"] for item in result
    ].index("six_one")
    assert by_ref["chapter_six"]["order_corrected"] is True
    assert by_ref["six_one"]["normalized_heading_path"] == [
        "Part II - Workflows",
        "6 Create an initial plan",
        "6.1 About this workflow",
    ]
    assert by_ref["six_three_one"]["normalized_heading_path"] == [
        "Part II - Workflows",
        "6 Create an initial plan",
        "6.3 Planning inputs",
        "6.3.1 Record the external beam contribution",
    ]
    assert by_ref["thirteen_one"]["normalized_heading_path"] == [
        "Part IV Appendices",
        "13 Regulatory and safety information",
        "13.1 Regulatory classification",
    ]
    assert raw[2]["provisional_heading_path"] == ["6.1 About this workflow"]


def test_numbered_toc_overrides_unreliable_body_levels() -> None:
    headings = [
        ("intro", "Introduction", 1),
        ("discover", "Discovering FreeCAD", 1),
        ("read_more", "Read more", 1),
        ("install", "Installing", 2),
        ("versions", "FreeCAD versions", 1),
        ("windows", "Installing on Windows", 1),
    ]
    sections = [
        {
            "ref": ref,
            "page_no": page,
            "level": level,
            "heading": title,
            "provisional_heading_path": [title],
        }
        for page, (ref, title, level) in enumerate(headings, start=4)
    ]
    pages = {
        2: [
            {
                "label": "document_index",
                "text": "| Introduction | 1.1 |\n| Discovering FreeCAD | 1.2 |\n| Installing | 1.2.2 |",
            }
        ],
        3: [{"label": "table", "text": "| Installing on Windows | 1.2.2.1 |"}],
    }

    result = normalize_sections(sections, pages)
    by_ref = {item["ref"]: item for item in result}

    assert by_ref["install"]["normalized_heading_path"] == [
        "1.2 Discovering FreeCAD",
        "1.2.2 Installing",
    ]
    assert by_ref["windows"]["normalized_heading_path"] == [
        "1.2 Discovering FreeCAD",
        "1.2.2 Installing",
        "1.2.2.1 Installing on Windows",
    ]
    assert by_ref["versions"]["role"] == "local_heading"
    assert by_ref["versions"]["normalized_heading_path"] == by_ref["install"]["normalized_heading_path"]
    assert by_ref["windows"]["normalization_source"] == "numbered_toc"


def test_level_one_chapter_title_anchors_next_page_subsection() -> None:
    sections = [
        {"ref": "part", "page_no": 5, "level": 1, "heading": "Part I - Foundations",
         "provisional_heading_path": ["Part I - Foundations"]},
        {"ref": "chapter", "page_no": 6, "level": 1, "heading": "Overview",
         "provisional_heading_path": ["Overview"]},
        {"ref": "child", "page_no": 7, "level": 2, "heading": "1.1 What the product does",
         "provisional_heading_path": ["Overview", "1.1 What the product does"]},
    ]
    result = {entry["ref"]: entry for entry in normalize_sections(sections, {})}
    assert result["child"]["normalization_status"] == "matched"
    assert result["child"]["normalized_heading_path"] == [
        "Part I - Foundations", "1 Overview", "1.1 What the product does"
    ]


def test_same_page_chapter_beats_previous_page_local_heading() -> None:
    sections = [
        {"ref": "local", "page_no": 29, "level": 1, "heading": "Tandem & Ring",
         "provisional_heading_path": ["Tandem & Ring"]},
        {"ref": "chapter", "page_no": 30, "level": 1, "heading": "Fully automated plan",
         "provisional_heading_path": ["Fully automated plan"]},
        {"ref": "child", "page_no": 30, "level": 2, "heading": "6.1 About this workflow",
         "provisional_heading_path": ["6.1 About this workflow"]},
    ]
    result = {entry["ref"]: entry for entry in normalize_sections(sections, {})}
    assert result["local"]["normalization_status"] == "unmatched"
    assert result["child"]["normalized_heading_path"] == [
        "6 Fully automated plan", "6.1 About this workflow"
    ]
