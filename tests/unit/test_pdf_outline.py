"""Regression for bookmark hierarchy when the parser emits headings out of order."""

from manuora.ingestion.pdf_outline import normalize_from_pdf_outline

import json

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size

from manuora.ingestion.outline_quality import evaluate_outline_quality


def test_outline_joins_misordered_and_repeated_headings() -> None:
    def node(index, title, level, parent, page, y):
        return dict(index=index, title=title, level=level, parent_index=parent,
                    page_no=page, y=y)

    def section(ref, heading, page, y):
        return dict(ref=ref, heading=heading, page_no=page, anchor_y=y,
                    level=1, provisional_heading_path=[heading])

    nodes = [
        node(0, "Files & synchronization", 0, None, 25, 720),
        node(1, "Accessing your files", 1, 0, 25, 573),
        node(2, "Problem", 2, 1, 25, 400),
        node(3, "Problem", 2, 1, 25, 300),
        node(4, "System Requirements", 2, 1, 26, 500),
    ]
    sections = [
        section("first", "5.1 Accessing your files", 25, 561),
        section("problem_two", "Problem", 25, 290),
        section("ui_label", "All files", 25, 200),
        section("chapter", "FILES & SYNCHRONIZATION", 25, 625),
        section("problem_one", "Problem", 25, 390),
        section("requirements", "5.1.3 SystemRequirements", 26, 490),
    ]

    result = normalize_from_pdf_outline(
        sections, nodes,
        {"files synchronization": "5", "accessing your files": "5.1"},
    )
    by_ref = {row["ref"]: row for row in result}

    assert by_ref["first"]["normalized_heading_path"] == [
        "5 Files & synchronization", "5.1 Accessing your files"
    ]
    assert by_ref["problem_one"]["outline_index"] == 2
    assert by_ref["problem_two"]["outline_index"] == 3
    assert by_ref["problem_two"]["match_evidence"][1] == "position"
    assert by_ref["requirements"]["normalized_heading_path"][-1] == (
        "5.1.3 System Requirements"
    )
    assert by_ref["requirements"]["match_confidence"] == "medium"
    assert by_ref["ui_label"]["normalization_status"] == "noncanonical"


def reference(tmp_path, entries, **updates):
    path = tmp_path / "trusted_toc.json"
    path.write_text(json.dumps(dict(
        schema_version=1, source_pdf_sha256="source", reference_type="reviewed_native_toc",
        independent_of_heading_detector=True, reviewed=True,
        provenance="Manually reviewed native PDF TOC with physical destinations", entries=entries,
        **updates,
    )), encoding="utf-8")
    return path


def outline_document():
    doc = DoclingDocument(name="outline quality fixture")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=600, height=800))
    return doc


@pytest.mark.parametrize("found,status", [(6, "FAIL"), (7, "WARN"), (10, "PASS")])
def test_outline_failure_threshold_requires_trusted_reference(tmp_path, monkeypatch, found, status):
    from manuora.ingestion import outline_quality
    from manuora.ingestion.files import sha256_file

    doc = outline_document()
    entries = [dict(entry_id=str(i), title=f"Heading number {i}", page_no=1, y_top=50 + i * 50)
               for i in range(10)]
    for entry in entries[:found]:
        add_text(doc, entry["title"], 1, entry["y_top"])
    path = reference(tmp_path, entries)
    result = evaluate_outline_quality(doc, {}, source_sha256="source", trusted_outline_path=path)
    assert result["outline_heading_coverage"] == found / 10
    assert result["outline_quality"]["status"] == status
    assert result["outline_quality"]["fail_gate_enabled"] is True

    source = tmp_path / "source.pdf"
    source.write_bytes(b"source identity fixture; bookmark reader mocked")
    monkeypatch.setattr(outline_quality, "_bookmarks",
                        lambda path: [outline_quality.OutlineEntry(**entry) for entry in entries])
    for trusted in (False, True):
        result = evaluate_outline_quality(doc, {}, source_sha256=sha256_file(source),
                                          source_path=source, trust_pdf_bookmarks=trusted)
        assert result["outline_quality"]["reference_trusted"] is trusted
        assert result["outline_quality"]["status"] == (status if trusted else "WARN" if found < 10 else "PASS")


def add_text(doc, text, page, y, label=DocItemLabel.TEXT):
    return doc.add_text(label=label, text=text, prov=ProvenanceItem(page_no=page,
        charspan=(0, len(text)), bbox=BoundingBox(l=40, t=y, r=400, b=y + 20)))


def test_independent_recall_and_specificity_separate_content_from_detector(tmp_path):
    doc = outline_document()
    add_text(doc, "1 Setup", 1, 50, DocItemLabel.SECTION_HEADER)
    # Title exists, but was not recognized as a heading by the parser.
    add_text(doc, "1.1 Alpha controls", 1, 200)
    a = add_text(doc, "Open the control panel", 1, 240)
    b = add_text(doc, "Review all saved changes", 2, 100)
    # Neither a running header nor a TOC elsewhere may rescue the missing title.
    add_text(doc, "Beta controls", 2, 10, DocItemLabel.PAGE_HEADER)
    add_text(doc, "Beta controls ........ 2", 1, 100)
    entries = [dict(entry_id="root", title="Setup", page_no=1, y_top=50),
               dict(entry_id="a", title="Alpha controls", parent_id="root", page_no=1, y_top=200),
               dict(entry_id="b", title="Beta controls", parent_id="root", page_no=2, y_top=50)]
    path = reference(tmp_path, entries)
    context = {a.self_ref: {"normalized_heading_path": ["1 Setup"]},
               b.self_ref: {"normalized_heading_path": ["1 Setup", "1.1 Alpha controls"]}}
    result = evaluate_outline_quality(doc, context, source_sha256="source", trusted_outline_path=path)
    assert result["outline_heading_coverage"] == pytest.approx(2 / 3)
    found = result["outline_quality"]["entries"]
    assert found[1]["parsed_label"] == "text"
    assert found[2]["status"] == "missing"
    issues = {item["block_id"]: item for item in result["section_specificity_details"]["issues"]}
    assert issues[a.self_ref]["status"] == "ancestor_only"
    assert issues[b.self_ref]["status"] == "wrong_branch"
    assert result["section_specificity"] < .5

    context[a.self_ref]["normalized_heading_path"] = ["Setup", "Alpha controls", "Optional detail"]
    context[b.self_ref]["normalized_heading_path"] = ["Setup", "Beta controls"]
    result = evaluate_outline_quality(doc, context, source_sha256="source", trusted_outline_path=path)
    # Missing Beta title remains missing even with a correct assigned path.
    assert result["outline_heading_coverage"] == pytest.approx(2 / 3)
    assert all(item["block_id"] not in {a.self_ref, b.self_ref}
               for item in result["section_specificity_details"]["issues"])


def test_no_reference_and_ambiguous_destinations_do_not_fabricate_specificity(tmp_path):
    doc = outline_document()
    body = add_text(doc, "All content has one broad heading", 1, 300)
    context = {body.self_ref: {"normalized_heading_path": ["Setup"]}}
    result = evaluate_outline_quality(doc, context, source_sha256="source")
    assert result["outline_heading_coverage"] is None
    assert result["section_specificity"] is None
    assert result["section_specificity_details"]["dominant_section_token_ratio"] == 1
    assert result["section_specificity_details"]["status"] == "UNMEASURED"
    path = reference(tmp_path, [dict(entry_id="root", title="Setup", page_no=1),
                               dict(entry_id="child", title="Controls", parent_id="root", page_no=1)])
    result = evaluate_outline_quality(doc, context, source_sha256="source", trusted_outline_path=path)
    assert result["section_specificity"] is None
    assert result["section_specificity_details"]["assessment_coverage"] == 0
    assert "ambiguous" in result["section_specificity_details"]["unassessed_blocks"][0]["reason"]


def test_repeated_titles_one_to_one_and_reference_guards(tmp_path):
    doc = outline_document()
    add_text(doc, "Notes", 1, 100)
    path = reference(tmp_path, [dict(entry_id="a", title="Notes", page_no=1, y_top=100),
                               dict(entry_id="b", title="Notes", page_no=1, y_top=300),
                               dict(entry_id="external", title="External page"),
                               dict(entry_id="skip", title="Cover", page_no=1, exclusion_reason="Reviewed non-section cover")])
    result = evaluate_outline_quality(doc, {}, source_sha256="source", trusted_outline_path=path)
    assert result["outline_heading_coverage"] == .5
    assert len(result["outline_quality"]["excluded_entries"]) == 2
    with pytest.raises(ValueError, match="source hash"):
        evaluate_outline_quality(doc, {}, source_sha256="different", trusted_outline_path=path)
    data = json.loads(path.read_text())
    data["independent_of_heading_detector"] = False
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_outline_quality(doc, {}, source_sha256="source", trusted_outline_path=path)

    multi = outline_document()
    item = add_text(multi, "Notes\nOther", 1, 100)
    item.prov[0].charspan = (0, 6)
    item.prov.append(ProvenanceItem(page_no=2, charspan=(6, 11), bbox=item.prov[0].bbox))
    path = reference(tmp_path, [dict(entry_id="n", title="Notes", page_no=2)])
    result = evaluate_outline_quality(multi, {}, source_sha256="source", trusted_outline_path=path)
    assert result["outline_heading_coverage"] == 0  # Title is on page 1 only.


def test_reads_real_pdf_bookmarks_with_xyz_and_page_only_destinations(tmp_path):
    from manuora.ingestion.files import sha256_file

    # Tiny real PDF: two bookmarks, one XYZ destination and one page-only Fit.
    objects = [b"<< /Type /Catalog /Pages 2 0 R /Outlines 4 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] >>",
               b"<< /Type /Outlines /First 5 0 R /Last 6 0 R /Count 2 >>",
               b"<< /Title (Setup) /Parent 4 0 R /Next 6 0 R /Dest [3 0 R /XYZ 0 700 0] >>",
               b"<< /Title (Controls) /Parent 4 0 R /Prev 5 0 R /Dest [3 0 R /Fit] >>"]
    data, offsets = b"%PDF-1.4\n", []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(data)
    data += b"xref\n0 7\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    data += f"trailer << /Size 7 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    source = tmp_path / "outline.pdf"
    source.write_bytes(data)
    doc = outline_document()
    add_text(doc, "Setup", 1, 100)
    add_text(doc, "Controls", 1, 200)
    result = evaluate_outline_quality(doc, {}, source_sha256=sha256_file(source), source_path=source)
    assert result["outline_quality"]["reference_type"] == "pdf_bookmarks"
    assert result["outline_heading_coverage"] == 1
