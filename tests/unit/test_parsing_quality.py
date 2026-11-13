"""Count retention regressions; synthetic PDFs only, no corpus re-evaluation."""

import pytest
from docling_core.types.doc import (
    BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size,
    TableCell, TableData,
)

from manuora.ingestion.profile_schemas import TextCoverageSettings
from manuora.ingestion.profiler import profile_document
from manuora.ingestion.quality import evaluate_parsing_quality
from manuora.ingestion import text_coverage
from manuora.ingestion.chunk_preservation import evaluate_chunk_preservation


def write_pdf(path, text):
    """A real one-page PDF with native text, without another test dependency."""
    lines = [text[i:i + 80] for i in range(0, len(text), 80)]
    stream = b"BT /F1 5 Tf 40 740 Td 8 TL\n"
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream += f"({safe}) Tj T*\n".encode("ascii")
    stream += b"ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    ]
    data = b"%PDF-1.4\n"
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    data += f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    path.write_bytes(data)


def provenance(page, end, start=0):
    return ProvenanceItem(page_no=page, charspan=(start, end),
                          bbox=BoundingBox(l=40, t=700, r=400, b=500))


def document(pages=1):
    doc = DoclingDocument(name="coverage fixture")
    for number in range(1, pages + 1):
        doc.add_page(page_no=number, size=Size(width=600, height=800))
    return doc


@pytest.fixture
def tokenizer(monkeypatch):
    class FixtureTokenizer:
        def encode(self, value, *, add_special_tokens, truncation):
            assert add_special_tokens is False and truncation is False
            assert "\n" not in value
            return value.split()

    result = FixtureTokenizer()
    monkeypatch.setattr(text_coverage, "_load_tokenizer", lambda *args: result)
    return result


def test_truncated_page_fails_despite_passing_presence_gate(tmp_path, tokenizer):
    source = tmp_path / "manual.pdf"
    write_pdf(source, "word " * 500)  # 2000 non-whitespace characters.
    profile = profile_document(source)
    doc = document()
    doc.add_text(label=DocItemLabel.TEXT, text="word " * 15, prov=provenance(1, 75))
    parsed = tmp_path / "parsed.json"
    doc.save_as_json(parsed)
    sections = tmp_path / "sections.jsonl"
    sections.write_text("", encoding="utf-8")
    report = evaluate_parsing_quality(profile, parsed, sections, source_path=source)
    row = report.page_quality[0]
    assert report.schema_version == 4
    assert "native_text_page_presence_rate" not in report.model_dump()
    assert "native_text_page_coverage" not in report.model_dump()
    assert row.native_text_chars == 2000 and row.parsed_coverage_chars == 60
    assert row.native_character_coverage_ratio == pytest.approx(.03)
    assert row.native_text_tokens == 500 and row.parsed_text_tokens == 15
    assert row.native_token_coverage_ratio == pytest.approx(.03)
    assert report.status == "FAIL" and report.text_coverage_failed_pages == [1]


def test_trusted_outline_failure_propagates_to_parse_report(tmp_path, tokenizer):
    import json

    source = tmp_path / "manual.pdf"
    write_pdf(source, "word " * 100)
    profile = profile_document(source)
    doc = document()
    doc.add_text(label=DocItemLabel.TEXT, text="word " * 100, prov=provenance(1, 500))
    parsed, sections, reference = (tmp_path / name for name in ("parsed.json", "sections.jsonl", "toc.json"))
    doc.save_as_json(parsed)
    sections.write_text("", encoding="utf-8")
    reference.write_text(json.dumps(dict(
        source_pdf_sha256=profile.source_sha256, reference_type="reviewed_native_toc",
        independent_of_heading_detector=True, reviewed=True, provenance="Independent fixture",
        entries=[dict(entry_id="missing", title="Missing heading", page_no=1)],
    )), encoding="utf-8")
    report = evaluate_parsing_quality(profile, parsed, sections, source_path=source,
                                     trusted_outline_path=reference)
    assert report.native_character_coverage_ratio == 1
    assert report.outline_quality["status"] == "FAIL"
    assert report.status == "FAIL"
    assert "Trusted outline heading coverage is below 70%" in report.reasons


def test_whole_page_counts_include_picture_text_and_table_cells_once():
    doc = document(2)
    pic = doc.add_picture(prov=provenance(1, 0))
    doc.add_text(label=DocItemLabel.TEXT, text="nested code", parent=pic, prov=provenance(1, 11))
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1, table_cells=[TableCell(
        text="table cell", start_row_offset_idx=0, end_row_offset_idx=1,
        start_col_offset_idx=0, end_col_offset_idx=1,
    )]), prov=provenance(1, 0))
    doc.add_text(label=DocItemLabel.TEXT, text="table cell", parent=table, prov=provenance(1, 10))
    item = doc.add_text(label=DocItemLabel.TEXT, text="first second", prov=provenance(1, 6))
    item.prov.append(provenance(2, 12, 6))
    values, notes = text_coverage.parsed_page_text(doc)
    assert values[1].count("table cell") == 1
    assert "nested code" in values[1]
    assert "first" in values[1] and "second" not in values[1]
    assert values[2] == "second"
    assert not any(notes.values())
    item.prov[1].charspan = (0, 12)  # Ambiguous overlapping spans, never duplicate.
    _, notes = text_coverage.parsed_page_text(doc)
    assert notes[1] and notes[2]


def test_thresholds_missing_tokens_and_blank_pages(tmp_path, tokenizer, monkeypatch):
    source = tmp_path / "manual.pdf"
    write_pdf(source, "word " * 100)
    profile = profile_document(source)
    doc = document()
    item = doc.add_text(label=DocItemLabel.TEXT, text="word " * 90, prov=provenance(1, 450))
    settings = TextCoverageSettings()
    measure = lambda: text_coverage.measure_text_coverage(profile, doc, source, settings)
    assert measure()["pages"][1]["coverage_status"] == "WARN"
    item.text = "a" * 400  # Character count alone must not hide token loss.
    result = measure()
    assert result["pages"][1]["native_character_coverage_ratio"] == 1
    assert result["pages"][1]["native_token_coverage_ratio"] == .01
    assert result["pages"][1]["coverage_status"] == "FAIL"
    item.text = "word " * 100
    assert measure()["pages"][1]["coverage_status"] == "PASS"
    item.text = "word " * 200
    result = measure()
    assert result["pages"][1]["native_character_coverage_ratio"] == 2
    assert result["native_character_coverage_ratio"] == 1
    monkeypatch.setattr(text_coverage, "_load_tokenizer", lambda *args: (_ for _ in ()).throw(OSError()))
    result = measure()
    assert result["token_counting_status"] == "unavailable_tokenizer"
    assert result["pages"][1]["native_token_coverage_ratio"] is None
    assert result["text_coverage_unmeasured_pages"] == [1]
    with pytest.raises(ValueError, match="must not exceed"):
        text_coverage.measure_text_coverage(profile, doc, source, TextCoverageSettings(fail_below=.99))
    write_pdf(source, "")
    with pytest.raises(ValueError, match="source hash"):
        measure()
    blank = profile_document(source)
    result = text_coverage.measure_text_coverage(blank, document(), source, settings)
    assert result["pages"][1]["coverage_status"] == "NOT_APPLICABLE"
    assert result["native_character_coverage_ratio"] is None


def test_excess_output_cannot_hide_a_truncated_page(tmp_path):
    source = tmp_path / "manual.pdf"
    write_pdf(source, "word " * 500)
    profile = profile_document(source)
    profile.page_count = 2
    profile.page_profiles.append(profile.page_profiles[0].model_copy(update={"page_number": 2}))
    doc = document(2)
    doc.add_text(label=DocItemLabel.TEXT, text="word " * 15, prov=provenance(1, 75))
    doc.add_text(label=DocItemLabel.TEXT, text="word " * 1000, prov=provenance(2, 5000))
    result = text_coverage.measure_text_coverage(profile, doc, None, TextCoverageSettings())
    assert result["native_character_coverage_ratio"] == pytest.approx(2060 / 4000)
    assert result["text_coverage_failed_pages"] == [1]
    assert result["native_token_coverage_ratio"] is None


def chunk(cid, text, refs, **extra):
    return dict(chunk_id=cid, text=text, source_block_ids=refs, refs=refs,
                source_pdf_sha256="pdf", parsed_sha256="parse", **extra)


def preservation(doc, chunks, tables=None, **extra):
    return evaluate_chunk_preservation(doc, chunks, tables or [],
                                      source_pdf_sha256="pdf", parsed_sha256="parse",
                                      include_all_blocks=True, **extra)


@pytest.mark.parametrize("recovered,status", [(94, "FAIL"), (95, "WARN"), (100, "PASS")])
def test_preservation_failure_threshold(recovered, status):
    doc = document()
    words = [f"word{i}" for i in range(100)]
    item = doc.add_text(label=DocItemLabel.TEXT, text=" ".join(words), prov=provenance(1, 689))
    result = preservation(doc, [chunk("a", " ".join(words[:recovered]), [item.self_ref])])
    assert result["chunk_content_recovery"] == recovered / 100
    assert result["status"] == status
    assert result["settings"]["content_recovery_fail_below"] == .95


def test_preservation_detects_missing_spans_wrong_content_and_duplicates():
    doc = document()

    def text(value, label=DocItemLabel.TEXT, parent=None):
        return doc.add_text(label=label, text=value, prov=provenance(1, len(value)), parent=parent)

    complete = text("Alpha beta gamma delta epsilon zeta")
    partial = text("Save project then review final output")
    wrong = text("Required evidence must remain searchable")
    repeated = text("Never disconnect power during this operation")
    header = text("Repeated running header", DocItemLabel.PAGE_HEADER)
    index = doc.add_table(label=DocItemLabel.DOCUMENT_INDEX, data=TableData(num_rows=1, num_cols=1,
        table_cells=[TableCell(text="Table of contents", start_row_offset_idx=0, end_row_offset_idx=1,
                              start_col_offset_idx=0, end_col_offset_idx=1)]), prov=provenance(1, 0))
    pic = doc.add_picture(prov=provenance(1, 0))
    nested = text("Import the essential library before execution", parent=pic)
    chunks = [
        chunk("a", "ALPHA beta gamma", [complete.self_ref]),
        chunk("b", "delta epsilon zeta", [complete.self_ref]),
        chunk("c", "Save project then", [partial.self_ref]),
        chunk("d", "Filler has exactly five words", [wrong.self_ref]),
        chunk("e", repeated.text + "\n" + repeated.text, [repeated.self_ref]),
        # Identical text elsewhere must not rescue an absent source reference.
        chunk("f", nested.text, []),
    ]
    report = preservation(doc, chunks)
    rows = {row["block_id"]: row for row in report["blocks"]}
    assert report["eligible_blocks"] == 5
    assert report["represented_blocks"] == 4
    assert rows[complete.self_ref]["recovery_ratio"] == 1
    assert rows[partial.self_ref]["recovery_ratio"] == .5
    assert rows[partial.self_ref]["omitted_spans"][0]["excerpt"] == "review final output"
    assert rows[wrong.self_ref]["recovery_ratio"] == 0
    assert set(report["missing_block_ids"]) == {wrong.self_ref, nested.self_ref}
    assert report["unreferenced_block_ids"] == [nested.self_ref]
    assert report["excessive_duplicate_token_occurrences"] == 6
    assert report["chunk_content_recovery"] == pytest.approx(15 / 29)
    excluded = {row["block_id"]: row["reason"] for row in report["excluded_blocks"]}
    assert all(ref in excluded for ref in (header.self_ref, index.self_ref, pic.self_ref))
    assert nested.self_ref not in excluded
    with pytest.raises(ValueError, match="parsed hash mismatch"):
        preservation(doc, [{**chunks[0], "parsed_sha256": "stale"}])
    with pytest.raises(ValueError, match="exclusion"):
        preservation(doc, chunks, exclusions={nested.self_ref: ""})
    result = preservation(doc, chunks, exclusions={nested.self_ref: "Reviewed code sample intentionally omitted"})
    assert result["eligible_blocks"] == 4


def test_preservation_corrected_rows_and_intentional_heading_repetition():
    doc = document()
    header = doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Synchronization settings", prov=provenance(1, 24))
    table = doc.add_table(data=TableData(num_rows=1, num_cols=1, table_cells=[TableCell(
        text="broken native table content", start_row_offset_idx=0, end_row_offset_idx=1,
        start_col_offset_idx=0, end_col_offset_idx=1,
    )]), prov=provenance(1, 0))
    tables = [dict(table_ref=table.self_ref, kind="configuration", source_pdf_sha256="pdf", parsed_sha256="parse",
                   rows=[dict(variable="pollInterval", default="30000", meaning="Poll the server", raw_row_indices=[1]),
                         dict(variable="syncInterval", default="60000", meaning="Sync all files", raw_row_indices=[2])])]
    first = f"{table.self_ref}/normalized_rows/0"
    second = f"{table.self_ref}/normalized_rows/1"
    chunks = [chunk("a", "pollInterval. Default: 30000. Meaning: Poll the server",
                    [table.self_ref, first, header.self_ref], headings=[header.text],
                    content_kind="corrected_table_row", table_ref=table.self_ref, table_row_index=0),
              chunk("b", "", [header.self_ref], headings=[header.text])]
    result = preservation(doc, chunks, tables)
    rows = {row["block_id"]: row for row in result["blocks"]}
    assert result["eligible_blocks"] == 3
    assert rows[first]["recovery_ratio"] == 1
    assert result["missing_block_ids"] == [second]
    assert rows[header.self_ref]["duplicate_token_occurrences"] == 2
    assert result["excessive_duplicate_token_occurrences"] == 0
    legacy = [{key: value for key, value in chunks[0].items() if key not in {"source_block_ids", "parsed_sha256"}}]
    assert preservation(doc, legacy, tables)["legacy_provenance_chunk_ids"] == ["a"]

    # A broad metadata claim must not let one occurrence recover two blocks.
    repeated_doc = document()
    first = repeated_doc.add_text(label=DocItemLabel.TEXT, text="Review the final result", prov=provenance(1, 23))
    second = repeated_doc.add_text(label=DocItemLabel.TEXT, text=first.text, prov=provenance(1, 23))
    single = chunk("single", first.text, [first.self_ref, second.self_ref])
    assert preservation(repeated_doc, [single])["chunk_content_recovery"] == .5
    single["text"] += "\n" + second.text
    recovered = preservation(repeated_doc, [single])
    assert recovered["chunk_content_recovery"] == 1
    assert recovered["excessive_duplicate_token_occurrences"] == 0


def test_chunk_assembly_preserves_explicit_ids_and_runs_metric(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from manuora.ingestion import retrieval_chunks

    doc = document()
    heading = doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Setup instructions", prov=provenance(1, 18))
    body = doc.add_text(label=DocItemLabel.TEXT, text="Open the configuration panel now", prov=provenance(1, 32))
    sections = [dict(ref=heading.self_ref, normalization_status="matched", normalized_heading_path=[heading.text])]

    class FixtureChunker:
        def __init__(self, **kwargs):
            pass

        def chunk(self, document):
            yield SimpleNamespace(text=body.text, meta=SimpleNamespace(doc_items=[body], headings=[heading.text]))

        def contextualize(self, raw):
            return raw.text

    monkeypatch.setattr(retrieval_chunks, "HybridChunker", FixtureChunker)
    source, parsed = tmp_path / "manual.pdf", tmp_path / "parsed.json"
    write_pdf(source, body.text)
    doc.save_as_json(parsed)
    chunks, report = retrieval_chunks.assemble_chunks(
        doc, sections, [], SimpleNamespace(product_id="product", version="1", language="en"),
        SimpleNamespace(id="manual", title="Manual", type="user_manual"), source, parsed,
        SimpleNamespace(count_tokens=lambda value: len(value.split()), max_tokens=512),
    )
    assert set(chunks[0]["source_block_ids"]) == {heading.self_ref, body.self_ref}
    assert chunks[0]["refs"] == [body.self_ref]
    assert chunks[0]["parsed_sha256"] == report["parsed_sha256"]
    assert report["source_to_chunk_preservation"]["chunk_content_recovery"] == 1
    assert report["source_to_chunk_preservation"]["status"] == "PASS"


def test_list_markers_are_counted_once():
    doc = document()
    doc.add_list_item(text="Feature A", marker="-", prov=provenance(1, 9))
    doc.add_list_item(text="- Feature B", marker="-", prov=provenance(1, 11))
    text, _ = text_coverage.parsed_page_text(doc)
    assert text[1] == "- Feature A\n- Feature B"


def test_nested_text_detection_and_assembly_recovery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from manuora.ingestion import retrieval_chunks
    from manuora.ingestion.failure_detection import detect_failures

    doc = document()
    picture = doc.add_picture(prov=provenance(1, 0))
    code = doc.add_text(label=DocItemLabel.CODE, text="import geometry from example library",
                        parent=picture, prov=provenance(1, 36))
    before = detect_failures(doc, chunks=[])
    assert before["counts"] == {"nested_text_missing_from_chunks": 1}
    # Simulate a serializer dropping the item. The final reconciliation must
    # recover it even with no resolved section. A metadata-only claim is not used.
    class EmptyChunker:
        def __init__(self, **kwargs):
            pass
        def chunk(self, document):
            return []
    monkeypatch.setattr(retrieval_chunks, "HybridChunker", EmptyChunker)
    source, parsed = tmp_path / "manual.pdf", tmp_path / "parsed.json"
    write_pdf(source, code.text)
    doc.save_as_json(parsed)
    chunks, report = retrieval_chunks.assemble_chunks(
        doc, [], [], SimpleNamespace(product_id="product", version="1", language="en"),
        SimpleNamespace(id="manual", title="Manual", type="user_manual"), source, parsed,
        SimpleNamespace(count_tokens=lambda value: len(value.split()), max_tokens=512))
    assert chunks[0]["text"] == code.text
    assert chunks[0]["context_fallback"] is True
    assert chunks[0]["source_block_ids"] == [code.self_ref]
    assert report["source_to_chunk_preservation"]["chunk_content_recovery"] == 1
    assert report["failure_detection"]["findings"] == []


def test_confidence_is_optional_hash_bound_and_not_a_quality_bypass(tmp_path):
    from types import SimpleNamespace
    from docling.datamodel.base_models import ConfidenceReport, PageConfidenceScores
    from manuora.ingestion.docling_parser import save_conversion_confidence
    from manuora.ingestion.failure_detection import read_confidence, detect_failures
    from manuora.ingestion.files import sha256_file

    source, parsed = tmp_path / "source.pdf", tmp_path / "parsed.json"
    source.write_bytes(b"source"); parsed.write_bytes(b"parse")
    assert read_confidence(parsed, sha256_file(source), sha256_file(parsed))["status"] == "unavailable"
    scores = ConfidenceReport(pages={1: PageConfidenceScores(parse_score=.3, layout_score=.95)})
    save_conversion_confidence(SimpleNamespace(confidence=scores), source, parsed)
    saved = read_confidence(parsed, sha256_file(source), sha256_file(parsed))
    assert saved["scores"]["pages"]["1"]["table_score"] is None
    assert detect_failures(document(), confidence=saved)["counts"]["low_docling_confidence"] == 1
    parsed.write_bytes(b"changed by OCR")
    assert read_confidence(parsed, sha256_file(source), sha256_file(parsed))["status"] == "stale"
