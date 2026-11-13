"""Regression for Docling cells split across columns and wrapped rows."""

from manuora.ingestion.table_normalization import _config_rows
from manuora.ingestion.failure_detection import configuration_text, detect_failures
from docling_core.types.doc import DoclingDocument, TableCell, TableData, Size, BoundingBox, ProvenanceItem


def test_callout_marker_kind():
    import pytest
    pytest.importorskip("pdfplumber")
    from manuora.ingestion.geometry_recovery import marker_kind
    assert marker_kind("\u00c1 Warning") == "warning"
    assert marker_kind("b Tip") == "tip"
    assert marker_kind("Note that this is prose") is None


def test_configuration_table_rejoins_split_values_and_rows() -> None:
    def cells(*values):
        return [{"text": value} for value in values]

    raw = [
        cells("Variable", "Default", "Default", "Meaning"),
        cells("remotePollInterval", "30000 sec)", "(30", "Poll interval."),
        cells("fullLocalDiscovery-", "3600000", "3600000", "Performs a full"),
        cells("Interval", "(1 hr)", "", "local discovery."),
    ]
    spec = {"expected_variables": ["remotePollInterval", "fullLocalDiscoveryInterval"]}

    rows = _config_rows(raw, spec)

    assert rows == [
        {
            "variable": "remotePollInterval", "default": "30000 (30 sec)",
            "meaning": "Poll interval.", "raw_row_indices": [1],
        },
        {
            "variable": "fullLocalDiscoveryInterval", "default": "3600000 (1 hr)",
            "meaning": "Performs a full local discovery.", "raw_row_indices": [2, 3],
        },
    ]


def test_cross_page_callout_preserves_partial_spans_and_rejects_conflicts():
    import copy
    import json
    from pathlib import Path
    import pytest
    pdfplumber = pytest.importorskip('pdfplumber')
    from manuora.ingestion.geometry_recovery import recover_cross_page_callout
    root=Path(__file__).resolve().parents[2]
    manifest=root/'data/evaluation/automatic_ingestion_v0_1/results.json'
    if not manifest.is_file():
        pytest.skip('Optional local PDF corpus is not installed')
    record=next(d for d in json.loads(manifest.read_text(encoding='utf-8'))['documents']
                if d['document_id']=='nextcloud_administration_manual_36')
    doc=json.loads((Path(record['output_dir'])/'parsed.json').read_text(encoding='utf-8'))
    marker=doc['texts'][4237]
    with pdfplumber.open(record['source_pdf'],pages=[206,207,208]) as pdf:
        p,n,wrong=pdf.pages
        result=recover_cross_page_callout(p,n,marker,doc['texts'])
        assert result['status']=='recovered'
        assert result['body_text']=='This setting is only taken into account in cron mode.'
        assert result['body_block_ids']==['#/texts/4236']
        assert result['source_spans'][0]['charspan']==[25,78]
        assert result['partial_block_ids']==['#/texts/4236']
        assert 'maintenance_window_start' not in result['body_text']
        changed=copy.deepcopy(doc['texts'])
        changed[4236]['prov'][1]['charspan']=[0,78]
        assert recover_cross_page_callout(p,n,marker,changed)['reason']=='native_and_selected_spans_disagree'
        assert recover_cross_page_callout(p,wrong,marker,doc['texts'])['reason']=='pages_not_adjacent_in_same_pdf'
        conflict=copy.deepcopy(doc['texts'][4236])
        conflict.update(self_ref='#/texts/test_conflict',text='Warning')
        conflict['prov']=[{**conflict['prov'][1],'charspan':[0,7]}]
        assert recover_cross_page_callout(p,n,marker,[*doc['texts'],conflict])['reason']=='competing_callout_marker'


def test_cross_page_rejects_partial_single_page_success_and_deduplicates_blocks():
    import json
    from pathlib import Path
    import pytest
    pdfplumber=pytest.importorskip('pdfplumber')
    from manuora.ingestion.geometry_recovery import recover_callout,recover_cross_page_callout
    root=Path(__file__).resolve().parents[2]
    manifest=root/'data/evaluation/automatic_ingestion_v0_1/results.json'
    if not manifest.is_file():
        pytest.skip('Optional local PDF corpus is not installed')
    records=json.loads(manifest.read_text(encoding='utf-8'))['documents']
    for ident,ref,page in [('qgis_desktopuserguide_en_3_44',11646,739),
                          ('nextcloud_administration_manual_36',4509,219)]:
        record=next(d for d in records if d['document_id']==ident)
        doc=json.loads((Path(record['output_dir'])/'parsed.json').read_text(encoding='utf-8'))
        with pdfplumber.open(record['source_pdf'],pages=[page,page+1]) as pdf:
            p,n=pdf.pages; marker=doc['texts'][ref]
            assert recover_callout(p,marker,doc['texts'])['status']=='unresolved'
            fixed=recover_cross_page_callout(p,n,marker,doc['texts'])
            assert fixed['status']=='recovered'
            assert len(fixed['body_block_ids'])==len(set(fixed['body_block_ids']))
            if ref==11646:
                assert 'also be achieved through' in fixed['body_text']
            else:
                assert fixed['body_block_ids']==['#/texts/4510']
                assert len(fixed['source_spans'])==2
                assert fixed['body_text'].count('The port is required')==1


def test_generic_configuration_header_preserves_spelling_and_values():
    doc = DoclingDocument(name="table")
    rows = [["[General] section"], ["Variable", "Default", "Meaning"],
            ["minChunk- Size", "5000000 (5 MB)", "Minimum upload size."],
            ["use-proxy", "false", "Enable proxy."]]
    cells = [TableCell(text=value, start_row_offset_idx=r, end_row_offset_idx=r+1,
                       start_col_offset_idx=c, end_col_offset_idx=(3 if r == 0 else c+1),
                       col_span=(3 if r == 0 else 1), column_header=r == 0)
             for r, row in enumerate(rows) for c, value in enumerate(row)]
    table = doc.add_table(data=TableData(num_rows=4, num_cols=3, table_cells=cells))
    text = configuration_text(table)
    assert "Variable: minChunk- Size; Default: 5000000 (5 MB); Meaning: Minimum upload size." in text
    assert "Variable: use-proxy" in text  # Do not strip genuine hyphens.
    assert "[General] section =" not in text
    found = detect_failures(doc)
    assert found["counts"]["wrapped_identifier_candidate"] == 1
    assert found["counts"]["table_banner_before_column_headers"] == 1


def test_unrecognized_table_is_not_rewritten():
    doc = DoclingDocument(name="table")
    cells = [TableCell(text=value, start_row_offset_idx=r, end_row_offset_idx=r+1,
                       start_col_offset_idx=c, end_col_offset_idx=c+1)
             for r, row in enumerate([["Name", "Value"], ["real-name", "42"]])
             for c, value in enumerate(row)]
    table = doc.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=cells))
    assert configuration_text(table) is None
    # No page evidence: the detector must not invent geometry or claim a repair.
    assert "possible_collapsed_table_rows" not in detect_failures(doc)["counts"]


def test_wrapped_prose_is_not_an_identifier_and_semantic_swaps_are_unmeasured():
    from manuora.ingestion.failure_detection import wrapped_identifier_candidate
    assert not wrapped_identifier_candidate("The inter- polation method")
    assert not wrapped_identifier_candidate("inter- polation")
    assert wrapped_identifier_candidate("maxChunk- Size")
    # A plausible cell-value swap needs independent semantic/position evidence.
    # It must not be advertised as detected by our current structural heuristics.
    doc = DoclingDocument(name="semantic blind spot")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    cells = [TableCell(text=value, start_row_offset_idx=r, end_row_offset_idx=r+1,
                       start_col_offset_idx=c, end_col_offset_idx=c+1)
             for r, row in enumerate([["Setting", "Default"], ["Retry count", "60"], ["Timeout", "3"]])
             for c, value in enumerate(row)]
    doc.add_table(data=TableData(num_rows=3, num_cols=2, table_cells=cells),
                  prov=ProvenanceItem(page_no=1, charspan=(0, 0),
                                      bbox=BoundingBox(l=40, t=700, r=500, b=400)))
    assert detect_failures(doc)["findings"] == []
