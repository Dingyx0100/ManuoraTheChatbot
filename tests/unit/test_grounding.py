from manuora.api.grounding import missing_terms, named_terms, parse_support, term_present


def test_named_terms_finds_names_identifiers_and_skips_products():
    assert named_terms("How do I add my Google Drive as external storage in Nextcloud 36?") == ["Google Drive"]
    assert named_terms("We moved from SQLite to MariaDB. How do I run occ files:scan?") == ["files:scan", "SQLite", "MariaDB"]
    assert named_terms("How do I 3D print a part in FreeCAD?") == ["3D"]
    assert named_terms("How can I change which widgets appear on my dashboard?") == []
    # the user's own host, mail and file names are not manual terms; a file's extension is
    assert named_terms("My server is cloud.example.com, mail admin@x.org. Can I open ports.shp?") == [".shp"]
    assert named_terms("Can I edit config.php or open a .rvt file?") == [".php", ".rvt"]


def test_term_present_accepts_whole_phrase_or_all_words_in_one_passage():
    assert term_present("Google Drive", ["Mount Google Drive here"])
    assert term_present("Layout Managers", ["the layout manager dialog"])
    assert not term_present("Google Drive", ["Google is a search engine", "Drive letters on Windows"])
    assert missing_terms("Where is the Blender render setting?", ["FreeCAD rendering with PovRay"]) == ["Blender"]


def test_headings_count_as_evidence_text():
    from manuora.api.grounding import passage_text
    hit = {"headings": ["9 Desktop Client", "9.4 Uninstallation"], "text": "Remove the app first."}
    assert missing_terms("Where did Desktop Client > Uninstallation go?", [passage_text(hit)]) == []


def test_parse_support_fails_closed():
    assert parse_support("SUPPORTED")
    assert parse_support("<think>\n</think>\nSupported.")
    assert not parse_support("UNSUPPORTED")
    assert not parse_support("I am not sure")


def test_spelling_mode_only_adds_matches():
    assert not term_present("PartDesign", ["use the Part Design workbench"])
    assert term_present("PartDesign", ["use the Part Design workbench"], spelling=True)
    assert term_present("SchematicEditors", ["open the Schematic Editor"], spelling=True)
    assert term_present("MiB", ["a fixed 5MiB chunk"], spelling=True)  # never stricter than the default
    assert not term_present("Blender", ["render with PovRay"], spelling=True)


def test_version_check_flags_features_missing_from_the_asked_version(tmp_path):
    import json
    import sqlite3
    from manuora.api.grounding import VersionIndex, version_reply

    db = sqlite3.connect(tmp_path / "idx.sqlite")
    db.execute("create table chunks (rowid integer primary key, chunk_id text, product_id text, product_version text,"
               " document_id text, source_pdf text, pages text, headings text, refs text, text text)")
    db.execute("create virtual table chunk_fts using fts5(text, headings, content='chunks', content_rowid='rowid')")
    rows = [("qgis", "2.18", ["Working with Raster"], "Load a raster layer."),
            ("qgis", "3.10", ["Working with Mesh Data", "Mesh layers"], "Load a mesh layer from a file.")]
    for i, (p, v, h, t) in enumerate(rows, 1):
        db.execute("insert into chunks values (?,?,?,?,?,?,?,?,?,?)", (i, f"c{i}", p, v, "d", "", "", json.dumps(h), "", t))
        db.execute("insert into chunk_fts(rowid, text, headings) values (?,?,?)", (i, t, json.dumps(h)))
    db.commit()
    index = VersionIndex(tmp_path / "idx.sqlite", source="headings")
    hits = index.other_version_features("How do I load a mesh layer in QGIS 2.18?", "qgis", "2.18")
    assert [h["feature"] for h in hits] == ["mesh layer"] and hits[0]["versions"] == ["3.10"]
    assert index.other_version_features("How do I load a mesh layer?", "qgis", "3.10") == []
    assert index.other_version_features("Did mesh layers change from 2.18 to 3.10?", "qgis", "2.18") == []  # comparison
    assert "documented from QGIS 3.10 on" in version_reply("qgis", "2.18", hits[0])
