import json
import sqlite3

from manuora.retrieval import outlines as ol


def _db(tmp_path):
    db = sqlite3.connect(tmp_path / "idx.sqlite")
    db.execute("create table chunks (rowid integer primary key, chunk_id text, product_id text, product_version text,"
               " document_id text, source_pdf text, pages text, headings text, refs text, text text)")
    rows = [("d:000002", ["3 Sharing", "3.1 Share a folder"], "Open the Sharing tab and enter the group name."),
            ("d:000001", ["3 Sharing", "3.1 Share a folder"], "Select the folder in the Files app."),
            ("d:000003", ["4 Groups"], "Create a group under Users > Groups.")]
    for i, (cid, heads, text) in enumerate(rows, 1):
        db.execute("insert into chunks values (?,?,?,?,?,?,?,?,?,?)",
                   (i, cid, "nextcloud", "36", "d", "", "[]", json.dumps(heads), "[]", text))
    db.commit()
    return tmp_path / "idx.sqlite"


def test_sections_group_consecutive_chunks_by_heading_in_document_order(tmp_path):
    secs = ol.sections(_db(tmp_path), ["d"])
    assert [[c["chunk_id"] for c in s["chunks"]] for s in secs] == [["d:000001", "d:000002"], ["d:000003"]]
    assert secs[0]["headings"] == ["3 Sharing", "3.1 Share a folder"]


def test_parse_outline_maps_passages_and_fails_closed(tmp_path):
    section = ol.sections(_db(tmp_path), ["d"])[0]
    reply = ('<think></think>{"goals": ["How do I share a folder with a group?"], "steps": ['
             '{"text": "select the folder", "passage": 1}, {"text": "enter the group name", "passage": 2},'
             ' {"text": "bad", "passage": 9}], "needs": ["create a group"]}')
    card = ol.parse_outline(reply, section)
    assert [s["chunk_id"] for s in card["steps"]] == ["d:000001", "d:000002"]
    assert card["needs"] == ["create a group"]
    assert ol.parse_outline('{"goals": []}', section) is None
    assert ol.parse_outline("not json", section) is None


def test_links_must_be_backed_by_the_passage():
    assert ol.backed_by("enter the group name", "Open the Sharing tab and enter the group name.")
    assert not ol.backed_by("enable server-side encryption", "Open the Sharing tab and enter the group name.")
    assert not ol.backed_by("click the button", "anything")  # no content words: never backed
    card = {"steps": [{"text": "enter the group name", "chunk_id": "a"}, {"text": "configure LDAP", "chunk_id": "a"}]}
    kept = ol.verify_steps(card, {"a": {"headings": ["Share"], "text": "Enter the group name."}})
    assert [s["text"] for s in kept["steps"]] == ["enter the group name"] and kept["dropped_steps"] == 1


def test_resolve_need_skips_own_section_and_low_scores():
    ranked = [{"chunk_id": "own", "rerank_score": 9.0, "headings": [], "text": "create a group"},
              {"chunk_id": "x", "rerank_score": 5.0, "headings": ["Groups"], "text": "To create a group, ..."},
              {"chunk_id": "y", "rerank_score": 1.0, "headings": [], "text": "create a group"}]
    assert ol.resolve_need("create a group", ranked, {"own"}, min_score=3.0) == "x"
    assert ol.resolve_need("create a group", ranked[2:], set(), min_score=3.0) is None


def test_card_chunks_and_merge_put_outline_after_the_best_passage():
    card = {"steps": [{"chunk_id": "s1"}, {"chunk_id": "s2"}], "linked_needs": [{"chunk_id": "n1"}, {"chunk_id": None}]}
    assert ol.card_chunks(card) == ["n1", "s1", "s2"]
    base = [{"chunk_id": "b1"}, {"chunk_id": "s2"}, {"chunk_id": "b3"}]
    merged = ol.merge_outline(base, [{"chunk_id": c} for c in ol.card_chunks(card)], max_passages=5)
    assert [h["chunk_id"] for h in merged] == ["b1", "n1", "s1", "s2", "b3"]


def test_outline_index_matches_within_product_and_version(tmp_path):
    cards = [dict(section_id="a", product_id="qgis", product_version="3.44", goals=["make a heatmap", "x"],
                  steps=[{"chunk_id": "c1"}]),
             dict(section_id="b", product_id="qgis", product_version="3.22", goals=["make a heatmap"],
                  steps=[{"chunk_id": "c2"}]),
             dict(section_id="c", product_id="qgis", product_version="3.44", goals=["no passages"], steps=[])]
    path = tmp_path / "outlines.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in cards), encoding="utf-8")

    class Encoder:
        def embed_queries(self, texts):
            return [[1.0, 0.0] if "heatmap" in t else [0.0, 1.0] for t in texts]

    index = ol.OutlineIndex(path, Encoder())
    assert [c["section_id"] for c in index.cards] == ["a", "b"]  # a card without passages is dropped
    found = index.match([1.0, 0.0], "qgis", ["3.44"], top_k=3, min_score=0.5)
    assert [(c["section_id"], round(s, 2)) for c, s in found] == [("a", 1.0)]
    assert index.match([1.0, 0.0], "qgis", ["3.44"], min_score=1.5) == []


def test_seed_links_follow_the_seed_sections_card_and_tasks():
    cards = {"A": {"steps": [{"chunk_id": "a1"}, {"chunk_id": "a2"}], "linked_needs": [{"chunk_id": "n1"}]}}
    tasks = {"A": [{"steps": [{"chunk_id": "a1"}, {"chunk_id": "b1"}]}]}
    section_of = {"a1": "A", "a2": "A"}
    assert ol.seed_links(["a1"], section_of, cards, tasks) == ["n1", "a2", "b1"]
    assert ol.seed_links(["a1"], section_of, cards, tasks, use=("task",)) == ["b1"]
    assert ol.seed_links(["zz"], section_of, cards, tasks) == []
