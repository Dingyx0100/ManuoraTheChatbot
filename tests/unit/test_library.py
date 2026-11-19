"""Library: incremental add/remove keeps the three indexes on one corpus identity; upload placement."""

import json
import shutil
from importlib.util import find_spec
from pathlib import Path

import pytest
import yaml

if find_spec("sentence_transformers") is None:
    pytest.skip("embedding dependencies are optional", allow_module_level=True)
pytest.importorskip("qdrant_client")

from manuora.library.indexer import LibraryIndex  # noqa: E402
from manuora.library.jobs import JobStore, Library, safe_filename  # noqa: E402
from manuora.retrieval.indexing import IndexConfig, build_index, chunk_files, corpus_identity  # noqa: E402
from manuora.retrieval.sparse import build_sparse_index, search_sparse  # noqa: E402


class FakeEncoder:
    dimension = 3
    max_tokens = 64
    revision = "fixed-revision"

    def count_document_tokens(self, text):
        return len(text.split())

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] if "footprint" in t.lower() else [0.0, 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def chunk(doc, n, version, text):
    return {"chunk_id": f"{doc}:{n:06d}", "product_id": "kicad", "product_version": version, "document_id": doc,
            "source_pdf": "guide.pdf", "source_pdf_sha256": "abc", "headings": ["Footprints"], "pages": [n],
            "refs": [], "text": text, "contextual_text": f"Footprints\n{text}"}


def write_chunks(root, version, doc, chunks):
    folder = root / "KiCad" / version / "processed" / doc
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "retrieval_chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    return folder


def test_add_then_remove_returns_to_the_same_corpus(tmp_path):
    sources = tmp_path / "data" / "sources"
    write_chunks(sources, "8", "kicad_guide_8", [chunk("kicad_guide_8", 1, "8", "Place symbols on the schematic."),
                                                 chunk("kicad_guide_8", 2, "8", "Assign a footprint to each symbol.")])
    config = IndexConfig(model={"id": "fake/model", "revision": "fixed-revision", "batch_size": 2},
                         store={"mode": "local", "path": "vector_store", "registry_path": "registry"},
                         index={"collection_prefix": "test", "distance": "cosine", "upsert_batch_size": 2}, root=tmp_path)
    dense = build_index(config, sources, encoder=FakeEncoder())
    sparse = build_sparse_index(tmp_path, sources)
    sparse_path = tmp_path / sparse["index"]
    original = sparse["corpus_hash"]
    index = LibraryIndex(config, dense["collection"], sparse_path, sources, encoder=FakeEncoder())
    assert not index.has_lexical  # no __bm25 collection in this test: only the dense collection + SQLite

    new = [chunk("kicad_guide_9", 1, "9", "Open the footprint editor to draw a new footprint."),
           chunk("kicad_guide_9", 2, "9", "Route tracks with the interactive router.")]
    write_chunks(sources, "9", "kicad_guide_9", new)
    embedded = []
    result = index.add_document(new, on_embed=lambda done, total: embedded.append((done, total)))
    assert embedded[-1] == (2, 2) and result["chunks"] == 2
    expected, _ = corpus_identity(chunk_files(sources), sources, None)
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in (index.dense_manifest_path, index.sparse_manifest_path)]
    assert all(m["corpus_hash"] == expected != original for m in manifests)
    assert manifests[0]["vector_points"] == 4 and manifests[1]["chunks"] == 4
    assert search_sparse(sparse_path, "interactive router", product_version="9")[0]["chunk_id"] == "kicad_guide_9:000002"

    # Re-adding the same manual replaces it instead of duplicating it.
    index.add_document(new)
    assert index.indexed_counts() == {"kicad_guide_8": 2, "kicad_guide_9": 2}

    shutil.rmtree(sources / "KiCad" / "9")
    assert index.remove_document("kicad_guide_9") == 2
    assert json.loads(index.sparse_manifest_path.read_text(encoding="utf-8"))["corpus_hash"] == original
    assert json.loads(index.dense_manifest_path.read_text(encoding="utf-8"))["vector_points"] == 2
    assert search_sparse(sparse_path, "interactive router") == []  # FTS rows are gone too


def test_refresh_refuses_when_disk_and_index_disagree(tmp_path):
    sources = tmp_path / "sources"
    write_chunks(sources, "8", "kicad_guide_8", [chunk("kicad_guide_8", 1, "8", "Place symbols.")])
    config = IndexConfig(model={"id": "fake/model", "revision": "fixed-revision"},
                         store={"mode": "local", "path": "vector_store", "registry_path": "registry"},
                         index={"collection_prefix": "test"}, root=tmp_path)
    dense = build_index(config, sources, encoder=FakeEncoder())
    sparse = build_sparse_index(tmp_path, sources)
    index = LibraryIndex(config, dense["collection"], tmp_path / sparse["index"], sources, encoder=FakeEncoder())
    write_chunks(sources, "9", "kicad_guide_9", [chunk("kicad_guide_9", 1, "9", "Not indexed yet.")])
    with pytest.raises(ValueError, match="disagree"):
        index.refresh_identity()


def test_upload_placement_suggestion_and_jobs(tmp_path):
    sources = tmp_path / "sources"
    (sources / "KiCad" / "9").mkdir(parents=True)
    (sources / "KiCad" / "9" / "manifest.yaml").write_text(yaml.safe_dump({
        "product_id": "kicad", "display_name": "KiCad", "version": "9", "language": "en",
        "documents": [{"id": "kicad_getting_started_in_kicad_9", "type": "getting_started_guide",
                       "title": "Getting Started in KiCad 9", "path": "getting_started_in_kicad.pdf"}]}), encoding="utf-8")
    library = Library(tmp_path, sources, tmp_path / "library", index_factory=lambda: None)

    suggestion = library.suggest("getting_started_in_kicad_10.pdf")
    assert suggestion == {"product_id": "kicad", "display_name": "KiCad", "version": "10",
                          "title": "Getting Started in KiCad 10", "type": "getting_started_guide"}
    assert library.suggest("RouterX_AdminGuide_v2.1.pdf")["display_name"] == "RouterX"
    assert safe_filename("..\\..\\evil name?.PDF") == "evil name.pdf"
    tricky = safe_filename("../a/b:c*d.pdf")  # "b:" is a drive on Windows: only the safety matters
    assert tricky.endswith("c_d.pdf") and not set(tricky) & set("/\\:*?\"<>|")

    target = library.new_upload("getting_started_in_kicad_10.pdf")
    target.write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="not a PDF"):
        library.finish_upload(target)
    target = library.new_upload("getting_started_in_kicad_10.pdf")
    target.write_bytes(b"%PDF-1.7 test")
    upload = library.finish_upload(target)

    job = library.submit(upload["upload_id"], display_name="kicad", version="10",
                         title="Getting Started in KiCad 10", type="getting_started_guide")
    assert job["product_id"] == "kicad" and job["display_name"] == "KiCad" and job["replaces"] is None
    library._stage_upload(library.store.get(job["id"]))
    placed = library.store.get(job["id"])
    manifest = yaml.safe_load(Path(placed["manifest"]).read_text(encoding="utf-8"))
    assert Path(placed["manifest"]).parent == sources / "KiCad" / "10"  # new version next to the existing one
    assert manifest["documents"][0]["id"] == placed["document_id"] == "kicad_getting_started_in_kicad_10"
    assert (sources / "KiCad" / "10" / "getting_started_in_kicad_10.pdf").read_bytes().startswith(b"%PDF-")

    # Same product, version and title again: a replacement of that manual, not a second copy.
    target = library.new_upload("kicad10_v2.pdf")
    target.write_bytes(b"%PDF-1.7 newer")
    again = library.submit(library.finish_upload(target)["upload_id"], display_name="KiCad", version="10",
                           title="getting started in kicad 10", type="getting_started_guide")
    assert again["replaces"] == "kicad_getting_started_in_kicad_10"

    store = JobStore(tmp_path / "library" / "jobs.sqlite")
    store.update(job["id"], status="running")
    store.interrupt_running()
    assert store.get(job["id"])["status"] == "failed"
    assert library.retry(job["id"])["status"] == "queued"
