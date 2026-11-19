"""One end-to-end local index check with a deterministic fake encoder."""

import json
from importlib.util import find_spec

import pytest

if find_spec("sentence_transformers") is None:
    pytest.skip("embedding dependencies are optional", allow_module_level=True)
QdrantClient = pytest.importorskip("qdrant_client").QdrantClient

from manuora.retrieval.indexing import IndexConfig, SentenceEncoder, build_index, search_index
from manuora.retrieval.hybrid import reciprocal_rank_fusion
from manuora.retrieval.sparse import build_sparse_index, get_chunks, index_path, search_sparse


def test_shared_request_rejects_query_scope_and_span_drift():
    from manuora.retrieval.request import RetrievalRequest, trace_excerpt
    from manuora.retrieval.rerank import QwenReranker
    request=RetrievalRequest('followup','Where is it?','Where is the restore button in Nextcloud 36?', 'nextcloud',('36',))
    candidate=dict(chunk_id='doc:1',product_id='nextcloud',product_version='36',document_id='doc',headings=['Restore'],text='restore button')
    class Model:
        def predict(self,pairs,**kwargs):
            self.pairs=pairs
            return [1.0]*len(pairs)
    reranker=QwenReranker.__new__(QwenReranker); reranker.model=Model()
    # Existing callers retain their exact input format.
    reranker.rerank(request.query,[candidate])
    assert reranker.model.pairs==[(request.query,'Restore\nrestore button')]
    reranker.rerank(request.query,[candidate],request=request,metadata=True)
    assert 'Manual version: 36' in reranker.model.pairs[0][1]
    with pytest.raises(ValueError,match='query differs'):
        reranker.rerank('Where is it?',[candidate],request=request)
    with pytest.raises(ValueError,match='scope'):
        reranker.rerank(request.query,[dict(candidate,product_version='35')],request=request)
    trace=trace_excerpt(candidate,'The restore button is here',4,18)
    assert trace['source_char_start']==4
    with pytest.raises(ValueError,match='does not equal'):
        trace_excerpt(candidate,'The restore button is here',0,14)
    from manuora.retrieval.hybrid import HybridRetriever
    retriever=HybridRetriever.__new__(HybridRetriever)
    calls=[]
    def search(query,**filters):
        calls.append((query,filters))
        return [dict(candidate,product_version=filters['product_version'])]
    retriever.search=search
    comparison=RetrievalRequest('comparison','What changed?','Compare restore in 35 and 36','nextcloud',('35','36'))
    hits=retriever.search_request(comparison,candidates=12)
    assert [f['product_version'] for _,f in calls]==['35','36']
    assert all(q==comparison.query for q,_ in calls)
    reranker.rerank(comparison.query,hits,request=comparison)
    with pytest.raises(ValueError,match='provenance'):
        reranker.rerank(comparison.query,[dict(hits[0],retrieval_query='stale query')],request=comparison)


class FakeEncoder:
    dimension = 3
    max_tokens = 12
    revision = "fixed-revision"

    def count_document_tokens(self, text):
        return len(text.split())

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] if "install" in text.lower() else [0.0, 1.0, 0.0]
                for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_split_index_reopen_filter_and_model_isolation(tmp_path):
    source = tmp_path / "sources" / "QGIS" / "processed" / "guide"
    source.mkdir(parents=True)
    common = {
        "product_id": "qgis", "document_id": "guide", "source_pdf": "guide.pdf",
        "source_pdf_sha256": "abc", "headings": ["Install"], "pages": [7],
        "refs": ["#/texts/1"],
    }
    chunks = [
        {**common, "chunk_id": "guide:1", "product_version": "3.44",
         "text": "Install QGIS on Windows by opening the setup program and following the installation instructions carefully.",
         "contextual_text": "Install\nInstall QGIS on Windows by opening the setup program and following the installation instructions carefully."},
        {**common, "chunk_id": "guide:2", "product_version": "3.22",
         "text": "Configure the server", "contextual_text": "Install\nConfigure the server"},
    ]
    (source / "retrieval_chunks.jsonl").write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks), encoding="utf-8"
    )

    def config(model_id):
        return IndexConfig(
            model={"id": model_id, "revision": "fixed-revision", "normalize": True},
            store={"mode": "local", "path": "vector_store", "registry_path": "registry"},
            index={"collection_prefix": "test", "distance": "cosine", "upsert_batch_size": 2},
            root=tmp_path,
        )

    first = build_index(config("fake/model-a"), tmp_path / "sources", encoder=FakeEncoder())
    assert first["input_chunks"] == 2
    assert first["vector_points"] > 2
    assert first["split_input_chunks"] == 1
    assert first["estimated_vector_bytes"] == first["vector_points"] * 3 * 4
    assert first["total_wall_seconds"] >= first["embedding_upsert_wall_seconds"]
    assert build_index(config("fake/model-a"), tmp_path / "sources", encoder=FakeEncoder()) == first
    client = QdrantClient(path=str(tmp_path / "vector_store"))
    try:
        assert client.count(first["collection"], exact=True).count == first["vector_points"]
    finally:
        client.close()
    hits = search_index(config("fake/model-a"), first["collection"], "install",
                        product_version="3.44", encoder=FakeEncoder())
    assert hits and all(hit["product_version"] == "3.44" for hit in hits)
    assert all(hit["pages"] == [7] and hit["source_pdf"] == "guide.pdf" for hit in hits)
    assert all(hit["embedding_tokens"] <= FakeEncoder.max_tokens for hit in hits)
    assert "".join(hit["text"] for hit in sorted(hits, key=lambda h: h["part_number"])) == chunks[0]["text"]

    sparse = build_sparse_index(tmp_path, tmp_path / "sources")
    assert sparse["chunks"] == 2
    sparse_path = index_path(tmp_path, sparse["corpus_hash"])
    lexical = search_sparse(sparse_path, "install QGIS", product_version="3.44")
    assert lexical[0]["chunk_id"] == "guide:1"
    assert get_chunks(sparse_path, ["guide:1"])["guide:1"]["pages"] == [7]
    fused = reciprocal_rank_fusion(hits, lexical)
    assert len(fused) == 1 and fused[0]["retrievers"] == ["dense", "bm25"]

    second = build_index(config("fake/model-b"), tmp_path / "sources", encoder=FakeEncoder())
    assert first["collection"] != second["collection"]
    with pytest.raises(ValueError, match="vector space"):
        search_index(config("fake/model-b"), first["collection"], "install",
                     encoder=FakeEncoder())


def test_named_query_prompt_is_used_for_encoding_and_token_check():
    class Array:
        def tolist(self):
            return [[1.0, 0.0, 0.0]]

    class Model:
        max_seq_length = 20
        prompts = {"query": "Instruct: retrieve manuals\nQuery: "}

        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return Array()

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    encoder = SentenceEncoder.__new__(SentenceEncoder)
    encoder.config = {"query_prompt_name": "query", "document_prompt": ""}
    encoder.model = Model()
    encoder.tokenizer = Tokenizer()
    encoder.dimension = 3
    assert encoder.embed_query("install QGIS") == [1.0, 0.0, 0.0]
    assert encoder.model.calls[0][1]["prompt_name"] == "query"
    assert encoder.count_document_tokens("install QGIS") == 2
    encoder.embed_documents(["install QGIS"])
    assert encoder.model.calls[1][1]["prompt"] == ""


def test_porter_sparse_index_is_separate_and_stems(tmp_path):
    source = tmp_path / "sources" / "KiCad" / "processed" / "guide"
    source.mkdir(parents=True)
    chunk = {"chunk_id": "guide:1", "product_id": "kicad", "product_version": "9", "document_id": "guide",
             "source_pdf": "guide.pdf", "headings": ["Footprints"], "pages": [3], "refs": [],
             "text": "Assigning footprints to symbols."}
    (source / "retrieval_chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    plain = build_sparse_index(tmp_path, tmp_path / "sources")
    porter = build_sparse_index(tmp_path, tmp_path / "sources", tokenizer="porter unicode61")
    plain_path, porter_path = (tmp_path / m["index"] for m in (plain, porter))
    assert plain_path == index_path(tmp_path, plain["corpus_hash"]) and plain_path != porter_path
    assert porter["tokenizer"] == "porter unicode61" and porter["corpus_hash"] == plain["corpus_hash"]
    assert search_sparse(plain_path, "assign footprint") == []
    assert search_sparse(porter_path, "assign footprint")[0]["chunk_id"] == "guide:1"
    with pytest.raises(ValueError, match="tokenizer"):
        build_sparse_index(tmp_path, tmp_path / "sources", tokenizer="porter'); DROP TABLE chunks; --")


def test_hybrid_rejects_unknown_or_inconsistent_arms(tmp_path):
    from manuora.retrieval.hybrid import HybridRetriever
    with pytest.raises(ValueError, match="retrieval.lexical"):
        HybridRetriever(None, "dense", tmp_path / "bm25.sqlite", lexical="elastic")
    with pytest.raises(ValueError, match="needs retrieval.lexical=qdrant_sparse"):
        HybridRetriever(None, "dense", tmp_path / "bm25.sqlite", fusion="qdrant")


class FakeQdrant:
    def __init__(self, points):
        self.points, self.calls = points, []

    def query_points(self, collection, **kwargs):
        from types import SimpleNamespace
        self.calls.append(kwargs)
        return SimpleNamespace(points=[SimpleNamespace(score=s, payload=p) for s, p in self.points])


def _qdrant_lexical(points):
    from manuora.retrieval.qdrant_lexical import QdrantLexical
    lexical = QdrantLexical.__new__(QdrantLexical)
    lexical.collection, lexical.options, lexical.client = "c__bm25", {"avg_len": 100.0}, FakeQdrant(points)
    return lexical


def test_qdrant_bm25_hits_fuse_like_sqlite_hits():
    from qdrant_client import models
    part = {"chunk_id": "doc:1", "parent_chunk_id": "doc:1", "vector_chunk_id": "doc:1:part-1-of-2"}
    lexical = _qdrant_lexical([(4.0, part), (3.0, {**part, "vector_chunk_id": "doc:1:part-2-of-2"}),
                               (2.0, {"chunk_id": "doc:2"})])
    hits = lexical.search("footprint editor", limit=5, product_id="kicad", product_version="9")
    call = lexical.client.calls[0]
    assert call["using"] == "bm25" and call["limit"] == 5
    assert call["query"].model == "qdrant/bm25" and call["query"].options == {"avg_len": 100.0}
    assert {c.key for c in call["query_filter"].must} == {"product_id", "product_version"}
    fused = reciprocal_rank_fusion([{"chunk_id": "doc:2", "score": 0.9}], hits)
    assert [h["chunk_id"] for h in fused] == ["doc:2", "doc:1"] and fused[0]["retrievers"] == ["dense", "bm25"]
    assert lexical.search("?!", limit=5) == [] and len(lexical.client.calls) == 1  # no terms, no request

    fusion = lexical.hybrid("footprint editor", [0.1, 0.2, 0.3], limit=5, rrf_k=60)
    call = lexical.client.calls[-1]
    assert [p.using for p in call["prefetch"]] == ["dense", "bm25"] and call["limit"] == 10
    assert isinstance(call["query"], models.RrfQuery) and call["query"].rrf.k == 60
    assert [(h["chunk_id"], h["matched_part"]) for h in fusion] == [("doc:1", "doc:1:part-1-of-2"), ("doc:2", None)]
