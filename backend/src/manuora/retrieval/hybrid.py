"""Fuse Qdrant dense hits and SQLite BM25 hits by source chunk."""

from __future__ import annotations

import json
from pathlib import Path

from manuora.retrieval.indexing import IndexConfig, SentenceEncoder, search_index, space_identity
from manuora.retrieval.sparse import get_chunks, search_sparse
from manuora.retrieval.request import RetrievalRequest


def reciprocal_rank_fusion(dense: list[dict], sparse: list[dict],
                           *, rank_constant: int = 60) -> list[dict]:
    """Give each parent chunk one rank per retriever, even if several parts match."""
    merged: dict[str, dict] = {}
    for name, results in (("dense", dense), ("bm25", sparse)):
        seen: set[str] = set()
        for result in results:
            parent = result.get("parent_chunk_id") or result["chunk_id"]
            if parent in seen:
                continue
            seen.add(parent)
            if parent not in merged:
                merged[parent] = {"chunk_id": parent, "fusion_score": 0.0,
                                  "retrievers": [], "matched_part": None}
            hit = merged[parent]
            hit["fusion_score"] += 1 / (rank_constant + len(seen))
            hit["retrievers"].append(name)
            if name == "dense":
                hit["matched_part"] = result.get("vector_chunk_id")
                hit["dense_score"] = result["score"]
            else:
                hit["bm25_score"] = result["score"]
    return sorted(merged.values(), key=lambda hit: hit["fusion_score"], reverse=True)


class HybridRetriever:
    LEXICAL = ("sqlite", "qdrant_sparse")
    FUSION = ("python", "qdrant")

    def __init__(self, config: IndexConfig, collection: str, sparse_path: Path, *,
                 lexical: str = "sqlite", fusion: str = "python", rrf_k: int = 60,
                 lexical_collection: str | None = None):
        """`lexical`/`fusion` pick an experiment-S arm; the defaults are production (SQLite BM25, RRF here).

        SQLite stays the chunk store in every arm: hits are always hydrated from it.
        """
        if lexical not in self.LEXICAL or fusion not in self.FUSION:
            raise ValueError(f"retrieval.lexical must be one of {self.LEXICAL}, retrieval.fusion one of {self.FUSION}")
        if fusion == "qdrant" and lexical != "qdrant_sparse":
            raise ValueError("retrieval.fusion=qdrant needs retrieval.lexical=qdrant_sparse")
        dense_manifest = json.loads((config.location("registry_path") / f"{collection}.json").read_text(encoding="utf-8"))
        sparse_manifest = json.loads(sparse_path.with_suffix(".json").read_text(encoding="utf-8"))
        if (dense_manifest.get("state") != "complete" or
                dense_manifest["corpus_hash"] != sparse_manifest["corpus_hash"]):
            raise ValueError("dense and BM25 indexes must cover the same completed corpus")
        self.config = config
        self.collection = collection
        self.sparse_path = sparse_path
        self.encoder = SentenceEncoder(config.model)
        self.lexical, self.fusion, self.rrf_k = lexical, fusion, rrf_k
        self.qdrant_lexical = None
        if lexical == "qdrant_sparse":
            from manuora.retrieval.qdrant_lexical import QdrantLexical, lexical_collection_name
            self.qdrant_lexical = QdrantLexical(config, lexical_collection or lexical_collection_name(collection))
            manifest = self.qdrant_lexical.manifest
            if manifest["corpus_hash"] != dense_manifest["corpus_hash"]:
                raise ValueError("Qdrant BM25 collection covers a different corpus than the dense index")
            if fusion == "qdrant":  # the fused query skips search_index, so check the vector space once here
                space_hash, _ = space_identity(config, self.encoder)
                if space_hash != manifest["space_hash"]:
                    raise ValueError("configured embedding model does not match the BM25 collection's dense vectors")

    def search_request(self, request: RetrievalRequest, *, candidates: int = 30) -> list[dict]:
        """Opt-in shared request; search each explicit version independently.

        Preserve all returned candidates and their match IDs. The downstream
        selector owns cross-version evidence allocation, not this adapter.
        """
        result=[]
        for version in dict.fromkeys(request.versions):
            hits=self.search(request.query,product_id=request.product_id,
                             product_version=version,document_id=request.document_id,
                             candidates=candidates)
            for hit in hits:
                request.check(request.query,hit)
                result.append({**hit,'retrieval_request_id':request.request_id,
                               'retrieval_query':request.query})
        return result

    def search(self, query: str, *, product_id: str | None = None,
               product_version: str | None = None, document_id: str | None = None,
               candidates: int = 30) -> list[dict]:
        filters = {"product_id": product_id, "product_version": product_version,
                   "document_id": document_id}
        if self.fusion == "qdrant":
            fused = self.qdrant_lexical.hybrid(query, self.encoder.embed_query(query), limit=candidates,
                                               rrf_k=self.rrf_k, **filters)
        else:
            dense = search_index(self.config, self.collection, query, limit=candidates,
                                 encoder=self.encoder, **filters)
            if self.lexical == "qdrant_sparse":
                sparse = self.qdrant_lexical.search(query, limit=candidates, **filters)
            else:
                sparse = search_sparse(self.sparse_path, query, limit=candidates, **filters)
            fused = reciprocal_rank_fusion(dense, sparse, rank_constant=self.rrf_k)
        sources = get_chunks(self.sparse_path, [hit["chunk_id"] for hit in fused])
        missing = [hit["chunk_id"] for hit in fused if hit["chunk_id"] not in sources]
        if missing:
            raise ValueError(f"dense and BM25 indexes disagree on source chunks: {missing[:3]}")
        return [{**sources[hit["chunk_id"]], **hit} for hit in fused]
