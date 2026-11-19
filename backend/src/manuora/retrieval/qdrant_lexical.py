"""Experiment S: BM25 inside Qdrant (sparse vectors + IDF modifier, server-side `qdrant/bm25` inference).

The lexical collection is a copy of a completed dense collection: same point ids and payloads, the dense
vector copied as the named vector "dense" (nothing is re-embedded) and a server-computed BM25 sparse vector
"bm25" over `contextual_text` (headings + body, the closest match to what the SQLite index searches).
The production dense collection is only read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient, models

from manuora.retrieval.indexing import IndexConfig, _atomic_json, open_store, scope_filter

BM25_MODEL = "qdrant/bm25"
DENSE, SPARSE = "dense", "bm25"


def lexical_collection_name(dense_collection: str) -> str:
    return f"{dense_collection}__bm25"


def _manifest_path(config: IndexConfig, collection: str) -> Path:
    return config.location("registry_path") / f"{collection}.json"


def load_manifest(config: IndexConfig, collection: str) -> dict:
    path = _manifest_path(config, collection)
    if not path.is_file():
        raise ValueError(f"no lexical collection manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("state") != "complete" or manifest.get("kind") != "qdrant_bm25":
        raise ValueError(f"lexical collection is not a completed qdrant_bm25 build: {collection}")
    return manifest


def build_lexical_collection(config: IndexConfig, dense_collection: str, *, batch_size: int = 256,
                             target: str | None = None) -> dict:
    if config.store["mode"] != "remote":
        raise ValueError("server-side BM25 inference needs a Qdrant server (store.mode: remote)")
    started = perf_counter()
    dense_manifest = json.loads(_manifest_path(config, dense_collection).read_text(encoding="utf-8"))
    if dense_manifest.get("state") != "complete":
        raise ValueError(f"dense collection is not complete: {dense_collection}")
    name = target or lexical_collection_name(dense_collection)
    client = open_store(config)
    try:
        expected_points = client.count(dense_collection, exact=True).count
        if expected_points != dense_manifest["vector_points"]:
            raise ValueError(f"{dense_collection} has {expected_points} points; manifest says "
                             f"{dense_manifest['vector_points']}")
        if client.collection_exists(name):
            path = _manifest_path(config, name)
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if (existing.get("state") == "complete" and existing.get("corpus_hash") == dense_manifest["corpus_hash"]
                        and client.count(name, exact=True).count == expected_points):
                    return existing
            client.delete_collection(name)  # partial or stale experiment copy; the source is untouched

        # BM25 length normalisation: average word count of what is indexed (Qdrant's default is 256).
        avg_len = _average_words(client, dense_collection)
        options = {"avg_len": round(avg_len, 1)}
        source = client.get_collection(dense_collection).config.params.vectors
        client.create_collection(
            name,
            vectors_config={DENSE: models.VectorParams(size=source.size, distance=source.distance)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        manifest = {
            "collection": name, "kind": "qdrant_bm25", "source_collection": dense_collection,
            "corpus_hash": dense_manifest["corpus_hash"], "space_hash": dense_manifest["space_hash"],
            "bm25_model": BM25_MODEL, "bm25_options": options, "indexed_field": "contextual_text",
            "vector_points": expected_points, "state": "building",
        }
        _atomic_json(_manifest_path(config, name), manifest)
        copied, offset = 0, None
        while True:
            points, offset = client.scroll(dense_collection, limit=batch_size, offset=offset,
                                           with_payload=True, with_vectors=True)
            if not points:
                break
            client.upsert(name, wait=True, points=[models.PointStruct(
                id=point.id, payload=point.payload,
                vector={DENSE: point.vector,
                        SPARSE: models.Document(text=point.payload["contextual_text"], model=BM25_MODEL,
                                                options=options)},
            ) for point in points])
            copied += len(points)
            if copied % (batch_size * 20) == 0:
                print(f"copied {copied}/{expected_points} in {perf_counter() - started:.1f}s", flush=True)
            if offset is None:
                break
        actual = client.count(name, exact=True).count
        if actual != expected_points:
            raise ValueError(f"{name} has {actual} points; expected {expected_points}")
        for field in ("product_id", "product_version", "document_id"):
            client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD, wait=True)
        manifest = {**manifest, "state": "complete", "build_wall_seconds": round(perf_counter() - started, 2)}
        _atomic_json(_manifest_path(config, name), manifest)
        return manifest
    finally:
        client.close()


def _average_words(client: QdrantClient, collection: str) -> float:
    total = count = 0
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=1024, offset=offset,
                                       with_payload=["contextual_text"], with_vectors=False)
        for point in points:
            total += len(re.findall(r"\w+", point.payload.get("contextual_text") or ""))
            count += 1
        if offset is None or not points:
            return total / max(count, 1)


class QdrantLexical:
    """BM25 search, and dense+BM25 fusion done by Qdrant, over one lexical collection."""

    def __init__(self, config: IndexConfig, collection: str):
        self.manifest = load_manifest(config, collection)
        self.collection = collection
        self.options = self.manifest["bm25_options"]
        self.client = open_store(config)  # kept open: one HTTP client for the process

    def _document(self, query: str) -> models.Document:
        return models.Document(text=query, model=BM25_MODEL, options=self.options)

    def search(self, query: str, *, limit: int = 30, **scope) -> list[dict]:
        """Lexical hits in the shape `search_sparse` returns (payload + score; parts carry parent_chunk_id)."""
        if not re.search(r"\w", query):
            return []
        result = self.client.query_points(self.collection, query=self._document(query), using=SPARSE,
                                          query_filter=scope_filter(**scope), limit=limit, with_payload=True)
        return [{"score": point.score, **(point.payload or {})} for point in result.points]

    def hybrid(self, query: str, query_vector: list[float], *, limit: int = 30, rrf_k: int = 60,
               **scope) -> list[dict]:
        """One request: prefetch `limit` dense and `limit` BM25 points, fuse with Qdrant RRF.

        Qdrant fuses points (split parts count separately), so parents are de-duplicated after fusion,
        keeping each parent's best-ranked part. Output matches `reciprocal_rank_fusion` minus the
        per-retriever scores, which the fused response does not carry.
        """
        query_filter = scope_filter(**scope)
        result = self.client.query_points(
            self.collection,
            prefetch=[models.Prefetch(query=query_vector, using=DENSE, filter=query_filter, limit=limit),
                      models.Prefetch(query=self._document(query), using=SPARSE, filter=query_filter, limit=limit)],
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)), limit=2 * limit, with_payload=True,
        )
        fused, seen = [], set()
        for point in result.points:
            payload = point.payload or {}
            parent = payload.get("parent_chunk_id") or payload["chunk_id"]
            if parent in seen:
                continue
            seen.add(parent)
            fused.append({"chunk_id": parent, "fusion_score": point.score, "retrievers": ["qdrant_rrf"],
                          "matched_part": payload.get("vector_chunk_id")})
        return fused

    def close(self) -> None:
        self.client.close()
