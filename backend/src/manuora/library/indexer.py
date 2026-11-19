"""Add or remove one manual in the live indexes, without a full rebuild.

Three stores hold every manual, keyed by document_id:
- the SQLite chunk store (+ FTS5 BM25), which hydrates every search hit;
- the dense Qdrant collection (python-fusion arms);
- the `__bm25` Qdrant collection (dense + BM25 sparse vectors; production fusion).

After every change the three manifests get the same new `corpus_hash`, computed from the chunk files
on disk exactly as a full build would (`corpus_identity`), so the retriever's consistency checks keep
meaning "these indexes hold this corpus". The collection and index *names* keep the hash they were built
with; only a full rebuild renames them.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Callable

from qdrant_client import models

from manuora.retrieval.indexing import (
    IndexConfig, _atomic_json, _payload, chunk_files, corpus_identity, open_store, space_identity, split_chunk,
)
from manuora.retrieval.qdrant_lexical import BM25_MODEL, DENSE, SPARSE, lexical_collection_name

Progress = Callable[[int, int], None]


class LibraryIndex:
    def __init__(self, config: IndexConfig, dense_collection: str, sparse_path: Path, source_root: Path,
                 lexical_collection: str | None = None, encoder=None):
        self.config = config
        self.dense = dense_collection
        self.lexical = lexical_collection or lexical_collection_name(dense_collection)
        self.sparse_path = sparse_path
        self.source_root = source_root
        self._encoder = encoder
        registry = config.location("registry_path")
        self.dense_manifest_path = registry / f"{self.dense}.json"
        self.lexical_manifest_path = registry / f"{self.lexical}.json"
        self.sparse_manifest_path = sparse_path.with_suffix(".json")
        self.has_lexical = self.lexical_manifest_path.is_file()

    @property
    def encoder(self):
        if self._encoder is None:
            from manuora.retrieval.indexing import SentenceEncoder
            self._encoder = SentenceEncoder(self.config.model)
        return self._encoder

    # ------------------------------------------------------------------ add
    def add_document(self, chunks: list[dict], *, on_embed: Progress | None = None,
                     on_store: Progress | None = None) -> dict:
        """Index one manual's chunks (replacing any earlier copy of the same document_id)."""
        if not chunks:
            raise ValueError("the manual produced no retrieval chunks")
        document_ids = {chunk["document_id"] for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError(f"expected chunks of one manual, got {sorted(document_ids)}")
        document_id = document_ids.pop()
        dense_manifest = json.loads(self.dense_manifest_path.read_text(encoding="utf-8"))
        space_hash, _ = space_identity(self.config, self.encoder)
        if space_hash != dense_manifest["space_hash"]:
            raise ValueError("the configured embedding model does not match the dense collection")
        bm25_options = None
        if self.has_lexical:
            bm25_options = json.loads(self.lexical_manifest_path.read_text(encoding="utf-8"))["bm25_options"]

        parts = [part for chunk in chunks for part in split_chunk(chunk, self.encoder)]
        vectors: list[list[float]] = []
        batch = int(self.config.model.get("batch_size", 8))
        for offset in range(0, len(parts), batch):
            vectors += self.encoder.embed_documents([p["contextual_text"] for p in parts[offset:offset + batch]])
            if on_embed:
                on_embed(min(offset + batch, len(parts)), len(parts))

        self.remove_document(document_id, refresh=False)
        client = open_store(self.config)
        try:
            corpus_hash = dense_manifest["corpus_hash"]  # replaced by refresh_identity below
            upsert = int(self.config.index.get("upsert_batch_size", 32))
            for offset in range(0, len(parts), upsert):
                window = list(zip(parts[offset:offset + upsert], vectors[offset:offset + upsert]))
                dense_points, lexical_points = [], []
                for part, vector in window:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.dense}/{part['vector_chunk_id']}"))
                    payload = _payload(part, space_hash, corpus_hash)
                    dense_points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
                    if self.has_lexical:
                        lexical_points.append(models.PointStruct(id=point_id, payload=payload, vector={
                            DENSE: vector,
                            SPARSE: models.Document(text=part["contextual_text"], model=BM25_MODEL, options=bm25_options)}))
                client.upsert(self.dense, dense_points, wait=True)
                if lexical_points:
                    client.upsert(self.lexical, lexical_points, wait=True)
                if on_store:
                    on_store(min(offset + upsert, len(parts)), len(parts) + 1)
        finally:
            client.close()
        self._sqlite_insert(chunks)
        identity = self.refresh_identity()
        if on_store:
            on_store(len(parts) + 1, len(parts) + 1)
        return {"document_id": document_id, "chunks": len(chunks), "vector_points": len(parts), **identity}

    def _sqlite_insert(self, chunks: list[dict]) -> None:
        connection = sqlite3.connect(self.sparse_path, timeout=30)
        try:
            with connection:
                for chunk in chunks:
                    cursor = connection.execute(
                        "INSERT INTO chunks(chunk_id, product_id, product_version, document_id, source_pdf, pages,"
                        " headings, refs, text) VALUES (?,?,?,?,?,?,?,?,?)",
                        (chunk["chunk_id"], chunk.get("product_id"), chunk.get("product_version"),
                         chunk.get("document_id"), chunk.get("source_pdf"), json.dumps(chunk.get("pages", [])),
                         json.dumps(chunk.get("headings", [])), json.dumps(chunk.get("refs", [])), chunk["text"]))
                    connection.execute("INSERT INTO chunk_fts(rowid, text, headings) VALUES (?,?,?)",
                                       (cursor.lastrowid, chunk["text"], " ".join(chunk.get("headings", []))))
        finally:
            connection.close()

    # ------------------------------------------------------------------ remove
    def remove_document(self, document_id: str, *, refresh: bool = True) -> int:
        """Delete a manual from all three stores; returns the number of source chunks removed."""
        connection = sqlite3.connect(self.sparse_path, timeout=30)
        try:
            with connection:
                rows = connection.execute("SELECT rowid, text, headings FROM chunks WHERE document_id = ?",
                                          (document_id,)).fetchall()
                for rowid, text, headings in rows:  # external-content FTS5: delete with the indexed values
                    connection.execute("INSERT INTO chunk_fts(chunk_fts, rowid, text, headings) VALUES ('delete',?,?,?)",
                                       (rowid, text, " ".join(json.loads(headings or "[]"))))
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        finally:
            connection.close()
        selector = models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]))
        client = open_store(self.config)
        try:
            for collection in (self.dense, self.lexical) if self.has_lexical else (self.dense,):
                client.delete(collection, points_selector=selector, wait=True)
        finally:
            client.close()
        if refresh:
            self.refresh_identity()
        return len(rows)

    # ------------------------------------------------------------------ identity
    def indexed_counts(self) -> dict[str, int]:
        connection = sqlite3.connect(f"file:{self.sparse_path.as_posix()}?mode=ro", uri=True, timeout=30)
        try:
            return dict(connection.execute("SELECT document_id, COUNT(*) FROM chunks GROUP BY document_id"))
        finally:
            connection.close()

    def refresh_identity(self) -> dict:
        """Stamp all three manifests with the corpus now on disk, after checking the indexes hold exactly it."""
        files = chunk_files(self.source_root)
        corpus_hash, chunks = corpus_identity(files, self.source_root, None)
        on_disk: dict[str, int] = {}
        for chunk in chunks:
            on_disk[chunk["document_id"]] = on_disk.get(chunk["document_id"], 0) + 1
        indexed = self.indexed_counts()
        if on_disk != indexed:
            drift = sorted(k for k in set(on_disk) | set(indexed) if on_disk.get(k) != indexed.get(k))
            raise ValueError(f"index and chunk files disagree for {drift[:5]}; run a full rebuild")
        client = open_store(self.config)
        try:
            points = client.count(self.dense, exact=True).count
            lexical_points = client.count(self.lexical, exact=True).count if self.has_lexical else None
        finally:
            client.close()
        if lexical_points is not None and lexical_points != points:
            raise ValueError(f"dense ({points}) and BM25 ({lexical_points}) collections disagree")
        sources = [str(path.relative_to(self.source_root)) for path in files]
        dense = json.loads(self.dense_manifest_path.read_text(encoding="utf-8"))
        _atomic_json(self.dense_manifest_path, {**dense, "corpus_hash": corpus_hash, "source_files": sources,
                                                "input_chunks": len(chunks), "vector_points": points,
                                                "updated_incrementally": True})
        sparse = json.loads(self.sparse_manifest_path.read_text(encoding="utf-8"))
        _atomic_json(self.sparse_manifest_path, {**sparse, "corpus_hash": corpus_hash, "chunks": len(chunks),
                                                 "source_files": len(files), "updated_incrementally": True})
        if self.has_lexical:
            lexical = json.loads(self.lexical_manifest_path.read_text(encoding="utf-8"))
            _atomic_json(self.lexical_manifest_path, {**lexical, "corpus_hash": corpus_hash,
                                                      "vector_points": points, "updated_incrementally": True})
        return {"corpus_hash": corpus_hash, "corpus_chunks": len(chunks), "corpus_points": points}
