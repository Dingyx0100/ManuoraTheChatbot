"""Persistent BM25 retrieval over the same source chunks used for dense indexing."""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path
from time import perf_counter

from manuora.retrieval.indexing import chunk_files, corpus_identity


DEFAULT_TOKENIZER = "unicode61"


def index_path(root: Path, corpus_hash: str, tokenizer: str = DEFAULT_TOKENIZER) -> Path:
    """bm25_<hash>.sqlite for the production tokenizer; bm25-<first word>_<hash> otherwise (e.g. bm25-porter)."""
    tag = "" if tokenizer == DEFAULT_TOKENIZER else "-" + re.sub(r"\W+", "", tokenizer.split()[0])
    return root / "data" / "sparse_indexes" / f"bm25{tag}_{corpus_hash[:12]}.sqlite"


def build_sparse_index(root: Path, source_root: Path, *, tokenizer: str = DEFAULT_TOKENIZER) -> dict:
    """`tokenizer` is an FTS5 tokenize spec, e.g. "porter unicode61" for English stemming."""
    if not re.fullmatch(r"[a-z0-9_ ]+", tokenizer):
        raise ValueError(f"unsupported FTS5 tokenizer spec: {tokenizer!r}")
    started = perf_counter()
    files = chunk_files(source_root)
    corpus_hash, chunks = corpus_identity(files, source_root, None)
    target = index_path(root, corpus_hash, tokenizer)
    manifest_path = target.with_suffix(".json")
    if target.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("corpus_hash") == corpus_hash and manifest.get("chunks") == len(chunks)
                and manifest.get("tokenizer", DEFAULT_TOKENIZER) == tokenizer):
            return manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".sqlite", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        connection = sqlite3.connect(temporary_path)
        try:
            connection.executescript(f"""
                CREATE TABLE chunks (
                    rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL,
                    product_id TEXT, product_version TEXT, document_id TEXT,
                    source_pdf TEXT, pages TEXT, headings TEXT, refs TEXT,
                    text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    text, headings, content='chunks', content_rowid='rowid',
                    tokenize='{tokenizer}'
                );
            """)
            with connection:
                for rowid, chunk in enumerate(chunks, 1):
                    fields = (
                        rowid, chunk["chunk_id"], chunk.get("product_id"),
                        chunk.get("product_version"), chunk.get("document_id"),
                        chunk.get("source_pdf"), json.dumps(chunk.get("pages", [])),
                        json.dumps(chunk.get("headings", [])), json.dumps(chunk.get("refs", [])),
                        chunk["text"],
                    )
                    connection.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)", fields)
                    connection.execute("INSERT INTO chunk_fts(rowid,text,headings) VALUES (?,?,?)",
                                       (rowid, chunk["text"], " ".join(chunk.get("headings", []))))
        finally:
            connection.close()
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
    manifest = {
        "corpus_hash": corpus_hash, "chunks": len(chunks), "source_files": len(files),
        "index": str(target.relative_to(root)), "tokenizer": tokenizer,
        "build_wall_seconds": round(perf_counter() - started, 2),
        "index_bytes": target.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def search_sparse(path: Path, query: str, *, limit: int = 20,
                  product_id: str | None = None, product_version: str | None = None,
                  document_id: str | None = None) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    if not terms:
        return []
    expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    conditions = ["chunk_fts MATCH ?"]
    parameters: list[object] = [expression]
    for field, value in (("product_id", product_id), ("product_version", product_version),
                         ("document_id", document_id)):
        if value is not None:
            conditions.append(f"c.{field} = ?")
            parameters.append(value)
    parameters.append(limit)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        records = connection.execute(f"""
            SELECT c.chunk_id, c.product_id, c.product_version, c.document_id,
                   c.source_pdf, c.pages, c.headings, c.refs, c.text,
                   -bm25(chunk_fts, 1.0, 2.0) AS score
            FROM chunk_fts JOIN chunks c ON c.rowid = chunk_fts.rowid
            WHERE {' AND '.join(conditions)}
            ORDER BY bm25(chunk_fts, 1.0, 2.0) LIMIT ?
        """, parameters).fetchall()
    finally:
        connection.close()
    return [{**dict(row), "pages": json.loads(row["pages"]),
             "headings": json.loads(row["headings"]), "refs": json.loads(row["refs"])}
            for row in records]


def get_chunks(path: Path, chunk_ids: list[str]) -> dict[str, dict]:
    """Fetch complete source chunks for dense hits before reranking."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
        ).fetchall()
    finally:
        connection.close()
    return {row["chunk_id"]: {**dict(row), "pages": json.loads(row["pages"]),
                              "headings": json.loads(row["headings"]),
                              "refs": json.loads(row["refs"])} for row in rows}
