"""Model-isolated, reproducible dense indexing in Qdrant."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import yaml
from qdrant_client import QdrantClient, models


SPLITTER_VERSION = "text-boundaries-v1"
REQUIRED_CHUNK_FIELDS = {
    "chunk_id", "contextual_text", "text", "headings", "pages", "product_id",
    "product_version", "document_id", "source_pdf", "source_pdf_sha256",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class IndexConfig:
    model: dict
    store: dict
    index: dict
    root: Path

    @classmethod
    def load(cls, path: Path, root: Path) -> IndexConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(k in data for k in ("model", "store", "index")):
            raise ValueError("embedding config needs model, store, and index mappings")
        model, store, index = (data[k] for k in ("model", "store", "index"))
        if not model.get("id") or not model.get("revision"):
            raise ValueError("model.id and an immutable model.revision are required")
        for side in ("document", "query"):
            if model.get(f"{side}_prompt") and model.get(f"{side}_prompt_name"):
                raise ValueError(f"model.{side}_prompt and model.{side}_prompt_name are mutually exclusive")
        if store.get("mode") not in {"local", "remote"}:
            raise ValueError("store.mode must be local or remote")
        if store["mode"] == "remote" and not store.get("url"):
            raise ValueError("remote Qdrant requires store.url")
        if index.get("distance", "cosine") not in {"cosine", "dot"}:
            raise ValueError("index.distance must be cosine or dot")
        for field, value in (("model.batch_size", model.get("batch_size", 16)),
                             ("index.upsert_batch_size", index.get("upsert_batch_size", 32))):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        return cls(model=model, store=store, index=index, root=root.resolve())

    def location(self, name: str) -> Path:
        return (self.root / self.store[name]).resolve()


class Encoder(Protocol):
    dimension: int
    max_tokens: int
    revision: str

    def count_document_tokens(self, text: str) -> int: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceEncoder:
    """Keep query/document prompts and tokenizer with the selected checkpoint."""

    def __init__(self, config: dict):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.model = SentenceTransformer(
            config["id"], revision=str(config["revision"]),
            device=config.get("device"),
            local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        self.revision = str(config["revision"])
        self.tokenizer = self.model.tokenizer
        if self.tokenizer is None or self.model.max_seq_length is None:
            raise ValueError("selected model must expose a tokenizer and max_seq_length")
        configured_limit = config.get("max_tokens") or int(self.model.max_seq_length)
        margin = int(config.get("token_margin", 8))
        if configured_limit > self.model.max_seq_length or configured_limit - margin < 32:
            raise ValueError("configured model token limit/margin is invalid")
        self.max_tokens = configured_limit - margin
        native_dimension = int(self.model.get_embedding_dimension())
        self.dimension = int(config.get("truncate_dim") or native_dimension)
        if not 1 <= self.dimension <= native_dimension:
            raise ValueError("truncate_dim must fit the model's output dimension")

    def _prompt(self, side: str) -> tuple[str, str | None]:
        name = self.config.get(f"{side}_prompt_name")
        if name:
            if name not in self.model.prompts:
                raise ValueError(f"model has no {side} prompt named {name!r}")
            return self.model.prompts[name], name
        return self.config.get(f"{side}_prompt") or "", None

    def count_document_tokens(self, text: str) -> int:
        prompt, _ = self._prompt("document")
        return len(self.tokenizer.encode(prompt + text, add_special_tokens=True, verbose=False))

    def _encode(self, texts: list[str], side: str) -> list[list[float]]:
        prompt, name = self._prompt(side)
        array = self.model.encode(
            texts, **({"prompt_name": name} if name else {"prompt": prompt}),
            batch_size=int(self.config.get("batch_size", 16)),
            normalize_embeddings=bool(self.config.get("normalize", True)),
            truncate_dim=self.dimension, convert_to_numpy=True,
            show_progress_bar=False,
        )
        return array.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "document")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Several texts on the query side (question-to-question matching, e.g. task outline goals)."""
        return self._encode(texts, "query")

    def embed_query(self, text: str) -> list[float]:
        prompt, _ = self._prompt("query")
        if len(self.tokenizer.encode(prompt + text, add_special_tokens=True, verbose=False)) > self.model.max_seq_length:
            raise ValueError("query exceeds the selected model's token limit")
        return self._encode([text], "query")[0]


def _heading_prefix(chunk: dict) -> str:
    return "\n".join(chunk["headings"]) + "\n" if chunk["headings"] else ""


def split_chunk(chunk: dict, encoder: Encoder) -> list[dict]:
    """Split only oversized bodies; preserve every character and repeat headings."""
    missing = REQUIRED_CHUNK_FIELDS - chunk.keys()
    if missing:
        raise ValueError(f"{chunk.get('chunk_id', 'chunk')} missing fields: {sorted(missing)}")
    prefix = _heading_prefix(chunk)
    body = chunk["text"]
    if not body.strip():
        raise ValueError(f"empty text in {chunk['chunk_id']}")
    if chunk["contextual_text"] != prefix + body:
        raise ValueError(f"contextual text does not match headings and body in {chunk['chunk_id']}")
    if encoder.count_document_tokens(prefix) >= encoder.max_tokens:
        raise ValueError(f"headings alone exceed token limit in {chunk['chunk_id']}")
    if encoder.count_document_tokens(prefix + body) <= encoder.max_tokens:
        parts = [body]
    else:
        parts = []
        remaining = body
        while remaining:
            if encoder.count_document_tokens(prefix + remaining) <= encoder.max_tokens:
                parts.append(remaining)
                break
            low, high = 1, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if encoder.count_document_tokens(prefix + remaining[:middle]) <= encoder.max_tokens:
                    low = middle
                else:
                    high = middle - 1
            cut = low
            if cut < len(remaining):
                boundary = max(remaining.rfind("\n", 0, cut + 1),
                               remaining.rfind(" ", 0, cut + 1))
                if boundary >= cut // 2:
                    cut = boundary + 1
            if encoder.count_document_tokens(prefix + remaining[:cut]) > encoder.max_tokens:
                cut = low
                while cut > 0 and encoder.count_document_tokens(prefix + remaining[:cut]) > encoder.max_tokens:
                    cut -= max(1, cut // 10)
            if cut <= 0:
                raise ValueError(f"cannot split {chunk['chunk_id']} within token limit")
            parts.append(remaining[:cut])
            remaining = remaining[cut:]
    assert "".join(parts) == body
    result = []
    for number, part in enumerate(parts, 1):
        vector_chunk_id = f"{chunk['chunk_id']}:part-{number}-of-{len(parts)}"
        result.append({
            **chunk, "vector_chunk_id": vector_chunk_id,
            "parent_chunk_id": chunk["chunk_id"],
            "part_number": number, "part_count": len(parts),
            "text": part, "contextual_text": prefix + part,
            "embedding_tokens": encoder.count_document_tokens(prefix + part),
        })
    return result


def chunk_files(source_root: Path, select: str | None = None) -> list[Path]:
    paths = sorted(source_root.rglob("retrieval_chunks.jsonl"))
    if select:
        parts = tuple(Path(select).parts)
        paths = [path for path in paths if path.relative_to(source_root).parts[:len(parts)] == parts]
    if not paths:
        raise ValueError("no retrieval chunk files match the selection")
    return paths


def corpus_identity(paths: list[Path], source_root: Path, max_chunks: int | None) -> tuple[str, list[dict]]:
    fingerprint = []
    chunks = []
    for path in paths:
        fingerprint.append((str(path.relative_to(source_root)), hashlib.sha256(path.read_bytes()).hexdigest()))
        for line in path.open(encoding="utf-8"):
            if max_chunks is not None and len(chunks) >= max_chunks:
                break
            chunks.append(json.loads(line))
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    if not chunks:
        raise ValueError("selected retrieval chunk files contain no chunks")
    ids = [chunk.get("chunk_id") for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("selected corpus contains duplicate chunk IDs")
    return _digest({"files": fingerprint, "max_chunks": max_chunks}), chunks


def space_identity(config: IndexConfig, encoder: Encoder) -> tuple[str, str]:
    details = {
        "model": config.model, "resolved_revision": encoder.revision,
        "dimension": encoder.dimension, "effective_max_tokens": encoder.max_tokens,
        "distance": config.index.get("distance", "cosine"),
        "splitter": SPLITTER_VERSION,
        "sentence_transformers_version": version("sentence-transformers"),
    }
    return _digest(details), _digest(details)[:12]


def collection_name(config: IndexConfig, space_short: str, corpus_hash: str) -> str:
    prefix = re.sub(r"[^a-z0-9_-]+", "_", config.index.get("collection_prefix", "manuora").lower()).strip("_")
    model = re.sub(r"[^a-z0-9_-]+", "_", config.model["id"].split("/")[-1].lower()).strip("_")[:28]
    return f"{prefix}_{model}_{space_short}_{corpus_hash[:12]}"


def open_store(config: IndexConfig) -> QdrantClient:
    if config.store["mode"] == "local":
        return QdrantClient(path=str(config.location("path")))
    env = config.store.get("api_key_env")
    key = os.environ.get(env) if env else None
    if env and not key:
        raise ValueError(f"missing Qdrant API key environment variable: {env}")
    return QdrantClient(url=config.store["url"], api_key=key)


def _payload(part: dict, space_hash: str, corpus_hash: str) -> dict:
    return {
        key: part.get(key) for key in (
            "chunk_id", "vector_chunk_id", "parent_chunk_id", "part_number", "part_count",
            "product_id", "product_version", "document_id", "document_title", "document_type",
            "language", "source_pdf", "source_pdf_sha256", "pages", "headings", "refs",
            "labels", "content_kind", "text", "contextual_text", "embedding_tokens",
        )
    } | {"embedding_space_hash": space_hash, "corpus_hash": corpus_hash}


def build_index(config: IndexConfig, source_root: Path, *, select: str | None = None,
                max_chunks: int | None = None, encoder: Encoder | None = None) -> dict:
    total_started = perf_counter()
    if max_chunks is not None and max_chunks < 1:
        raise ValueError("max_chunks must be positive")
    files = chunk_files(source_root, select)
    corpus_hash, chunks = corpus_identity(files, source_root, max_chunks)
    if config.store["mode"] == "local" and len(chunks) > 10_000:
        raise ValueError("Manuora caps local pilot indexes at 10,000 points; configure a Qdrant server for this corpus")
    encoder = encoder or SentenceEncoder(config.model)
    parts = [part for chunk in chunks for part in split_chunk(chunk, encoder)]
    if config.store["mode"] == "local" and len(parts) > 10_000:
        raise ValueError("Manuora caps local pilot indexes at 10,000 points; configure a Qdrant server for this corpus")
    space_hash, space_short = space_identity(config, encoder)
    name = collection_name(config, space_short, corpus_hash)
    manifest_path = config.location("registry_path") / f"{name}.json"
    expected = {
        "collection": name, "space_hash": space_hash, "corpus_hash": corpus_hash,
        "model": config.model, "resolved_revision": encoder.revision,
        "dimension": encoder.dimension, "distance": config.index.get("distance", "cosine"),
        "effective_max_tokens": encoder.max_tokens, "splitter_version": SPLITTER_VERSION,
        "source_files": [str(path.relative_to(source_root)) for path in files],
        "input_chunks": len(chunks), "vector_points": len(parts),
        "estimated_vector_bytes": len(parts) * encoder.dimension * 4,
        "split_input_chunks": sum(part["part_count"] > 1 and part["part_number"] == 1 for part in parts),
        "qdrant_client_version": version("qdrant-client"),
        "sentence_transformers_version": version("sentence-transformers"),
    }
    client = open_store(config)
    try:
        exists = client.collection_exists(name)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(existing.get(key) != value for key, value in expected.items()):
                raise ValueError(f"index manifest disagrees with current config: {manifest_path}")
            if not exists:
                raise ValueError(f"index manifest exists but Qdrant collection is missing: {name}")
            vectors = client.get_collection(name).config.params.vectors
            expected_distance = (models.Distance.COSINE if expected["distance"] == "cosine"
                                 else models.Distance.DOT)
            if not isinstance(vectors, models.VectorParams) or (vectors.size, vectors.distance) != (
                encoder.dimension, expected_distance
            ):
                raise ValueError(f"Qdrant collection vector space disagrees with its manifest: {name}")
            current_points = client.count(name, exact=True).count
            if current_points > len(parts):
                raise ValueError(f"Qdrant collection has {current_points} points; expected at most {len(parts)}")
            if existing.get("state") == "complete" and current_points == len(parts):
                _ensure_filter_indexes(client, name, config)
                return existing
        elif exists:
            raise ValueError(f"Qdrant collection exists without a managed manifest: {name}")
        else:
            distance = models.Distance.COSINE if expected["distance"] == "cosine" else models.Distance.DOT
            client.create_collection(name, vectors_config=models.VectorParams(size=encoder.dimension, distance=distance))
        _atomic_json(manifest_path, {**expected, "state": "building"})
        started = perf_counter()
        prepare_seconds = started - total_started
        batch_size = int(config.index.get("upsert_batch_size", 32))
        print(f"Indexing {len(parts)} vectors into {name} ({len(chunks)} source chunks)", flush=True)
        for offset in range(0, len(parts), batch_size):
            batch = parts[offset:offset + batch_size]
            vectors = encoder.embed_documents([part["contextual_text"] for part in batch])
            if len(vectors) != len(batch) or any(len(vector) != encoder.dimension for vector in vectors):
                raise ValueError("embedding model returned unexpected vector dimensions")
            points = [models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{part['vector_chunk_id']}")),
                vector=vector, payload=_payload(part, space_hash, corpus_hash),
            ) for part, vector in zip(batch, vectors)]
            client.upsert(collection_name=name, points=points, wait=True)
            indexed = min(offset + batch_size, len(parts))
            if indexed == len(parts) or (offset // batch_size + 1) % 10 == 0:
                print(f"Upserted {indexed}/{len(parts)} vectors in {perf_counter() - started:.1f}s", flush=True)
        actual = client.count(name, exact=True).count
        if actual != len(parts):
            raise ValueError(f"Qdrant has {actual} points; expected {len(parts)}")
        _ensure_filter_indexes(client, name, config)
        result = {
            **expected, "state": "complete",
            "prepare_wall_seconds": round(prepare_seconds, 2),
            "embedding_upsert_wall_seconds": round(perf_counter() - started, 2),
            "total_wall_seconds": round(perf_counter() - total_started, 2),
            "gpu_usage": None,
        }
        _atomic_json(manifest_path, result)
        return result
    finally:
        client.close()


def _ensure_filter_indexes(client: QdrantClient, collection: str, config: IndexConfig) -> None:
    """Make the fields used by filtered retrieval searchable on Qdrant server."""
    if config.store["mode"] != "remote":
        return
    schema = client.get_collection(collection).payload_schema
    for field in ("product_id", "product_version", "document_id"):
        if field not in schema:
            client.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD, wait=True)


def search_index(config: IndexConfig, collection: str, query: str, *, limit: int = 5,
                 product_id: str | None = None, product_version: str | None = None,
                 document_id: str | None = None, encoder: Encoder | None = None) -> list[dict]:
    manifest_path = config.location("registry_path") / f"{collection}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise ValueError("collection has no completed index manifest")
    encoder = encoder or SentenceEncoder(config.model)
    space_hash, _ = space_identity(config, encoder)
    if space_hash != manifest["space_hash"] or encoder.dimension != manifest["dimension"]:
        raise ValueError("configured embedding model does not match the collection's vector space")
    conditions = [models.FieldCondition(key=key, match=models.MatchValue(value=value))
                  for key, value in (("product_id", product_id), ("product_version", product_version),
                                     ("document_id", document_id)) if value is not None]
    client = open_store(config)
    try:
        if not client.collection_exists(collection):
            raise ValueError(f"missing Qdrant collection: {collection}")
        result = client.query_points(
            collection_name=collection, query=encoder.embed_query(query),
            query_filter=models.Filter(must=conditions) if conditions else None,
            limit=limit, with_payload=True,
        )
        return [{"score": point.score, **(point.payload or {})} for point in result.points]
    finally:
        client.close()
