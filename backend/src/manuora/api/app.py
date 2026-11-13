"""Local Manuora chat API and review interface."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid4

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from manuora.generation.local import ManuoraGraphGenerator
from manuora.generation.model_control import is_loaded, load_model, unload_model
from manuora.api.grounding import VersionIndex, missing_terms, passage_text, register_products, version_reply
from manuora.api.routing import answer_lacks_support, early_response, resolve_scope
from manuora.retrieval.evidence import select_evidence
from manuora.retrieval.hybrid import HybridRetriever
from manuora.retrieval.indexing import IndexConfig
from manuora.retrieval.outlines import OutlineIndex, card_chunks, merge_outline
from manuora.retrieval.rerank import QwenReranker
from manuora.retrieval.sparse import get_chunks


ROOT = Path(__file__).resolve().parents[4]
SETTINGS = ROOT / "config" / "assistant.yaml"
MODEL_LOCK = threading.RLock()
UNCERTAIN_RESPONSE = "I can't verify that from the information I have. Could you share a little more detail?"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        with MODEL_LOCK:
            # Do not construct a generator just to close an unused app.
            if conversation_generator.cache_info().currsize:
                conversation_generator().close()
                conversation_generator.cache_clear()
            pipeline.cache_clear()


app = FastAPI(title="Manuora", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    product_id: str | None = None
    product_version: str | None = None
    document_id: str | None = None
    # Clients reuse the returned ID for subsequent questions. No shared default.
    conversation_id: UUID = Field(default_factory=uuid4)


LOCAL_SETTINGS = ROOT / "config" / "local.yaml"  # optional, not in git: machine-specific overrides


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in (extra or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


@lru_cache(maxsize=1)
def settings() -> dict:
    options = yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))
    if LOCAL_SETTINGS.is_file():
        options = _merge(options, yaml.safe_load(LOCAL_SETTINGS.read_text(encoding="utf-8")) or {})
    register_products(options.get("products") or {})
    return options


def sparse_path() -> Path:
    return ROOT / settings()["sparse_index"]


@lru_cache(maxsize=1)
def conversation_generator() -> ManuoraGraphGenerator:
    options = settings()
    generation = dict(options["generation"])
    # Legacy keys from the archived stateless DockerGenerator; the graph generator ignores them.
    generation.pop("history_turns", None)
    generation.pop("history_chars_per_turn", None)
    memory = dict(options["conversation"])
    if memory.get('context_router_model_path'):
        memory['context_router_model_path']=str(ROOT / memory['context_router_model_path'])
    db_path = Path(memory.pop("db_path"))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = options["evidence"]
    return ManuoraGraphGenerator(**generation, **memory, db_path=str(db_path),
                                 max_evidence_passages=evidence["max_passages"],
                                 passage_char_limit=evidence["passage_char_limit"])


@lru_cache(maxsize=2)
def version_index(source: str) -> VersionIndex:
    return VersionIndex(sparse_path(), source=source)


@lru_cache(maxsize=1)
def outline_index(path: str) -> OutlineIndex:
    return OutlineIndex(ROOT / path, pipeline()[0].encoder)


@lru_cache(maxsize=1)
def pipeline() -> tuple[HybridRetriever, QwenReranker, ManuoraGraphGenerator]:
    options = settings()
    collection = options.get("dense_collection")
    if not collection:
        raise ValueError("Qwen full-corpus collection is not configured yet")
    config = IndexConfig.load(ROOT / options["embedding_config"], ROOT)
    retriever = HybridRetriever(config, collection, sparse_path())
    reranker = QwenReranker(**options["reranker"])
    return retriever, reranker, conversation_generator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/options")
def options() -> dict:
    connection = sqlite3.connect(f"file:{sparse_path().as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute("""
            SELECT DISTINCT product_id, product_version, document_id
            FROM chunks ORDER BY product_id, product_version, document_id
        """).fetchall()
    finally:
        connection.close()
    return {"documents": [{"product_id": product, "product_version": version,
                            "document_id": document} for product, version, document in rows]}


@app.get("/api/model/status")
def model_status() -> dict:
    model = settings()["generation"]["model"]
    try:
        return {"model": model, "loaded": is_loaded(model)}
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/model/load")
def model_load() -> dict:
    generation = settings()["generation"]
    try:
        with MODEL_LOCK:
            if not is_loaded(generation["model"]):
                load_model(generation["model"], generation["url"])
        return {"model": generation["model"], "loaded": True}
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/model/unload")
def model_unload() -> dict:
    model = settings()["generation"]["model"]
    try:
        with MODEL_LOCK:
            unload_model(model)
        return {"model": model, "loaded": False}
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    # Local single-user service: serialize the entire read/draft/commit exchange,
    # including model unload. Multi-user serving will need per-session scheduling.
    with MODEL_LOCK:
        return _chat(request)


def _chat(request: ChatRequest) -> dict:
    conversation_id = str(request.conversation_id)
    generator = None
    context = None
    draft = None
    turn = None

    def reply(answer: str, sources: list | None = None, *, kind: str = 'answer') -> dict:
        generator.record_displayed_reply(request.question, answer, thread_id=conversation_id,
            sources=sources or [], kind=kind, context=context, draft=draft)
        return {"answer": answer, "sources": sources or [],
                "conversation_id": conversation_id,
                "routing":({k:turn.get(k) for k in ('action','method','reason','confidence','decision_seconds','rewrite_seconds','rewrite_issues')}
                           if turn else {'action':'chat','method':'early_rules'})}

    try:
        documents = options()["documents"]
        generator = conversation_generator()
        context = generator.prepare_retrieval_context(request.question, thread_id=conversation_id,
            documents=documents, ui_scope={'product_id':request.product_id,
                'product_version':request.product_version,'document_id':request.document_id})
        response = early_response(request.question, settings()["assistant_name"],
                                  {row["product_id"] for row in documents})
        if response is not None:
            return reply(response, kind='chat')
        turn = generator.prepare_turn(request.question,thread_id=conversation_id,documents=documents,context=context)
        if turn['action']=='clarify':
            return reply(turn['reason'],kind='clarify')
        if turn['action']=='edit_previous_answer':
            edited=generator.edit_previous_answer(request.question,context=context)
            turn['reason'] += '; previous answer retained because edit could not be verified' if edited.get('fallback') else ''
            return reply(edited['answer'],edited['sources'],kind=edited['kind'])
        # Serialize first construction too: lru_cache may otherwise construct
        # two generators on simultaneous first requests.
        # Optional version check (off by default; experiment P): a feature that the asked version's
        # manual never mentions, while another version's does, gets a grounded premise reply.
        source = (settings().get("grounding") or {}).get("version_check")
        if source:
            scope = turn['scope']
            hits = version_index(source).other_version_features(
                request.question, scope.get('product_id'), scope.get('product_version'))
            if hits:
                return reply(version_reply(scope['product_id'], str(scope['product_version']), hits[0]),
                             kind='abstain')
        with MODEL_LOCK:
            retriever, reranker, generator = pipeline()
        controls = settings()["retrieval"]
        candidates = retriever.search(
            turn['query'], **turn['scope'],
            candidates=controls["candidates_per_method"],
        )
        ranked = reranker.rerank(
            turn['query'], candidates, limit=controls["rerank_candidates"],
            input_limit=controls["rerank_candidates"],
        )
        evidence = select_evidence(ranked, **settings()["evidence"])
        if not evidence:
            return reply(UNCERTAIN_RESPONSE, kind='abstain')
        # Optional task outlines (off by default; experiment Q): passages linked offline to the
        # matched task join the evidence after the best reranked passage.
        outline = controls.get("outlines")
        if outline:
            scope = turn['scope']
            matched = outline_index(outline["path"]).match(
                retriever.encoder.embed_query(turn['query']), scope.get('product_id'),
                [scope.get('product_version')], top_k=int(outline.get("top_k", 1)),
                min_score=float(outline.get("min_score", 0.7)))
            ids = list(dict.fromkeys(c for card, _ in matched for c in card_chunks(card)))
            found = get_chunks(sparse_path(), ids)
            extra = [{**found[c], "rerank_score": None, "retrievers": ["outline"]} for c in ids if c in found]
            evidence = merge_outline(evidence, extra, settings()["evidence"]["max_passages"])
        with MODEL_LOCK:
            draft = generator.draft_answer(request.question, evidence, thread_id=conversation_id,
                                           retrieval_question=turn['query'])
            answer = draft['answer']
            # Budget packing retains a ranked prefix; validate citations against
            # only passages actually supplied to generation.
            used = draft['stats'].get("evidence_count", len(evidence))
            evidence = evidence[:used]
        if answer_lacks_support(answer):
            return reply(UNCERTAIN_RESPONSE, kind='abstain')
        cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
        if not cited or any(number < 1 or number > len(evidence) for number in cited):
            return reply(UNCERTAIN_RESPONSE, kind='abstain')
        # Optional grounding checks (off by default; experiment O).
        grounding = settings().get("grounding") or {}
        if grounding.get("term_check"):
            texts = [passage_text(hit) for number, hit in enumerate(evidence, 1)
                     if grounding["term_check"] == "packed" or number in cited]
            if missing_terms(request.question, texts, spelling=bool(grounding.get("spelling"))):
                return reply(UNCERTAIN_RESPONSE, kind='abstain')
        if grounding.get("support_check"):
            with MODEL_LOCK:
                verdict = generator.check_support(request.question, answer, evidence)
            if not verdict["supported"]:
                return reply(UNCERTAIN_RESPONSE, kind='abstain')
        return reply(answer, [
            {"citation": number, "chunk_id": hit["chunk_id"],
             "document_id": hit["document_id"], "product_id": hit["product_id"],
             "product_version": hit["product_version"], "pages": hit["pages"],
             "headings": hit["headings"], "preview": hit["text"][:550],
             "retrievers": hit["retrievers"], "rerank_score": hit["rerank_score"]}
            for number, hit in enumerate(evidence, 1) if number in cited
        ])
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/source")
def source(product_id: str, document_id: str, product_version: str = "") -> FileResponse:
    source_root = ROOT / "data" / "sources"
    for chunks_path in source_root.rglob("retrieval_chunks.jsonl"):
        with chunks_path.open(encoding="utf-8") as stream:
            first = json.loads(next(stream))
        if (first.get("product_id"), first.get("product_version") or "",
                first.get("document_id")) != (product_id, product_version, document_id):
            continue
        filename = first.get("source_pdf", "")
        if Path(filename).name != filename:
            break
        for directory in (chunks_path.parent, *chunks_path.parents):
            if not directory.is_relative_to(source_root):
                break
            candidate = directory / filename
            if candidate.is_file():
                return FileResponse(candidate, media_type="application/pdf", filename=filename,
                                    content_disposition_type="inline")
    raise HTTPException(status_code=404, detail="Source PDF not found")
