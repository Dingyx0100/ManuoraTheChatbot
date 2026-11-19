"""Local Manuora chat API and review interface."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid4

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from manuora.generation.local import ManuoraGraphGenerator
from manuora.generation.model_control import is_loaded, load_model, unload_model
from manuora.api.grounding import VersionIndex, missing_terms, passage_text, register_products, version_reply
from manuora.api.routing import answer_lacks_support, early_response, resolve_scope
from manuora.library import catalog
from manuora.library.indexer import LibraryIndex
from manuora.library.jobs import TYPES as MANUAL_TYPES, Library
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
        if _LIBRARY is not None:
            _LIBRARY.stop()


app = FastAPI(title="Manuora", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=ROOT / "frontend" / "assets"), name="assets")


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    product_id: str | None = None
    product_version: str | None = None
    document_id: str | None = None
    # Clients reuse the returned ID for subsequent questions. No shared default.
    conversation_id: UUID = Field(default_factory=uuid4)
    # Adds a "trace" (passages per stage, timings, outcome) to the reply; used by the evaluation runner.
    debug: bool = False
    # Evaluation only: stop after evidence packing; no model call, nothing stored in the conversation.
    retrieval_only: bool = False


# Settings are layered: config/assistant.yaml (or $MANUORA_CONFIG), then config/local.yaml (optional,
# not in git), then environment variables, then OVERRIDES (set in-process by the evaluation runner).
LOCAL_SETTINGS = ROOT / "config" / "local.yaml"
OVERRIDES: dict = {}
REQUIRED_SETTINGS = ("assistant_name", "embedding_config", "dense_collection", "sparse_index", "reranker.model_id",
                     "generation.model", "generation.url", "conversation.db_path", "retrieval.candidates_per_method",
                     "retrieval.rerank_candidates", "evidence.max_passages", "evidence.gate")


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in (extra or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def check_settings(options: dict) -> list[str]:
    """Problems that would only surface mid-request otherwise."""
    problems = []
    for dotted in REQUIRED_SETTINGS:
        node = options
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node in (None, ""):
            problems.append(f"missing setting {dotted}")
    gate = (options.get("evidence") or {}).get("gate")
    if gate not in (None, "absolute", "relative"):
        problems.append(f"evidence.gate must be absolute or relative, not {gate!r}")
    return problems


@lru_cache(maxsize=1)
def settings() -> dict:
    base = Path(os.environ.get("MANUORA_CONFIG") or SETTINGS)
    options = yaml.safe_load(base.read_text(encoding="utf-8"))
    if LOCAL_SETTINGS.is_file():
        options = _merge(options, yaml.safe_load(LOCAL_SETTINGS.read_text(encoding="utf-8")) or {})
    if os.environ.get("MANUORA_LLM_URL"):
        options = _merge(options, {"generation": {"url": os.environ["MANUORA_LLM_URL"]}})
    options = _merge(options, OVERRIDES)
    problems = check_settings(options)
    if problems:
        raise ValueError(f"invalid settings ({base}): " + "; ".join(problems))
    register_products(options.get("products") or {})
    return options


def data_root() -> Path:
    """Where manuals, indexes and conversations live: $MANUORA_DATA_DIR, else <repo>/data."""
    return Path(os.environ.get("MANUORA_DATA_DIR") or ROOT / "data")


def resolve_path(value: str | Path) -> Path:
    """Config paths: absolute as given; "data/..." under data_root(); anything else under the repo."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return data_root().joinpath(*path.parts[1:])
    return ROOT / path


def sparse_path() -> Path:
    return resolve_path(settings()["sparse_index"])


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
    db_path = resolve_path(memory.pop("db_path"))
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
    return OutlineIndex(resolve_path(path), pipeline()[0].encoder)


@lru_cache(maxsize=1)
def pipeline() -> tuple[HybridRetriever, QwenReranker, ManuoraGraphGenerator]:
    options = settings()
    collection = options.get("dense_collection")
    if not collection:
        raise ValueError("Qwen full-corpus collection is not configured yet")
    config = IndexConfig.load(ROOT / options["embedding_config"], ROOT)
    arm = options.get("retrieval") or {}  # experiment S; the defaults are production
    retriever = HybridRetriever(config, collection, sparse_path(),
                                lexical=arm.get("lexical") or "sqlite", fusion=arm.get("fusion") or "python",
                                rrf_k=int(arm.get("rrf_k") or 60), lexical_collection=arm.get("lexical_collection"))
    reranker = QwenReranker(**options["reranker"])
    return retriever, reranker, conversation_generator()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process answers. Use /ready for dependencies."""
    return {"status": "ok"}


def qdrant_status(options: dict) -> dict:
    """Is the dense collection reachable? (remote Qdrant: one HTTP call; local store: the folder exists)."""
    config = IndexConfig.load(ROOT / options["embedding_config"], ROOT)
    collection = options["dense_collection"]
    if ((options.get("retrieval") or {}).get("lexical") or "sqlite") != "sqlite":  # BM25 lives in Qdrant too
        from manuora.retrieval.qdrant_lexical import lexical_collection_name
        collection = options["retrieval"].get("lexical_collection") or lexical_collection_name(collection)
    if config.store["mode"] == "local":
        path = config.location("path")
        return {"ok": path.exists(), "detail": f"local store {path}"}
    url = config.store["url"].rstrip("/") + f"/collections/{collection}"
    with urllib.request.urlopen(url, timeout=3) as response:
        result = json.load(response).get("result") or {}
    return {"ok": True, "detail": f"{collection}: {result.get('points_count')} points, status {result.get('status')}"}


@app.get("/ready")
def ready():
    """Readiness: settings, the sparse index and Qdrant must work. The model is reported, not required,
    because it is loaded on demand from the page."""
    checks: dict[str, dict] = {}
    try:
        options = settings()
        checks["config"] = {"ok": True}
    except (ValueError, OSError, yaml.YAMLError) as exc:
        return JSONResponse(status_code=503, content={"ready": False, "checks": {"config": {"ok": False, "detail": str(exc)}}})
    try:
        connection = sqlite3.connect(f"file:{sparse_path().as_posix()}?mode=ro", uri=True)
        try:
            count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        finally:
            connection.close()
        checks["sparse_index"] = {"ok": count > 0, "detail": f"{count} chunks"}
    except (sqlite3.Error, OSError) as exc:
        checks["sparse_index"] = {"ok": False, "detail": f"{sparse_path()}: {exc}"}
    try:
        checks["qdrant"] = qdrant_status(options)
    except (OSError, ValueError, KeyError) as exc:  # URLError is an OSError
        checks["qdrant"] = {"ok": False, "detail": str(exc)}
    model = options["generation"]["model"]
    try:
        loaded = is_loaded(model)
    except (OSError, RuntimeError):
        loaded = None  # Docker Model Runner not reachable
    ok = all(check["ok"] for check in checks.values())
    body = {"ready": ok, "checks": checks, "model": {"model": model, "loaded": loaded}}
    return body if ok else JSONResponse(status_code=503, content=body)


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
    products, titles = catalog.names(data_root() / "sources")
    products = {**products, **{str(k): str(v) for k, v in (settings().get("products") or {}).items()}}
    return {"documents": [{"product_id": product, "product_version": version,
                            "document_id": document} for product, version, document in rows],
            "products": {product: products.get(product, product) for product, _, _ in rows},
            "titles": {document: titles.get(document, document) for _, _, document in rows}}


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


def _passages(hits: list[dict], char_limit: int | None = None) -> list[dict]:
    """What the trace keeps of a passage: enough to score it against gold (packed: the text the model sees)."""
    return [{"chunk_id": h.get("chunk_id"), "product_version": h.get("product_version"),
             "text": (h.get("text") or "")[:char_limit] if char_limit else h.get("text") or "",
             "rerank_score": h.get("rerank_score")} for h in hits]


def _chat(request: ChatRequest) -> dict:
    conversation_id = str(request.conversation_id)
    generator = None
    context = None
    draft = None
    turn = None
    # Passages per stage, seconds per step and why the reply ended the way it did.
    trace: dict = {"stages": {}, "seconds": {}, "outcome": None}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        trace["seconds"][name] = round(now - clock, 3)
        clock = now

    def reply(answer: str, sources: list | None = None, *, kind: str = 'answer', outcome: str | None = None) -> dict:
        if not request.retrieval_only:
            generator.record_displayed_reply(request.question, answer, thread_id=conversation_id,
                sources=sources or [], kind=kind, context=context, draft=draft)
        trace["outcome"] = outcome or kind
        body = {"answer": answer, "sources": sources or [],
                "conversation_id": conversation_id,
                "kind": kind, "abstained": kind == 'abstain',
                "routing":({k:turn.get(k) for k in ('action','method','reason','confidence','decision_seconds','rewrite_seconds','rewrite_issues')}
                           if turn else {'action':'chat','method':'early_rules'})}
        if request.debug:
            trace["query"], trace["scope"] = (turn or {}).get('query'), (turn or {}).get('scope')
            body["trace"] = trace
        return body

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
        lap("route")
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
                             kind='abstain', outcome='version_check')
        with MODEL_LOCK:
            retriever, reranker, generator = pipeline()
        controls = settings()["retrieval"]
        candidates = retriever.search(
            turn['query'], **turn['scope'],
            candidates=controls["candidates_per_method"],
        )
        lap("retrieve")
        ranked = reranker.rerank(
            turn['query'], candidates, limit=controls["rerank_candidates"],
            input_limit=controls["rerank_candidates"],
        )
        lap("rerank")
        evidence = select_evidence(ranked, **settings()["evidence"])
        if request.debug:
            trace["stages"].update(candidates=_passages(candidates), reranked=_passages(ranked), selected=_passages(evidence))
        if not evidence:
            return reply(UNCERTAIN_RESPONSE, kind='abstain', outcome='no_evidence')
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
            if request.debug:
                trace["stages"]["selected"] = _passages(evidence)
        lap("select")
        char_limit = settings()["evidence"].get("passage_char_limit")
        if request.retrieval_only:
            used = generator.packed_count(request.question, evidence)
            trace["stages"]["packed"] = _passages(evidence[:used], char_limit)
            return reply("", kind='retrieval_only')
        with MODEL_LOCK:
            draft = generator.draft_answer(request.question, evidence, thread_id=conversation_id,
                                           retrieval_question=turn['query'])
            answer = draft['answer']
            # Budget packing retains a ranked prefix; validate citations against
            # only passages actually supplied to generation.
            used = draft['stats'].get("evidence_count", len(evidence))
            evidence = evidence[:used]
        lap("generate")
        if request.debug:
            trace["stages"]["packed"] = _passages(evidence, char_limit)
            trace["draft"] = answer
        if answer_lacks_support(answer):
            return reply(UNCERTAIN_RESPONSE, kind='abstain', outcome='model_insufficient')
        cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
        if not cited or any(number < 1 or number > len(evidence) for number in cited):
            return reply(UNCERTAIN_RESPONSE, kind='abstain', outcome='invalid_citation')
        # Optional grounding checks (off by default; experiment O).
        grounding = settings().get("grounding") or {}
        if grounding.get("term_check"):
            texts = [passage_text(hit) for number, hit in enumerate(evidence, 1)
                     if grounding["term_check"] == "packed" or number in cited]
            if missing_terms(request.question, texts, spelling=bool(grounding.get("spelling"))):
                return reply(UNCERTAIN_RESPONSE, kind='abstain', outcome='term_check')
        if grounding.get("support_check"):
            with MODEL_LOCK:
                verdict = generator.check_support(request.question, answer, evidence)
            if not verdict["supported"]:
                return reply(UNCERTAIN_RESPONSE, kind='abstain', outcome='support_check')
        return reply(answer, outcome='answered', sources=[
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
    source_root = data_root() / "sources"
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


# --------------------------------------------------------------------------- library: upload and manage manuals
_LIBRARY: Library | None = None


def library() -> Library:
    """The upload worker starts on first use of the library API (not at import, so tests stay model-free)."""
    global _LIBRARY
    if _LIBRARY is None:
        sources = data_root() / "sources"

        def index_factory() -> LibraryIndex:
            options = settings()
            config = IndexConfig.load(ROOT / options["embedding_config"], ROOT)
            return LibraryIndex(config, options["dense_collection"], sparse_path(), sources,
                                lexical_collection=(options.get("retrieval") or {}).get("lexical_collection"))

        def changed() -> None:
            version_index.cache_clear()
            outline_index.cache_clear()
            register_products(catalog.names(sources)[0])

        def gpu_busy() -> bool:
            try:
                return bool(is_loaded(settings()["generation"]["model"]))
            except Exception:  # unknown: assume the GPU is taken
                return True

        _LIBRARY = Library(ROOT, sources, data_root() / "library", index_factory, changed, gpu_busy)
        _LIBRARY.start()
    return _LIBRARY


class SubmitRequest(BaseModel):
    upload_id: str
    display_name: str
    version: str | None = None
    title: str
    type: str
    product_id: str | None = None


@app.get("/api/library")
def library_overview() -> dict:
    lib = library()
    connection = sqlite3.connect(f"file:{sparse_path().as_posix()}?mode=ro", uri=True)
    try:
        counts = dict(connection.execute("SELECT document_id, COUNT(*) FROM chunks GROUP BY document_id"))
    finally:
        connection.close()
    products: dict[str, dict] = {}
    for manual in catalog.manuals(data_root() / "sources"):
        product = products.setdefault(manual.product_id, {"product_id": manual.product_id,
                                                          "display_name": manual.display_name, "manuals": []})
        product["manuals"].append({"document_id": manual.document_id, "title": manual.title, "type": manual.type,
                                   "version": manual.version, "chunks": counts.get(manual.document_id, 0),
                                   "indexed": manual.document_id in counts})
    return {"products": sorted(products.values(), key=lambda p: p["display_name"].casefold()),
            "jobs": [{k: v for k, v in job.items() if k not in ("upload_path", "manifest")} for job in lib.store.list()],
            "types": list(MANUAL_TYPES)}


@app.post("/api/library/uploads")
async def library_upload(request: Request, filename: str) -> dict:
    lib = library()
    target = lib.new_upload(filename)
    with target.open("wb") as out:
        async for chunk in request.stream():
            out.write(chunk)
    try:
        return lib.finish_upload(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/library/uploads/{upload_id}")
def library_discard(upload_id: str) -> dict:
    library().discard_upload(upload_id)
    return {"ok": True}


@app.post("/api/library/jobs")
def library_submit(body: SubmitRequest) -> dict:
    try:
        return library().submit(body.upload_id, display_name=body.display_name, version=body.version,
                                title=body.title, type=body.type, product_id=body.product_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/library/jobs/{job_id}/retry")
def library_retry(job_id: str) -> dict:
    try:
        return library().retry(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/library/jobs/{job_id}")
def library_dismiss(job_id: str) -> dict:
    try:
        library().dismiss(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/api/library/jobs/{job_id}/log")
def library_log(job_id: str) -> dict:
    return {"log": library().log(job_id)}


@app.delete("/api/library/documents/{document_id}")
def library_remove(document_id: str) -> dict:
    try:
        return library().remove(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="manual not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
