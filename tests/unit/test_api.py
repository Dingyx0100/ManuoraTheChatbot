"""FastAPI endpoints, routing guards and model controls."""
from fastapi.testclient import TestClient
import importlib
import sqlite3
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from manuora.api.app import app
from manuora.api.routing import answer_lacks_support, early_response
from manuora.generation.context import explicit_scope


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_scope_routes_weather_and_explicit_version() -> None:
    documents = [{"product_id": "nextcloud", "product_version": "36", "document_id": "guide"}]
    assert early_response("How is the weather now?", "Manuora", {"nextcloud"}) is not None
    assert early_response("How do I configure the weather app in Nextcloud?", "Manuora", {"nextcloud"}) is None
    assert early_response("What's the weather in Nextcloud now?", "Manuora", {"nextcloud"}) is not None
    scope, _, conflict = explicit_scope("Where is SMTP in Nextcloud 36?", {}, documents)
    assert (scope["product_id"], scope["product_version"], conflict) == ("nextcloud", "36", None)
    # Rules v2 also answers greetings and thanks with a few extra words, but never a question.
    assert early_response("hi there!", "Manuora", {"nextcloud"}) is None
    assert early_response("hi there!", "Manuora", {"nextcloud"}, rules="v2") is not None
    assert early_response("thanks, that helped a lot", "Manuora", {"nextcloud"}, rules="v2") == "You're welcome!"
    assert early_response("thanks, how do I export it?", "Manuora", {"nextcloud"}, rules="v2") is None
    assert early_response("hello Nextcloud", "Manuora", {"nextcloud"}, rules="v2") is None
    assert answer_lacks_support("The manual does not provide configuration instructions [1].")
    assert not answer_lacks_support("Open the administration settings [1].")
    # The marker after an explanation is still a refusal (negative N09 in experiment N).
    assert answer_lacks_support("The retrieved evidence does not cover the Layout Manager [4].\n\nINSUFFICIENT_EVIDENCE")
    assert answer_lacks_support("Therefore, the answer is: INSUFFICIENT_EVIDENCE.")


def test_chat_rejects_weak_evidence_and_only_shows_cited_sources(monkeypatch) -> None:
    api = importlib.import_module("manuora.api.app")
    documents = [{"product_id": "nextcloud", "product_version": "36", "document_id": "guide"}]
    monkeypatch.setattr(api, "options", lambda: {"documents": documents})
    monkeypatch.setattr(api, "settings", lambda: {"assistant_name": "Manuora",
        "retrieval": {"candidates_per_method": 30, "rerank_candidates": 12},
        "evidence": {"max_passages": 4, "gate": "absolute", "minimum_score": 3.0,
                     "relative_margin": None, "passage_char_limit": 850}})

    class Retriever:
        def search(self, query, **filters):
            assert filters["product_id"] == "nextcloud"
            assert filters["product_version"] == "36"
            return [{}]

    class Reranker:
        score = 2.0

        def rerank(self, query, candidates, **kwargs):
            return [{"rerank_score": self.score, "chunk_id": str(i),
                     "document_id": "guide", "product_id": "nextcloud",
                     "product_version": "36", "pages": [i], "headings": [],
                     "text": "sample evidence", "retrievers": ["dense"]}
                    for i in (1, 2)]

    class Generator:
        answer = "The setting is here [1]."
        last_prompt_stats = {"evidence_count": 1}

        def draft_answer(self, question, evidence, *, thread_id, retrieval_question=None):
            assert thread_id == conversation_id
            return {'answer':self.answer, 'stats':self.last_prompt_stats}

        def prepare_retrieval_context(self, *args, **kwargs):
            return {}

        def prepare_turn(self, question, **kwargs):
            return {'action':'direct_retrieval','query':question,'scope':{'product_id':'nextcloud','product_version':'36','document_id':None}}

        def record_displayed_reply(self, *args, **kwargs):
            pass

        def close(self):
            pass

    reranker = Reranker()
    generator = Generator()
    conversation_id = str(uuid4())
    monkeypatch.setattr(api, 'conversation_generator', lru_cache(maxsize=1)(lambda:generator))
    monkeypatch.setattr(api, "pipeline", lru_cache(maxsize=1)(lambda: (Retriever(), reranker, generator)))
    with TestClient(app) as client:
        weak = client.post("/api/chat", json={"question": "Where is SMTP in Nextcloud 36?"}).json()
        reranker.score = 6.0
        strong = client.post("/api/chat", json={"question": "Where is SMTP in Nextcloud 36?",
                                                "conversation_id": conversation_id}).json()
        Generator.answer = "The setting is here [2]."
        omitted = client.post("/api/chat", json={"question": "Where is SMTP in Nextcloud 36?",
                                                 "conversation_id": conversation_id}).json()
    assert weak["sources"] == [] and "can't verify" in weak["answer"]
    assert weak["abstained"] is True and "trace" not in weak
    assert strong["conversation_id"] == conversation_id
    assert [source["citation"] for source in strong["sources"]] == [1]
    assert strong["abstained"] is False and strong["kind"] == "answer"
    # The UI shows what was searched under each reply, abstentions included.
    assert strong["scope"] == weak["scope"] == {"products": ["nextcloud"], "versions": ["36"]}
    assert omitted["sources"] == [] and "can't verify" in omitted["answer"]
    assert omitted["abstained"] is True


def _fake_pipeline(monkeypatch, api):
    """A model-free turn that always answers "Use SMTP [1]." from one Nextcloud 36 passage; returns what was recorded."""
    documents = [{"product_id": "nextcloud", "product_version": "36", "document_id": "guide"}]
    monkeypatch.setattr(api, "options", lambda: {"documents": documents})
    monkeypatch.setattr(api, "settings", lambda: {"assistant_name": "Manuora",
        "retrieval": {"candidates_per_method": 30, "rerank_candidates": 12},
        "evidence": {"max_passages": 4, "gate": "absolute", "minimum_score": 3.0,
                     "relative_margin": None, "passage_char_limit": 850}})
    hit = {"rerank_score": 6.0, "chunk_id": "1", "document_id": "guide", "product_id": "nextcloud",
           "product_version": "36", "pages": [1], "headings": [], "text": "sample evidence", "retrievers": ["dense"]}
    recorded = []

    class Retriever:
        def search(self, query, **filters): return [hit]

    class Reranker:
        def rerank(self, query, candidates, **kwargs): return [hit]

    class Generator:
        def draft_answer(self, question, evidence, **kwargs):
            return {"answer": "Use SMTP [1].", "stats": {"evidence_count": 1}}
        def prepare_retrieval_context(self, *args, **kwargs): return {}
        def prepare_turn(self, question, **kwargs):
            return {"action": "direct_retrieval", "query": question,
                    "scope": {"product_id": "nextcloud", "product_version": "36", "document_id": None}}
        def record_displayed_reply(self, question, answer, **kwargs): recorded.append(answer)
        def close(self): pass

    generator = Generator()
    monkeypatch.setattr(api, "conversation_generator", lru_cache(maxsize=1)(lambda: generator))
    monkeypatch.setattr(api, "pipeline", lru_cache(maxsize=1)(lambda: (Retriever(), Reranker(), generator)))
    return recorded


def test_chat_stream_reports_stages_then_the_checked_reply(monkeypatch) -> None:
    import json
    api = importlib.import_module("manuora.api.app")
    recorded = _fake_pipeline(monkeypatch, api)
    with TestClient(app) as client:
        response = client.post("/api/chat/stream", json={"question": "Where is SMTP in Nextcloud 36?"})
        events = [json.loads(line) for line in response.text.splitlines()]
        plain = client.post("/api/chat", json={"question": "Where is SMTP in Nextcloud 36?"}).json()
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [e["stage"] for e in events[:-1]] == ["route", "search", "rerank", "write", "check"]
    final = events[-1]["reply"]
    # Only stages stream; the reply is the same checked body /api/chat returns.
    assert final["answer"] == plain["answer"] == "Use SMTP [1]." and final["sources"][0]["citation"] == 1
    assert final["scope"] == {"products": ["nextcloud"], "versions": ["36"]}
    assert recorded == ["Use SMTP [1].", "Use SMTP [1]."]


def test_cancelled_turn_stops_at_the_next_stage_and_records_nothing(monkeypatch) -> None:
    api = importlib.import_module("manuora.api.app")
    recorded = _fake_pipeline(monkeypatch, api)
    seen = []

    def progress(stage):
        if stage: seen.append(stage)
        if stage == "write":
            raise api.Cancelled

    with pytest.raises(api.Cancelled):
        api._chat(api.ChatRequest(question="Where is SMTP in Nextcloud 36?"), progress)
    assert seen == ["route", "search", "rerank", "write"] and recorded == []


def test_debug_trace_reports_stages_timings_and_outcome(monkeypatch) -> None:
    api = importlib.import_module("manuora.api.app")
    documents = [{"product_id": "nextcloud", "product_version": "36", "document_id": "guide"}]
    monkeypatch.setattr(api, "options", lambda: {"documents": documents})
    monkeypatch.setattr(api, "settings", lambda: {"assistant_name": "Manuora",
        "retrieval": {"candidates_per_method": 30, "rerank_candidates": 12},
        "evidence": {"max_passages": 4, "gate": "absolute", "minimum_score": 3.0,
                     "relative_margin": None, "passage_char_limit": None}})
    hits = [{"rerank_score": s, "chunk_id": c, "document_id": "guide", "product_id": "nextcloud",
             "product_version": "36", "pages": [1], "headings": [], "text": f"text {c}", "retrievers": ["dense"]}
            for c, s in (("a", 6.0), ("b", 5.0), ("c", 1.0))]

    class Retriever:
        def search(self, query, **filters):
            return hits

    class Reranker:
        def rerank(self, query, candidates, **kwargs):
            return candidates

    class Generator:
        def draft_answer(self, question, evidence, **kwargs):
            return {"answer": "Use it [1].", "stats": {"evidence_count": 1}}

        def prepare_retrieval_context(self, *args, **kwargs):
            return {}

        def prepare_turn(self, question, **kwargs):
            return {"action": "direct_retrieval", "query": question,
                    "scope": {"product_id": "nextcloud", "product_version": "36", "document_id": None}}

        def record_displayed_reply(self, *args, **kwargs):
            pass

        def close(self):
            pass

    generator = Generator()
    monkeypatch.setattr(api, "conversation_generator", lru_cache(maxsize=1)(lambda: generator))
    monkeypatch.setattr(api, "pipeline", lru_cache(maxsize=1)(lambda: (Retriever(), Reranker(), generator)))
    with TestClient(app) as client:
        body = client.post("/api/chat", json={"question": "Where is SMTP in Nextcloud 36?", "debug": True}).json()
    trace = body["trace"]
    assert trace["outcome"] == "answered" and body["abstained"] is False
    assert [p["chunk_id"] for p in trace["stages"]["candidates"]] == ["a", "b", "c"]
    assert [p["chunk_id"] for p in trace["stages"]["selected"]] == ["a", "b"]  # gate drops c
    assert [p["chunk_id"] for p in trace["stages"]["packed"]] == ["a"]  # the generator packed one passage
    assert {"route", "retrieve", "rerank", "select", "generate"} <= set(trace["seconds"])


def test_settings_layers_env_overrides_and_validation(monkeypatch, tmp_path) -> None:
    api = importlib.import_module("manuora.api.app")
    base = yaml.safe_load(api.SETTINGS.read_text(encoding="utf-8"))
    config = tmp_path / "assistant.yaml"
    config.write_text(yaml.safe_dump(base), encoding="utf-8")
    monkeypatch.setenv("MANUORA_CONFIG", str(config))
    monkeypatch.setenv("MANUORA_LLM_URL", "http://model-runner:12434/v1/chat")
    monkeypatch.setenv("MANUORA_DATA_DIR", str(tmp_path / "srv"))
    monkeypatch.setattr(api, "LOCAL_SETTINGS", tmp_path / "absent.yaml")
    monkeypatch.setattr(api, "OVERRIDES", {"evidence": {"max_passages": 12}})
    api.settings.cache_clear()
    try:
        options = api.settings()
        assert options["generation"]["url"] == "http://model-runner:12434/v1/chat"
        assert options["evidence"]["max_passages"] == 12 and options["evidence"]["gate"] == base["evidence"]["gate"]
        assert api.sparse_path() == tmp_path / "srv" / Path(base["sparse_index"]).relative_to("data")
        assert api.resolve_path("config/x.yaml") == api.ROOT / "config/x.yaml"
        del base["dense_collection"]
        base["evidence"]["gate"] = "sometimes"
        config.write_text(yaml.safe_dump(base), encoding="utf-8")
        api.settings.cache_clear()
        with pytest.raises(ValueError, match="dense_collection.*evidence.gate"):
            api.settings()
    finally:
        api.settings.cache_clear()


def test_ready_checks_dependencies_and_reports_the_model(monkeypatch, tmp_path) -> None:
    api = importlib.import_module("manuora.api.app")
    index = tmp_path / "bm25.sqlite"
    connection = sqlite3.connect(index)
    connection.execute("create table chunks (chunk_id text)")
    connection.execute("insert into chunks values ('a')")
    connection.commit()
    connection.close()
    monkeypatch.setattr(api, "settings", lambda: {"sparse_index": str(index), "generation": {"model": "m"}})
    monkeypatch.setattr(api, "qdrant_status", lambda options: {"ok": True, "detail": "stub"})
    monkeypatch.setattr(api, "is_loaded", lambda model: False)
    with TestClient(app) as client:
        good = client.get("/ready")
        assert good.status_code == 200 and good.json()["ready"] is True
        assert good.json()["model"] == {"model": "m", "loaded": False}  # not loaded is still ready
        monkeypatch.setattr(api, "qdrant_status", lambda options: (_ for _ in ()).throw(OSError("connection refused")))
        down = client.get("/ready")
        assert down.status_code == 503 and "refused" in down.json()["checks"]["qdrant"]["detail"]
        monkeypatch.setattr(api, "settings", lambda: {"sparse_index": str(tmp_path / "missing.sqlite"),
                                                      "generation": {"model": "m"}})
        monkeypatch.setattr(api, "qdrant_status", lambda options: {"ok": True})
        missing = client.get("/ready")
        assert missing.status_code == 503 and missing.json()["checks"]["sparse_index"]["ok"] is False


def test_model_load_and_unload_controls(monkeypatch) -> None:
    api = importlib.import_module("manuora.api.app")
    state = {"loaded": False}
    monkeypatch.setattr(api, "settings", lambda: {"generation": {"model": "test-model", "url": "http://localhost/test"}})
    monkeypatch.setattr(api, "is_loaded", lambda model: state["loaded"])
    monkeypatch.setattr(api, "load_model", lambda model, url: state.update(loaded=True))
    monkeypatch.setattr(api, "unload_model", lambda model: state.update(loaded=False))
    with TestClient(app) as client:
        assert client.get("/api/model/status").json()["loaded"] is False
        assert client.post("/api/model/load").json()["loaded"] is True
        assert client.post("/api/model/unload").json()["loaded"] is False


def test_api_commits_only_displayed_replies_including_early_and_rejected_answers(memory_model_transport, monkeypatch, tmp_path):
    local, payloads, replies = memory_model_transport
    api=importlib.import_module('manuora.api.app')
    config=dict(api.settings())
    config['conversation']=dict(config['conversation'],db_path=str(tmp_path/'displayed.sqlite'),summarize=False,
        context_router_method='rules',context_rules='v1')
    monkeypatch.setattr(api,'settings',lambda:config)
    hit={'chunk_id':'one','document_id':'guide','product_id':'nextcloud','product_version':'36',
         'pages':[1],'headings':['SMTP'],'text':'PRIVATE_EVIDENCE_BODY','retrievers':['dense'],'rerank_score':6.0}
    monkeypatch.setattr(api,'options',lambda:{'documents':[hit]})
    class Retriever:
        def search(self,*args,**kwargs): return [hit]
    class Reranker:
        def rerank(self,*args,**kwargs): return [hit]
    monkeypatch.setattr(api,'pipeline',lru_cache(maxsize=1)(lambda:(Retriever(),Reranker(),api.conversation_generator())))
    cid=str(uuid4())
    with TestClient(api.app) as client:
        def ask(question):
            return client.post('/api/chat',json={'question':question,'conversation_id':cid})
        greeting=ask('Hello').json()
        gen=api.conversation_generator()
        assert gen.get_history(cid)[-1].content==greeting['answer'] and not payloads
        accepted=ask('Where is SMTP in Nextcloud 36?').json()
        assert accepted['sources'] and gen.get_history(cid)[-1].additional_kwargs['sources'][0]['chunk_id']=='one'
        replies['answer']='REJECTED_DRAFT_MARKER [99].'
        rejected=ask('Which SMTP option in Nextcloud 36?').json()
        assert rejected['answer']==api.UNCERTAIN_RESPONSE and gen.get_history(cid)[-1].content==rejected['answer']
        checkpoints=list(gen._saver.list({'configurable':{'thread_id':cid}}))
        assert 'REJECTED_DRAFT_MARKER' not in str(checkpoints) and 'PRIVATE_EVIDENCE_BODY' not in str(checkpoints)
        hit['rerank_score']=1.0
        before=len(payloads)
        weak=ask('How do I configure Nextcloud 36 SMTP?').json()
        assert gen.get_history(cid)[-1].content==weak['answer'] and len(payloads)==before
        hit['rerank_score']=6.0
        replies['answer']=''
        old=[m.content for m in gen.get_history(cid)]
        failure=ask('Where is Nextcloud 36 SMTP?')
        assert failure.status_code==503 and [m.content for m in gen.get_history(cid)]==old
        assert len(old)==8  # Exactly four completed exchanges, no failed-request user tail.


def test_four_action_api_paths_share_rewrite_query_and_keep_edit_citations(memory_model_transport,monkeypatch,tmp_path):
    local,payloads,replies=memory_model_transport
    api=importlib.import_module('manuora.api.app')
    configuration=dict(api.settings())
    configuration['conversation']=dict(configuration['conversation'],db_path=str(tmp_path/'routes.sqlite'),summarize=False,
        context_router_method='rules',context_rules='v1')
    monkeypatch.setattr(api,'settings',lambda:configuration)
    hit={'chunk_id':'smtp','document_id':'guide','product_id':'nextcloud','product_version':'36',
         'pages':[1],'headings':['SMTP'],'text':'Use SMTP settings.','rerank_score':7.0,'retrievers':['dense']}
    monkeypatch.setattr(api,'options',lambda:{'documents':[hit]})
    searched=[];reranked=[]
    class Retriever:
        def search(self,query,**filters): searched.append((query,filters));return [hit]
    class Reranker:
        def rerank(self,query,hits,**kwargs): reranked.append(query);return hits
    monkeypatch.setattr(api,'pipeline',lru_cache(maxsize=1)(lambda:(Retriever(),Reranker(),api.conversation_generator())))
    cid=str(uuid4())
    with TestClient(api.app) as client:
        def ask(question,thread=cid):
            response=client.post('/api/chat',json={'question':question,'conversation_id':thread})
            assert response.status_code==200,response.text
            return response.json()
        direct=ask('Where is SMTP in Nextcloud 36?')
        assert direct['routing']['action']=='direct_retrieval'
        rewrite=ask('Where do I configure that?')
        assert rewrite['routing']['action']=='contextual_rewrite'
        assert searched[-1][0]==reranked[-1]==replies['rewrite']
        assert searched[-1][1]['product_version']=='36'
        assert 'Resolved search question' in payloads[-1]['messages'][-1]['content']
        before=len(searched)
        edited=ask('Make that answer shorter.')
        assert edited['routing']['action']=='edit_previous_answer' and len(searched)==before
        assert edited['answer']==replies['edit'] and edited['sources'][0]['chunk_id']=='smtp'
        replies['edit']='UNSUPPORTED_EDIT [99].'
        failed_edit=ask('Make that answer shorter.')
        assert failed_edit['answer']==edited['answer'] and len(searched)==before
        count=len(payloads)
        clarify=ask('What does that button do?',str(uuid4()))
        assert clarify['routing']['action']=='clarify' and len(payloads)==count and len(searched)==before
        # Only replies that searched the manuals carry a scope for the UI to show.
        assert direct['scope']=={'products':['nextcloud'],'versions':['36']} and edited['scope'] is clarify['scope'] is None
        replies['rewrite']='Why is FreeCAD 0.16 limited to 2D?'
        rejected=ask('Where do I configure that?')
        assert rejected['routing']['rewrite_issues'] and searched[-1][0]==reranked[-1]
        assert 'Conversation context' in searched[-1][0]
        checkpoints=str(list(api.conversation_generator()._saver.list({'configurable':{'thread_id':cid}})))
        assert 'UNSUPPORTED_EDIT' not in checkpoints
