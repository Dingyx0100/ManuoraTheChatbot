"""LangGraph generator: persistence, trimming, summaries and token budgets."""
from fastapi.testclient import TestClient
import importlib
import json
from functools import lru_cache
from uuid import uuid4
import httpx
import pytest
from manuora.api.app import app
from conftest import assert_requests_fit


def test_graph_conversations_persist_isolate_and_survive_model_controls(monkeypatch, tmp_path) -> None:
    """Real API + graph + SQLite + ChatOpenAI serialization; only inference is fake."""
    api = importlib.import_module("manuora.api.app")
    local = importlib.import_module("manuora.generation.local")
    configuration = dict(api.settings())
    configuration["conversation"] = dict(configuration["conversation"],
        db_path="memory/chat.sqlite", summary_after_exchanges=8)
    monkeypatch.setattr(api, "settings", lambda: configuration)
    monkeypatch.setattr(api, "ROOT", tmp_path)
    hit = {"chunk_id": "one", "document_id": "guide", "product_id": "nextcloud",
           # The SMTP heading keeps the questions' names in the evidence (grounding.term_check).
           "product_version": "36", "pages": [1], "headings": ["Setup", "SMTP"],
           "text": "EVIDENCE_ONLY_MARKER", "retrievers": ["dense"], "rerank_score": 6.0}
    monkeypatch.setattr(api, "options", lambda: {"documents": [hit]})

    class Retriever:
        def search(self, *args, **kwargs):
            return [hit]

    class Reranker:
        def rerank(self, *args, **kwargs):
            return [hit]

    # Exercise the production generator factory, without loading retrieval models.
    monkeypatch.setattr(api, "pipeline", lru_cache(maxsize=1)(
        lambda: (Retriever(), Reranker(), api.conversation_generator())))
    requests = []

    def respond(request):
        assert str(request.url) == configuration["generation"]["url"]
        payload = json.loads(request.content)
        assert payload["model"] == configuration["generation"]["model"]
        requests.append(payload)
        is_summary = payload["messages"][0]["content"].startswith("Update the conversation summary")
        content = ("The user's Nextcloud SMTP server is alpha.example." if is_summary
                   else "Open the settings [1].")
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}]})

    actual_client = local.ChatOpenAI
    monkeypatch.setattr(local, "ChatOpenAI", lambda **kwargs: actual_client(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    loaded = {"value": True}
    monkeypatch.setattr(api, "is_loaded", lambda model: loaded["value"])
    monkeypatch.setattr(api, "load_model", lambda *args: loaded.update(value=True))
    monkeypatch.setattr(api, "unload_model", lambda *args: loaded.update(value=False))
    first_question = "Where is SMTP in Nextcloud 36? My server is alpha.example."
    second_question = "Which Nextcloud 36 SMTP setting comes next?"
    other_id = str(uuid4())

    with TestClient(app) as client:
        first = client.post("/api/chat", json={"question": first_question})
        assert first.status_code == 200, first.text
        conversation_id = first.json()["conversation_id"]
        generator = api.conversation_generator()
        assert generator.summary_llm is not None
        assert [m.content for m in generator.get_history(conversation_id)] == [
            first_question, "Open the settings [1]."]
        assert (tmp_path / "memory/chat.sqlite").is_file()
        assert client.post("/api/chat", json={"question": second_question,
            "conversation_id": conversation_id}).status_code == 200
        prompt = requests[-1]["messages"]
        assert any(m["content"] == first_question for m in prompt)
        assert any(m["content"] == "Open the settings [1]." for m in prompt)
        assert len(generator.get_history(conversation_id)) == 4

        assert client.post("/api/chat", json={"question": "Where is Nextcloud 36 SMTP?",
            "conversation_id": other_id}).status_code == 200
        assert "alpha.example" not in json.dumps(requests[-1])
        assert len(generator.get_history(other_id)) == 2
        # Evidence is run-scoped, absent even from intermediate checkpoints.
        checkpoints = list(generator._saver.list({"configurable": {"thread_id": conversation_id}}))
        assert checkpoints and "EVIDENCE_ONLY_MARKER" not in str(checkpoints)
        assert client.post("/api/model/unload").json()["loaded"] is False
        assert len(generator.get_history(conversation_id)) == 4
        assert client.post("/api/model/load").json()["loaded"] is True
        assert len(generator.get_history(conversation_id)) == 4
        assert client.post("/api/chat", json={"question": second_question,
            "conversation_id": "invalid"}).status_code == 422

    assert generator._saver_cm is None
    assert generator.llm.root_client.is_closed()
    assert api.conversation_generator.cache_info().currsize == 0
    # A fresh app lifecycle reopens SQLite; no client-supplied history is needed.
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"question": second_question,
            "conversation_id": conversation_id})
        assert response.status_code == 200, response.text
        reopened = api.conversation_generator()
        assert reopened is not generator
        assert len(reopened.get_history(conversation_id)) == 6
        assert len(reopened.get_history(other_id)) == 2
        assert "alpha.example" in json.dumps(requests[-1])
        # Continue to the ninth question: summary is a separate state field,
        # not an extra message or part of the exchange count.
        for _ in range(6):
            response = client.post("/api/chat", json={"question": second_question,
                "conversation_id": conversation_id})
            assert response.status_code == 200, response.text
        state = reopened.graph.get_state({"configurable": {"thread_id": conversation_id}}).values
        assert state["summary"] == "The user's Nextcloud SMTP server is alpha.example."
        assert len(state["messages"]) == 8
        assert all(m.content != state["summary"] for m in state["messages"])
        summary_calls = [r for r in requests if r["messages"][0]["content"].startswith(
            "Update the conversation summary")]
        assert len(summary_calls) == 1
        assert state["summary"] in requests[-1]["messages"][0]["content"]

    with TestClient(app):
        restored = api.conversation_generator()
        state = restored.graph.get_state({"configurable": {"thread_id": conversation_id}}).values
        assert "alpha.example" in state["summary"]
        assert len(state["messages"]) == 8


def test_graph_trimming_keeps_stored_messages_and_model_errors_are_translated(monkeypatch, tmp_path) -> None:
    local = importlib.import_module("manuora.generation.local")
    prompts = []
    fail = False

    def respond(request):
        if fail:
            raise httpx.ConnectError("Model runner unavailable", request=request)
        prompts.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "fake", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Reply " * 300}}]})

    actual_client = local.ChatOpenAI
    monkeypatch.setattr(local, "ChatOpenAI", lambda **kwargs: actual_client(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    with local.ManuoraGraphGenerator("fake", "http://localhost/v1/chat/completions",
            db_path=str(tmp_path / "trim.sqlite"), history_token_budget=80) as generator:
        generator.generate("OLDER_QUESTION", [], thread_id="trim")
        generator.generate("CURRENT_QUESTION", [], thread_id="trim")
        assert "OLDER_QUESTION" not in json.dumps(prompts[-1])
        assert len(generator.get_history("trim")) == 4
        assert generator.get_history("trim")[0].content == "OLDER_QUESTION"
        fail = True
        with pytest.raises(OSError, match="Local generation request failed"):
            generator.generate("Offline question", [], thread_id="offline")
        assert not any(m.type == "ai" for m in generator.get_history("offline"))


def test_token_pressure_batches_summary_and_preserves_state_on_failure(memory_model_transport, tmp_path):
    local, payloads, replies = memory_model_transport
    # Simulate a persisted backlog from the old, unbounded history implementation.
    backlog = [local.HumanMessage(content="Older settings " * 650 + " TAIL_MARKER"),
               local.AIMessage(content="Acknowledged.")]
    with local.ManuoraGraphGenerator("fake", "http://localhost/v1/chat/completions",
            db_path=str(tmp_path / "pressure.sqlite"), summarize=True,
            context_window_tokens=1024, max_output_tokens=100,
            summary_max_output_tokens=90, safety_margin_tokens=64,
            history_token_budget=200, system_prompt="Use the current manual evidence.") as generator:
        generator.graph.update_state({"configurable": {"thread_id": "pressure"}}, {"messages": backlog})
        generator.generate("Which SMTP port?", [], thread_id="pressure")
        assert generator.last_prompt_stats["summary_trigger"] == "tokens"
        assert generator.last_prompt_stats["summary_calls"] > 1
        assert generator.last_prompt_stats["trimmed_messages"] == 0
        summary_requests = payloads[:-1]
        assert "TAIL_MARKER" in str(summary_requests)
        assert all("Older conversation:" in r["messages"][0]["content"] for r in summary_requests)
        state = generator.graph.get_state({"configurable": {"thread_id": "pressure"}}).values
        assert state["summary"] == replies["summary"]
        assert [m.content for m in state["messages"]] == ["Which SMTP port?", replies["answer"]]
        assert_requests_fit(generator, payloads)

        generator.graph.update_state({"configurable": {"thread_id": "failure"}}, {"messages": backlog})
        replies["summary"] = ""
        with pytest.raises(ValueError, match="empty summary"):
            generator.generate("Continue", [], thread_id="failure")
        assert generator.get_history("failure")[0].content == backlog[0].content
        assert len(generator.get_history("failure")) == 3  # Old exchange + current question.


def test_short_exchanges_do_not_trigger_summary_when_count_trigger_disabled(memory_model_transport, tmp_path):
    local, payloads, _ = memory_model_transport
    with local.ManuoraGraphGenerator("fake", "http://localhost/v1/chat/completions",
            db_path=str(tmp_path / "short.sqlite"), summarize=True,
            context_window_tokens=4096, history_token_budget=2500,
            system_prompt="Use the current evidence.") as generator:
        for number in range(12):
            generator.generate(f"Short question {number}", [], thread_id="short")
            assert generator.last_prompt_stats["summary_calls"] == 0
        assert len(payloads) == 12 and len(generator.get_history("short")) == 24
        assert_requests_fit(generator, payloads)


@pytest.mark.parametrize("context_tokens", [300, 900])
def test_full_prompt_budget_packs_evidence_and_rejects_oversized_question(
        memory_model_transport, tmp_path, context_tokens):
    local, payloads, _ = memory_model_transport
    evidence = [{"document_id": f"doc-{i}", "pages": [i], "headings": ["Configuration"],
                 "text": f"SOURCE_{i} " + "manual content " * 100} for i in range(1, 5)]
    with local.ManuoraGraphGenerator("fake", "http://localhost/v1/chat/completions",
            db_path=str(tmp_path / "evidence.sqlite"), summarize=True,
            context_window_tokens=context_tokens, max_output_tokens=80,
            safety_margin_tokens=32, summary_max_output_tokens=64,
            system_prompt="Use current evidence and cite sources.") as generator:
        generator.generate("Where is the setting?", evidence, thread_id="evidence")
        used = generator.last_prompt_stats["evidence_count"]
        assert 1 <= used < 4
        assert "SOURCE_1" in payloads[-1]["messages"][-1]["content"]
        assert f"doc-{used + 1}" not in payloads[-1]["messages"][-1]["content"]
        assert_requests_fit(generator, payloads)
        before = len(payloads)
        with pytest.raises(ValueError, match="context budget"):
            generator.generate("Oversized question " * 1000, evidence, thread_id="too-large")
        assert len(payloads) == before


def test_draft_commit_rejects_stale_and_cross_session_updates(memory_model_transport, tmp_path):
    local, _, replies=memory_model_transport
    with local.ManuoraGraphGenerator('fake','http://localhost/v1/chat/completions',db_path=str(tmp_path/'draft.sqlite')) as gen:
        draft=gen.draft_answer('A question?',[],thread_id='one')
        assert gen.get_history('one')==[]
        with pytest.raises(ValueError,match='mismatched'):
            gen.record_displayed_reply('A question?',draft['answer'],thread_id='other',draft=draft)
        gen.record_displayed_reply('A different question?','Displayed answer.',thread_id='one',kind='chat')
        with pytest.raises(ValueError,match='Stale'):
            gen.record_displayed_reply('A question?',draft['answer'],thread_id='one',draft=draft)
        assert len(gen.get_history('one'))==2 and gen.get_history('other')==[]


def test_library_history_trimming_matches_whole_exchange_policy(memory_model_transport, tmp_path):
    """Differential check of both budgets, summary pressure and message metadata."""
    import random
    local, _, _ = memory_model_transport
    randomizer = random.Random(42)
    with local.ManuoraGraphGenerator('fake', 'http://localhost/v1/chat/completions',
            db_path=str(tmp_path/'library.sqlite')) as gen:
        for n in range(240):
            history = []
            for i in range(randomizer.randrange(0, 9)):
                history.extend([local.HumanMessage(content='Question ' * randomizer.randrange(1, 100), id=f'{i}:u'),
                    local.AIMessage(content='Answer [2] ' * randomizer.randrange(1, 100), id=f'{i}:a',
                                    additional_kwargs={'sources':[{'citation':2,'chunk_id':'original'}]})])
            # A persisted unanswered user message is a supported legacy case.
            if n % 4 == 0:
                history.append(local.HumanMessage(content='Unanswered question', id='tail'))
            before = [m.model_dump() for m in history]
            summary = 'Prior topic. ' * randomizer.randrange(0, 160)
            evidence = 'Evidence body. ' * randomizer.randrange(0, 250)
            gen.history_token_budget = randomizer.choice([0, 50, 200, 1400])
            groups = gen._history_groups(history)
            expected = history
            while groups and not gen._fits('Current question?', evidence, summary, expected):
                groups.pop(0)
                expected = [m for group in groups for m in group]
            actual = gen._trim_history('Current question?', evidence, summary, history)
            assert [m.model_dump() for m in actual] == [m.model_dump() for m in expected], n
            assert [m.model_dump() for m in history] == before
