"""Shared fixtures: a fake OpenAI-compatible model transport (no network, no model)."""
import importlib
import json
import httpx
import pytest


@pytest.fixture
def memory_model_transport(monkeypatch):
    """Capture actual serialized requests without loading a model or using the network."""
    local = importlib.import_module("manuora.generation.local")
    payloads = []
    replies = {"summary": "The server is alpha.example. SMTP setup is unresolved.",
               "answer": "Use the SMTP setting [1].",
               "rewrite":"Where is the SMTP setting in Nextcloud 36?",
               "edit":"Use SMTP [1]."}

    def respond(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        is_summary = payload["messages"][0]["content"].startswith("Update the conversation summary")
        kind=('summary' if is_summary else 'rewrite' if payload['messages'][0]['content'].startswith('Rewrite the latest')
              else 'edit' if payload['messages'][0]['content'].startswith('Rewrite the supplied previous answer') else 'answer')
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "fake", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": replies[kind]}}]})

    actual_client = local.ChatOpenAI
    monkeypatch.setattr(local, "ChatOpenAI", lambda **kwargs: actual_client(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    return local, payloads, replies


def assert_requests_fit(generator, payloads):
    for payload in payloads:
        output = payload.get("max_completion_tokens", payload.get("max_tokens"))
        assert isinstance(output, int)
        assert generator.count_tokens(payload["messages"]) + output + \
            generator.safety_margin_tokens <= generator.context_window_tokens
