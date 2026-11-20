"""Experiment R: rules v2, the shared routing decision and its checks, and the routing tier."""
import pytest

from manuora.generation.context import ContextRouter, prepare_context, remember_reply
from manuora.generation.router import (RouteDecision, checked, from_context, normalize_version, parse_decision)

DOCS = ([{"product_id": "nextcloud", "product_version": v, "document_id": f"nc{v}"} for v in ("12", "35", "36")]
        + [{"product_id": "kicad", "product_version": v, "document_id": f"kc{v}"} for v in ("6.0", "8", "9")]
        + [{"product_id": "qgis", "product_version": v, "document_id": f"qg{v}"} for v in ("3.10", "3.22", "3.44")])


def route(question, rules="v2", ui=None, memory=None, last=None):
    context = prepare_context(question, ui or {}, DOCS, memory or {}, thread_id="t", rules=rules)
    context["last_displayed_reply"], context["ui"] = last, ui or {}
    return checked(from_context(context, ContextRouter().decide(context)), context, DOCS), context


@pytest.mark.parametrize("question,action", [
    # A pronoun whose subject is in the same question (v1 asked for clarification)
    ("Can Nextcloud scan uploaded files for viruses? How do I set that up?", "direct"),
    ("We started on SQLite and now it is slow. How do I move Nextcloud to MariaDB?", "direct"),
    ("How do I add components, connect them, and prepare them for the PCB in KiCad 9?", "direct"),
    ("Why might an older client stop connecting after I enable 2FA, and what should I use instead?", "direct"),
    ("In Nextcloud 36 the desktop manual has a Search subsection; does that prove it was new?", "direct"),
    # Nothing in the question to refer to: still clarify
    ("How do I undo it?", "clarify"),
    ("Why did it fail?", "clarify"),
    ("What does that button do?", "clarify"),
    ("Can I do the same for the other ones?", "clarify"),
    ("What should I use instead?", "clarify"),
])
def test_v2_resolves_pronouns_inside_a_self_contained_question(question, action):
    assert route(question)[0].action == action


def test_v1_is_unchanged_baseline():
    assert route("Can Nextcloud scan uploaded files for viruses? How do I set that up?", rules="v1")[0].action == "clarify"
    assert route("How do I export gerber files in KiCad 6?", rules="v1")[0].action == "clarify"


def test_v2_versions_normalize_and_comparisons_get_every_version():
    assert route("How do I export gerber files in KiCad 6?")[0].versions == ["6.0"]
    assert route("In QGIS v3.22, how do I label only selected features?")[0].versions == ["3.22"]
    for question in ("What changed between Nextcloud 35 and 36 for account setup?",
                     "In Nextcloud 35 I clicked Edit Ignored Files. Where is it documented in 36?",
                     "From Nextcloud 35 to 36, did file locking change?"):
        decision, context = route(question)
        assert decision.action == "direct" and sorted(decision.versions) == ["35", "36"], question
        assert context["scope"]["product_version"] is None
    # Counts are not versions; unknown explicit versions still ask
    assert route("In Nextcloud 36 I have 12 users. How do I add a group?")[0].versions == ["36"]
    assert route("How do I add a user in Nextcloud 37?")[0].action == "clarify"
    # A comparison may include the page's version; a different single version still conflicts
    assert route("Compare Nextcloud 35 and 36 dashboards.", ui={"product_id": "nextcloud", "product_version": "36"})[0].action == "direct"
    assert route("In Nextcloud 12, which side is the restore icon on?",
                 ui={"product_id": "nextcloud", "product_version": "36"})[0].action == "clarify"


def remembered(*questions, rules="v2"):
    memory = {}
    for n, q in enumerate(questions):
        context = prepare_context(q, {}, DOCS, memory, thread_id="t", rules=rules)
        memory = remember_reply(memory, context, f"answer {n}", [], turn_id=f"t{n}", kind="answer")
    return memory


def test_v2_follow_ups_corrections_returns_and_edits():
    memory = remembered("How do I enable snapping while digitizing in QGIS 3.22?")
    decision, context = route("Sorry, I meant QGIS 3.10.", memory=memory)
    assert decision.action == "rewrite" and decision.versions == ["3.10"]
    assert route("And on macOS?", memory=memory)[0].action == "rewrite"
    assert route("What about version 3.44?", memory=memory)[0].versions == ["3.44"]
    # "back to" only at the start counts as returning to a task
    rollback = "In Nextcloud 12, which side is the restore icon on when rolling a file back to an older version?"
    assert route(rollback)[0].action == "direct"
    memory = remembered("In Nextcloud 36, how do I add a user to a group?", "Switch to KiCad 8: how do I add a mounting hole?")
    back, _ = route("Back to the Nextcloud question: where are group settings?", memory=memory)
    assert back.action == "rewrite" and back.product == "nextcloud"
    last = {"content": "answer", "message_id": "m", "sources": [], "truncated": False}
    assert route("Shorter please.", memory=memory, last=last)[0].action == "edit"
    assert route("Can you put that as a numbered list?", memory=memory, last=last)[0].action == "edit"
    assert route("Summarize your answer in two sentences.")[0].action == "clarify"  # nothing to edit


def test_v2_greetings_and_thanks_are_chat_but_questions_are_not():
    from manuora.api.routing import early_response
    products = {"nextcloud"}
    assert early_response("hi there!", "Manuora", products, rules="v2")
    assert early_response("thanks, that helped a lot", "Manuora", products, rules="v2")
    assert early_response("thanks, and how do I share it?", "Manuora", products, rules="v2") is None


def test_normalize_version_and_shared_checks():
    assert normalize_version("kicad", "6", DOCS) == ["6.0"]
    assert normalize_version("qgis", "v3.22", DOCS) == ["3.22"]
    assert normalize_version("qgis", "3", DOCS) == ["3.10", "3.22", "3.44"]
    assert normalize_version("nextcloud", "37", DOCS) == []
    context = {"question": "q", "ui": {}, "last_displayed_reply": None}
    assert checked(RouteDecision("direct", "kicad", ["6"]), context, DOCS).versions == ["6.0"]
    assert checked(RouteDecision("direct", "nextcloud", ["37"]), context, DOCS).action == "clarify"
    assert checked(RouteDecision("direct", "blender", []), context, DOCS).action == "clarify"
    assert checked(RouteDecision("direct", None, ["36"]), context, DOCS).action == "clarify"
    assert checked(RouteDecision("edit"), context, DOCS).action == "clarify"
    ui = {**context, "ui": {"product_id": "qgis"}}
    assert checked(RouteDecision("direct", "kicad", []), ui, DOCS).action == "clarify"
    assert checked(RouteDecision("direct", "kicad", []), context, DOCS).needs == ["q"]


def test_parse_decision_accepts_only_the_schema():
    d = parse_decision('<think></think>{"action": "direct", "product": "kicad", "versions": ["9"], '
                       '"needs": ["create a symbol", "assign a footprint"]}')
    assert d.action == "direct" and d.versions == ["9"] and len(d.needs) == 2
    assert parse_decision("not json") is None
    assert parse_decision('{"action": "search"}') is None
    assert parse_decision('{"action": "direct", "versions": "9"}') is None


def test_routing_tier_replays_history_and_scores_the_decision():
    from manuora.evaluation import bench
    from manuora.evaluation.run import aggregate, route_case

    base = dict(category="conversation", split="dev", status="checked", product_id="nextcloud", versions=["36"],
                expect="answer", gold=None, tags=["routing_only"], _private=False)
    follow = dict(base, id="t1", question="How do I restore one?",
                  history=[{"role": "user", "content": "In Nextcloud 36 I'm looking at older versions of a file."}])
    no_history = dict(base, id="t2", question="How do I restore one?", history=[], product_id="", versions=[],
                      expect="clarify", ui={"product_id": None, "product_version": None})
    compare = dict(base, id="t3", category="version", question="What changed between Nextcloud 35 and 36?",
                   history=[], versions=["35", "36"], tags=[])
    assert bench.expected_route(follow)["actions"] == {"rewrite"}
    assert bench.ui_of(compare) == {"product_id": "nextcloud", "product_version": None}
    records = [route_case(c, DOCS, ContextRouter(), rules) for c in (follow, no_history, compare) for rules in ("v1", "v2")]
    by = {(r["id"], n % 2): r for n, r in enumerate(records)}
    assert not by[("t1", 0)]["route_ok"] and by[("t1", 1)]["route_ok"]   # "restore one": v1 misses the follow-up
    assert by[("t2", 0)]["wrong_direct"] and by[("t2", 1)]["action_ok"]   # nothing to refer to: v2 clarifies
    # comparison: v1 silently searches only the first version, v2 searches both
    assert by[("t3", 0)]["versions"] == ["35"] and not by[("t3", 0)]["scope_ok"] and by[("t3", 1)]["route_ok"]
    row = aggregate([by[("t3", 0)], by[("t3", 1)]])["version"]
    assert row["route_ok"] == 1 and row["scope_ok"] == 1 and "p50_ms" in row and "p50_seconds" not in row


def test_routing_only_cases_stay_out_of_retrieval_tiers(monkeypatch):
    from manuora.evaluation import bench, run
    cases = [dict(id="a", category="single", tags=[]), dict(id="b", category="conversation", tags=["routing_only"]),
             dict(id="c", category="single", tags=["routing_only"])]
    monkeypatch.setattr(bench, "load", lambda *a, **k: cases)
    assert [c["id"] for c in run.select_cases("routing", "dev", True, {}, None)[0]] == ["a", "b", "c"]
    assert [c["id"] for c in run.select_cases("retrieval", "dev", True, {}, None)[0]] == ["a"]


def test_search_versions_interleaves_one_search_per_version():
    from manuora.retrieval.hybrid import HybridRetriever
    retriever = object.__new__(HybridRetriever)
    calls = []

    def fake_search(query, *, product_id, product_version, document_id, candidates):
        calls.append(product_version)
        return [{"chunk_id": f"{product_version}:{i}"} for i in range(3)] + [{"chunk_id": "shared"}]

    retriever.search = fake_search
    hits = retriever.search_versions("q", product_id="nextcloud", versions=["35", "36", "35"], candidates=5)
    assert calls == ["35", "36"]
    assert [h["chunk_id"] for h in hits] == ["35:0", "36:0", "35:1", "36:1", "35:2", "36:2", "shared"]


class _Fixed:
    def __init__(self, action):
        self.action, self.method, self.calls = action, f"fixed_{action}", 0

    def route(self, context):
        self.calls += 1
        return RouteDecision(self.action, method=self.method)


def _llm(fallback=None, generate=None):
    from manuora.generation.router import LLMRouter
    router = LLMRouter(DOCS, backend="llamacpp", url="http://127.0.0.1:9")
    router.fallback = fallback
    if generate:
        router.generate = generate
    return router


def test_llm_router_falls_back_when_its_server_is_unreachable():
    def down(text):
        raise ConnectionRefusedError("no server")
    classifier = _Fixed("direct")
    router = _llm(classifier, down)
    decision = router.route({"question": "how do I enable 2FA in nextcloud 35", "earlier": [], "ui": {}})
    assert decision.action == "direct" and decision.method.endswith("_fallback")
    assert classifier.calls == 1 and router.unreachable == 1 and router.fallbacks == 1


def test_llm_router_falls_back_on_unusable_json():
    classifier = _Fixed("chat")
    router = _llm(classifier, lambda text: "not json")
    assert router.route({"question": "hi", "earlier": [], "ui": {}}).action == "chat"
    assert router.unreachable == 0 and router.fallbacks == 1


def test_hybrid_routes_first_messages_to_the_classifier_and_followups_to_the_llm():
    from manuora.generation.router import HybridRouter
    first, followup = _Fixed("direct"), _Fixed("rewrite")
    hybrid = HybridRouter(first, followup)
    assert hybrid.route({"question": "how do I enable 2FA", "earlier": []}).action == "direct"
    assert hybrid.route({"question": "and for 36?", "earlier": ["how do I enable 2FA in nextcloud 35"]}).action == "rewrite"
    assert (first.calls, followup.calls) == (1, 1)


def test_latest_means_the_newest_version_in_the_library():
    from manuora.api.routing import early_response
    from manuora.generation.router import asks_latest, newest_version

    assert newest_version("qgis", DOCS) == "3.44" and newest_version("kicad", DOCS) == "9"
    # Rules: "latest" in a comparison or a question about the newest product resolves to the newest version.
    decision, context = route("What changed in the server requirements between Nextcloud 12 and the latest version?")
    assert decision.action == "direct" and decision.versions == ["12", "36"]
    decision, _ = route("How do I write a processing script in the newest QGIS?")
    assert decision.versions == ["3.44"]
    # Any router: the shared check adds the newest version when the message asks about the latest.
    context = {"question": "is it any different in the current version?"}
    proposal = RouteDecision("rewrite", product="kicad", versions=["8"], method="llm")
    assert checked(proposal, context, DOCS).versions == ["8", "9"]
    # Not "latest" in another sense, and not the user's own current version.
    assert not asks_latest("How do I restore the latest backup in Nextcloud?", "nextcloud")
    assert not asks_latest("My current version is 35, how do I upgrade?", "nextcloud")
    # Asking what the latest version is gets the library's newest, said plainly; asking about it goes on.
    reply = early_response("What's the latest version of Nextcloud?", "Manuora", {"nextcloud"}, newest={"nextcloud": "36"})
    assert reply.startswith("The newest Nextcloud manual in this library is version 36")
    assert early_response("what changed between nextcloud 12 and the latest version", "Manuora", {"nextcloud"}) is None
    assert early_response("is it any different in the current version?", "Manuora", {"kicad"}) is None


def test_several_products_are_searched_not_clarified():
    # Rules v2: both products, no versions, no conflict (round 3 convention, 2026-11-20).
    decision, context = route("Can QGIS read the files that Nextcloud stores in version 36?")
    assert (decision.action, decision.product, decision.products, decision.versions) == ("direct", None, ["qgis", "nextcloud"], [])
    # A page selection that is neither product still conflicts.
    decision, _ = route("Can QGIS read the files that Nextcloud stores?", ui={"product_id": "kicad", "product_version": "9"})
    assert decision.action == "clarify"
    # A router may answer a list; the shared check keeps known products only.
    proposal = parse_decision('{"action": "direct", "product": ["qgis", "nextcloud"], "versions": ["3.22"], "needs": []}')
    assert (proposal.product, proposal.products) == (None, ["qgis", "nextcloud"])
    final = checked(proposal, {"question": "q"}, DOCS)
    assert (final.action, final.products, final.versions, final.needs) == ("direct", ["qgis", "nextcloud"], [], ["q"])
    assert checked(parse_decision('{"action": "direct", "product": ["qgis", "blender"]}'), {"question": "q"}, DOCS).action == "clarify"
    # A one-item list is just that product.
    single = checked(parse_decision('{"action": "direct", "product": ["kicad"], "versions": ["9"]}'), {"question": "q"}, DOCS)
    assert (single.product, single.products, single.versions) == ("kicad", [], ["9"])


def test_shared_check_keeps_versions_the_message_names_and_a_single_version():
    docs = DOCS + [{"product_id": "freecad", "product_version": "0.16", "document_id": "fc"}]
    # A router that drops a version the user wrote gets it back.
    proposal = RouteDecision("direct", product="kicad", versions=[], method="llm")
    assert checked(proposal, {"question": "How do I route tracks in KiCad 8?"}, docs).versions == ["8"]
    proposal = RouteDecision("direct", product="nextcloud", versions=["35"], method="llm")
    assert checked(proposal, {"question": "What changed between Nextcloud 35 and 36?"}, docs).versions == ["35", "36"]
    # A product with one version in the library: that version.
    proposal = RouteDecision("direct", product="freecad", versions=[], method="llm")
    assert checked(proposal, {"question": "What is the FEM workbench for?"}, docs).versions == ["0.16"]
    # Several versions and none named: still none (search across versions).
    proposal = RouteDecision("direct", product="qgis", versions=[], method="llm")
    assert checked(proposal, {"question": "How do I add a layer?"}, docs).versions == []


def test_app_turn_uses_the_shared_router_and_its_checks():
    from manuora.generation.local import CLARIFY_DEFAULT, ManuoraGraphGenerator

    class Stub:
        def __init__(self, decision):
            self.decision = decision

        def route(self, context):
            return self.decision

    def turn(decision, question, **context):
        generator = ManuoraGraphGenerator.__new__(ManuoraGraphGenerator)
        generator._router, generator._memory_lock = Stub(decision), __import__("threading").RLock()
        context = {"question": question, "scope": {"document_id": None}, **context}
        return generator._shared_route(context, DOCS), context

    (action, query), context = turn(RouteDecision("rewrite", product="kicad", versions=["9"], query="How in KiCad 9?",
                                                  method="llm_llamacpp"), "and in 9?")
    assert (action["action"], query, context["scope"]["product_version"]) == ("contextual_rewrite", "How in KiCad 9?", "9")
    (action, _), context = turn(RouteDecision("direct", products=["qgis", "nextcloud"], method="llm_llamacpp"),
                                "Can QGIS open Nextcloud files?")
    assert context["scope"]["products"] == ["qgis", "nextcloud"] and context["scope"]["product_id"] is None
    (action, _), _ = turn(RouteDecision("clarify", reason="router model", method="llm_llamacpp"), "how do I turn that off")
    assert action["action"] == "clarify" and action["reason"] == CLARIFY_DEFAULT
    (action, _), _ = turn(RouteDecision("edit", method="llm_llamacpp"), "shorter", last_displayed_reply=None)
    assert action["reason"] == "There is no previous displayed answer to edit."
    (action, _), _ = turn(RouteDecision("chat", method="llm_llamacpp"), "you are slow")
    assert action["action"] == "chat"


def test_a_checkout_without_trained_models_still_routes(tmp_path):
    from manuora.generation.router import LLMRouter, build_router

    llm = build_router({"context_router_method": "llm", "context_router_llm": {
        "backend": "llamacpp", "url": "http://127.0.0.1:9", "prompt": "v2", "timeout": 0.5,
        "fallback_classifier": "data/models/missing.joblib"}}, DOCS, tmp_path)
    assert isinstance(llm, LLMRouter) and llm.fallback is None and llm.timeout == 0.5
    context = prepare_context("How do I enable 2FA in Nextcloud 36?", {}, DOCS, {}, thread_id="t", rules="v2")
    decision = llm.route(context)  # nothing listens on port 9: the rules route instead
    assert decision.action == "direct" and decision.method.endswith("_fallback") and llm.unreachable == 1
    assert isinstance(build_router({"context_router_method": "classifier",
                                    "context_router_model_path": "data/models/missing.joblib"}, DOCS, tmp_path),
                      ContextRouter)
