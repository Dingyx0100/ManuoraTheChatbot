"""One routing decision for every router (rules, classifiers, small fine-tuned LLMs).

A router reads the question with its context proposal (generation/context.py: scope, task history, reference
signals) and says what the question needs:

    action    direct (search the question) | rewrite (search a standalone rewrite) | clarify | edit | chat
    product   product id, or None
    versions  the versions to search; several for a comparison
    needs     separate things to find (one for most questions, one per step for a multi-step task)
    query     the standalone question, when action is rewrite and the router wrote one

Code checks every proposal the same way whichever router made it (see `checked`): unknown products and
versions, a scope that conflicts with the page selection, and an edit with nothing to edit become clarify.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

ACTIONS = ("direct", "rewrite", "clarify", "edit", "chat")
# ContextRouter's action names -> decision actions.
FROM_CONTEXT_ROUTER = {"direct_retrieval": "direct", "contextual_rewrite": "rewrite", "clarify": "clarify",
                       "edit_previous_answer": "edit"}


@dataclass
class RouteDecision:
    action: str
    product: str | None = None
    versions: list[str] = field(default_factory=list)
    # Several products (round 3 convention, 2026-11-20): one search per product, answered for each; product is
    # then None and versions are [].
    products: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    query: str | None = None
    reason: str = ""
    method: str = "rules"
    confidence: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def scope_versions(scope: dict) -> list[str]:
    """Versions a context scope asks for: the comparison list, else the single version."""
    if scope.get("versions"):
        return [str(v) for v in scope["versions"]]
    return [str(scope["product_version"])] if scope.get("product_version") else []


def from_context(context: dict, decided: dict) -> RouteDecision:
    """A ContextRouter decision plus its context proposal as a RouteDecision."""
    scope = context.get("scope") or {}
    action = FROM_CONTEXT_ROUTER[decided["action"]]
    return RouteDecision(action=action, product=scope.get("product_id"), versions=scope_versions(scope),
                         products=list(scope.get("products") or []), needs=[context["question"]], reason=decided.get("reason", ""),
                         method=decided.get("method", "rules"), confidence=decided.get("confidence"))


def chat_decision(reason: str = "early reply") -> RouteDecision:
    return RouteDecision(action="chat", reason=reason, method="early_rules")


def known_versions(documents: list[dict]) -> dict[str, list[str]]:
    out: dict[str, set] = {}
    for d in documents:
        if d.get("product_version"):
            out.setdefault(d["product_id"], set()).add(str(d["product_version"]))
    return {p: sorted(v) for p, v in out.items()}


def version_key(version: str) -> list:
    return [(0, int(x)) if x.isdigit() else (1, x) for x in re.split(r"[.]", str(version))]


def newest_version(product: str, documents: list[dict]) -> str | None:
    """The newest version of a product in the library (numeric order: 3.44 > 3.4, 10 > 9)."""
    stored = known_versions(documents).get(product, [])
    return max(stored, key=version_key) if stored else None


# "latest version", "newest kicad release", "the current version"; not "my current version" (that names theirs).
LATEST = re.compile(r"(?<!\bmy )(?<!\bour )\b(?:latest|newest|most recent|current)\s+(?:\w+\s+)?(?:version|release)s?\b",
                    re.I)


def asks_latest(question: str, product: str | None = None) -> bool:
    """The message asks about the newest version, which means the newest in the library (user decision 2026-11-20)."""
    if LATEST.search(question):
        return True
    return bool(product) and bool(re.search(rf"\b(?:latest|newest|most recent)\s+{re.escape(product)}\b", question, re.I))


def latest_note(product: str, version: str) -> str:
    """The line an answer starts with when "latest" was read as the newest version in the library."""
    from manuora.api.grounding import PRODUCT_NAMES

    return f"(Using {PRODUCT_NAMES.get(product, product)} {version}, the newest version in this library.)"


def normalize_version(product: str, written: str, documents: list[dict]) -> list[str]:
    """Stored versions a written version refers to: "6" -> ["6.0"], "v3.22" -> ["3.22"], "3" -> every 3.x.

    Exact matches win; otherwise a major-only or ".0"-less form matches the versions it is a prefix of.
    Returns [] when nothing matches.
    """
    stored = known_versions(documents).get(product, [])
    token = written.strip().lower().removeprefix("v")
    if token in stored:
        return [token]
    if token + ".0" in stored:
        return [token + ".0"]
    return [v for v in stored if v.startswith(token + ".")]


def checked(decision: RouteDecision, context: dict, documents: list[dict]) -> RouteDecision:
    """The same post-router checks for every router. A learned router proposes; this decides what runs."""
    products = {d["product_id"] for d in documents}
    if decision.action in ("direct", "rewrite", "edit") and context.get("scope_conflict"):
        # The question names a product or version that disagrees with the page selection (explicit_scope).
        return RouteDecision("clarify", reason=str(context["scope_conflict"]), method=decision.method)
    if decision.action in ("direct", "rewrite") and decision.method.startswith("llm"):
        # A learned router can write a product id the library doesn't have (an invented one, for a manual it
        # never saw). Use the product the rules found in the message or the page instead, else search every
        # product; asking the user would be wrong, since they named nothing unknown.
        fallback = (context.get("scope") or {}).get("product_id") or (context.get("ui") or {}).get("product_id")
        fallback = fallback if fallback in products else None
        if decision.products:
            kept = [p for p in decision.products if p in products]
            if len(kept) != len(decision.products):
                decision.products = kept if len(kept) > 1 else []
                decision.product = kept[0] if len(kept) == 1 else (decision.product or fallback)
                decision.reason = f"{decision.reason}; unknown product dropped".lstrip("; ")
        if decision.product and decision.product not in products:
            decision.product, decision.versions = fallback, []
            decision.reason = f"{decision.reason}; unknown product replaced".lstrip("; ")
        # Likewise a product the message never names, against the page's selection: the selection wins (the rules
        # already report a real conflict, a product named in the message, as scope_conflict above).
        ui_product = (context.get("ui") or {}).get("product_id")
        from manuora.api.grounding import PRODUCT_NAMES
        named = lambda p: re.search(rf"\b({re.escape(p)}|{re.escape(PRODUCT_NAMES.get(p, p))})\b",  # noqa: E731
                                    " ".join([context.get("question", "")] + earlier_messages(context)), re.I)
        if (ui_product in products and decision.product and decision.product != ui_product
                and not decision.products and not named(decision.product)):
            selected = (context.get("ui") or {}).get("product_version")
            decision.product, decision.versions = ui_product, [str(selected)] if selected else []
            decision.reason = f"{decision.reason}; page selection kept".lstrip("; ")
    if decision.action in ("direct", "rewrite") and len(set(decision.products)) > 1:
        unknown = [p for p in decision.products if p not in products]
        if unknown:
            return RouteDecision("clarify", reason=f"Unknown product {unknown[0]!r}.", method=decision.method)
        ui_product = (context.get("ui") or {}).get("product_id")
        if ui_product and ui_product not in decision.products:
            return RouteDecision("clarify", reason="The question and selected product disagree.", method=decision.method)
        decision.products = list(dict.fromkeys(decision.products))
        decision.product, decision.versions = None, []
        decision.needs = decision.needs or [context.get("question", "")]
        return decision
    if decision.action in ("direct", "rewrite"):
        if len(decision.products) == 1 and not decision.product:
            decision.product = decision.products[0]
        decision.products = []
        if decision.product and decision.product not in products:
            return RouteDecision("clarify", reason=f"Unknown product {decision.product!r}.", method=decision.method)
        if decision.product:
            resolved = []
            for version in decision.versions:
                found = normalize_version(decision.product, str(version), documents)
                if not found:
                    return RouteDecision("clarify", reason=f"Version {version} is not in the available manuals.",
                                         method=decision.method)
                resolved += [v for v in found if v not in resolved]
            # Versions the message itself names for this product, which a router that writes its own versions
            # can drop (F3 left out "FreeCAD 0.16"); added after round 3, the same for every router.
            # Rules and classifiers take versions from the rules-v2 extraction already; the v1 baseline stays as it was.
            if decision.method.startswith("llm"):
                from manuora.generation.context import mentioned_versions
                named, _ = mentioned_versions(context.get("question", ""), decision.product, documents)
                resolved += [v for v in named if v not in resolved]
                stored = known_versions(documents).get(decision.product, [])
                if not resolved and len(stored) == 1:  # a product with one version: that is the version searched
                    resolved = list(stored)
            newest = newest_version(decision.product, documents)
            if newest and newest not in resolved and asks_latest(context.get("question", ""), decision.product):
                # "latest" is the newest version in the library, whatever the router proposed for it.
                resolved.append(newest)
                decision.reason = f"{decision.reason}; latest = {newest}".lstrip("; ")
            decision.versions = resolved
        elif decision.versions:
            return RouteDecision("clarify", reason="A version alone does not identify a product.", method=decision.method)
        ui = (context.get("ui") or {})
        if ui.get("product_id") and decision.product and ui["product_id"] != decision.product:
            return RouteDecision("clarify", reason="The question and selected product disagree.", method=decision.method)
        if not decision.needs:
            decision.needs = [context.get("question", "")]
    if decision.action == "edit" and not context.get("last_displayed_reply"):
        return RouteDecision("clarify", reason="There is no previous displayed answer to edit.", method=decision.method)
    return decision


def parse_decision(text: str, method: str = "llm") -> RouteDecision | None:
    """A router model's JSON reply as a decision; None when it is not usable (the caller then falls back)."""
    import json

    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("action") not in ACTIONS:
        return None
    versions = data.get("versions") or []
    needs = data.get("needs") or []
    if not isinstance(versions, list) or not isinstance(needs, list):
        return None
    product = data.get("product") or None
    several = [str(p) for p in product if str(p).strip()] if isinstance(product, list) else []
    if isinstance(product, list):
        product = several[0] if len(several) == 1 else None
    elif product is not None and not isinstance(product, str):
        return None
    return RouteDecision(action=data["action"], product=product, products=several if len(several) > 1 else [],
                         versions=[str(v) for v in versions if str(v).strip()],
                         needs=[str(n).strip() for n in needs if str(n).strip()][:4],
                         query=(str(data["query"]).strip() or None) if data.get("query") else None,
                         reason="router model", method=method)


# --------------------------------------------------------------------------- learned routers (experiment R)
# Every learned router sees the same inputs: the question, the user's earlier messages in this conversation
# (raw, not the task the rules picked), whether an answer is on screen to edit, the page selection, and the
# rules' signals.
def earlier_messages(context: dict) -> list[str]:
    earlier = context.get("earlier")
    if earlier is None:
        earlier = [h["content"] for h in context.get("history") or [] if h.get("role") == "user"]
    return list(earlier)


def features(context: dict) -> dict:
    scope = context.get("scope") or {}
    ui = context.get("ui") or {}
    earlier = earlier_messages(context)
    flags = {"has_earlier": bool(earlier), "has_reply": bool(context.get("last_displayed_reply")),
             "rules_need_history": bool(context.get("needs_history")), "rules_edit": bool(context.get("answer_edit")),
             "rules_clarify": bool(context.get("clarification")), "scope_product": bool(scope.get("product_id")),
             "ui_product": bool(ui.get("product_id")), "several_versions": len(scope_versions(scope)) > 1}
    return {"question": context["question"], "earlier": " | ".join(earlier[-2:]), "flags": flags}


def classifier_text(f: dict) -> str:
    """One string for TF-IDF: the flags become tokens."""
    return (f["question"] + " EARLIER " + f["earlier"] + " " +
            " ".join(f"FLAG_{k}_{int(v)}" for k, v in sorted(f["flags"].items())))


class ClassifierRouter:
    """Predicts the action with a classifier trained on the synthetic train split (scripts/router/train_classifier.py).

    Scope comes from the context proposal (rules v2 extraction); needs is the question itself; a rewrite query
    is left to the app's rewrite step. The artifact is a joblib dict {"method": "tfidf"|"minilm", "model": ...}.
    """

    def __init__(self, model_path: str):
        import joblib

        self.bundle = joblib.load(model_path)
        self.method = self.bundle["method"]
        self._encoder = None

    def vector(self, f: dict):
        import numpy as np

        if self.method == "tfidf":
            return [classifier_text(f)]
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.bundle["encoder"], device="cpu")
        q, e = self._encoder.encode([f["question"], f["earlier"] or "(none)"], normalize_embeddings=True)
        return np.concatenate([q, e, [float(v) for _, v in sorted(f["flags"].items())]])[None, :]

    def route(self, context: dict) -> RouteDecision:
        model = self.bundle["model"]
        probabilities = model.predict_proba(self.vector(features(context)))[0]
        index = int(probabilities.argmax())
        scope = context.get("scope") or {}
        return RouteDecision(action=str(model.classes_[index]), product=scope.get("product_id"),
                             versions=scope_versions(scope), products=list(scope.get("products") or []),
                             needs=[context["question"]],
                             reason=f"{self.method} classifier", method=self.method,
                             confidence=float(probabilities[index]))


ROUTER_INSTRUCTIONS = """Route one message sent to a support chatbot for product manuals. Reply with JSON only:
{"action": "...", "product": "...", "versions": [...], "needs": [...], "query": "..."}
action: direct (search the message as written), rewrite (it continues an earlier message: write `query`, the
message as one self-contained question using only what the user said), clarify (it cannot be understood or
answered as written: it points at something never named, names two different products, or asks to edit an
answer when none is shown), edit (change the form of the answer shown, nothing new to find), chat (greeting,
thanks, questions about the assistant, off-topic).
product: a product id from the list, only if the user named it, the page has it selected, or an earlier
message named it; else null. versions: the versions the user named or selected (several for a comparison);
[] when none. needs: one short search phrase per separate thing to find (one for most questions, one per
step of a multi-step task); [] for clarify, edit and chat. query: only for rewrite, else null."""


# Round 3 (2026-11-20): several products are searched, not clarified; "latest" is resolved by code (checked).
ROUTER_INSTRUCTIONS_V2 = """Route one message sent to a support chatbot for product manuals. Reply with JSON only:
{"action": "...", "product": "...", "versions": [...], "needs": [...], "query": "..."}
action: direct (search the message as written), rewrite (it continues an earlier message: write `query`, the
message as one self-contained question using only what the user said), clarify (it cannot be understood or
answered as written: it points at something never named, or asks to edit an answer when none is shown), edit
(change the form of the answer shown, nothing new to find), chat (greeting, thanks, questions about the
assistant, off-topic).
product: a product id from the list, only if the user named it, the page has it selected, or an earlier
message named it; a list of product ids when the message asks about several products; else null. versions: the
versions the user named or selected (several for a comparison); [] when none or when several products are
named. needs: one short search phrase per separate thing to find (one for most questions, one per step of a
multi-step task); [] for clarify, edit and chat. query: only for rewrite, else null."""
PROMPTS = {"v1": ROUTER_INSTRUCTIONS, "v2": ROUTER_INSTRUCTIONS_V2}


# Short output (arm F-short): the decision only. Needs default to the question, and a rewrite query comes from
# the app's rewrite step, as for the classifier routers. About a quarter of the full output's tokens.
SHORT_INSTRUCTIONS = """Route one message sent to a support chatbot for product manuals. Reply with JSON only:
{"action": "...", "product": "...", "versions": [...]}
action: direct (search the message as written), rewrite (it continues an earlier message), clarify (it cannot
be understood or answered as written: it points at something never named, names two different products, or
asks to edit an answer when none is shown), edit (change the form of the answer shown, nothing new to find),
chat (greeting, thanks, questions about the assistant, off-topic).
product: a product id from the list, only if the user named it, the page has it selected, or an earlier
message named it; else null. versions: the versions the user named or selected (several for a comparison);
[] when none."""


def router_input(context: dict, catalog: dict[str, list[str]]) -> str:
    catalog = context.get("catalog") or catalog  # an evaluation example may offer its own products
    ui = context.get("ui") or {}
    selected = " ".join(filter(None, [ui.get("product_id"), ui.get("product_version")])) or "nothing"
    earlier = earlier_messages(context)
    return "\n".join([
        "Products: " + "; ".join(f"{p} ({', '.join(v)})" for p, v in sorted(catalog.items())),
        f"Page selection: {selected}",
        f"Answer shown: {'yes' if context.get('last_displayed_reply') else 'no'}",
        "Earlier messages: " + (" | ".join(m[:300] for m in earlier[-2:]) or "none"),
        f"Message: {context['question']}"])


def label_json(label: dict, short: bool = False) -> str:
    """The training target for one synthetic label, in the router's output schema."""
    import json

    search = label["action"] in ("direct", "rewrite")
    out = {"action": label["action"], "product": label.get("product") if search else None,
           "versions": (label.get("versions") or []) if search else []}
    if not short:
        out |= {"needs": (label.get("needs") or []) if search else [],
                "query": label.get("query") if label["action"] == "rewrite" else None}
    return json.dumps(out, ensure_ascii=False)


def completion_prompt(text: str, shots=(), short: bool = False, prompt: str = "v1") -> str:
    """The plain-text prompt the small models are tuned and scored on (also used few-shot, untuned)."""
    parts = [SHORT_INSTRUCTIONS if short else PROMPTS[prompt], ""]
    for inp, out in shots:
        parts += ["### Input", inp, "### Output", out, ""]
    parts += ["### Input", text, "### Output", ""]
    return "\n".join(parts)


class LLMRouter:
    """A language model writes the whole decision as JSON; unusable JSON falls back to the rules.

    backend "dmr": the app's chat model through Docker Model Runner (few-shot). backend "transformers": a local
    model directory, optionally with a LoRA adapter, on `device`. backend "llamacpp": a llama-server at `url`
    serving the merged router as GGUF (scripts/router/merge_lora.py, then convert); same prompt as transformers.
    """

    def __init__(self, documents: list[dict], backend: str = "dmr", model_path: str | None = None,
                 adapter_path: str | None = None, shots=(), device: str = "cuda", max_new_tokens: int = 160,
                 short: bool = False, url: str = "http://127.0.0.1:8091", prompt: str = "v1",
                 timeout: float = 5.0):
        from manuora.api.grounding import PRODUCT_NAMES

        self.catalog = known_versions(documents)
        self.names = {}
        for pid in self.catalog:
            self.names[pid.casefold()] = pid
            self.names[PRODUCT_NAMES.get(pid, pid).casefold()] = pid
        self.backend, self.shots, self.device, self.short = backend, list(shots), device, short
        self.max_new_tokens = min(max_new_tokens, 48) if short else max_new_tokens
        self.method = f"llm_{backend}" + ("_short" if short else "")
        self.fallbacks = 0
        self.url = url.rstrip("/")
        self.prompt = prompt
        self.timeout = timeout  # llamacpp: a slow or hung server falls back instead of holding the turn
        if backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)
            if adapter_path:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path).merge_and_unload()
            self.model.eval()

    def generate(self, text: str) -> str:
        if self.backend == "dmr":
            from manuora.evaluation.common import call

            messages = [{"role": "system", "content": PROMPTS[self.prompt] + " /no_think"}]
            for inp, out in self.shots:
                messages += [{"role": "user", "content": inp}, {"role": "assistant", "content": out}]
            messages.append({"role": "user", "content": text})
            return re.sub(r"<think>.*?</think>", "", call(messages, max_tokens=self.max_new_tokens)["text"], flags=re.S)
        if self.backend == "llamacpp":
            import json as _json
            import urllib.request

            payload = {"prompt": completion_prompt(text, self.shots, self.short, self.prompt), "n_predict": self.max_new_tokens,
                       "temperature": 0, "top_k": 1, "stop": ["\n"], "cache_prompt": True}
            request = urllib.request.Request(self.url + "/completion", _json.dumps(payload).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = _json.loads(response.read())["content"]
            return reply.split("\n###")[0].strip().split("\n")[0]
        import torch

        ids = self.tokenizer(completion_prompt(text, self.shots, self.short, self.prompt), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            # Only the first output line is used, so stop there: an untuned model would otherwise run on to
            # max_new_tokens. The kept text is identical either way.
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id, stop_strings=["\n"],
                                      tokenizer=self.tokenizer)
        reply = self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply.split("\n###")[0].strip().split("\n")[0]

    fallback = None  # a router (e.g. ClassifierRouter) used when this one fails; None -> the v1 rules
    unreachable = 0

    def _fall_back(self, context: dict, why: str) -> RouteDecision:
        self.fallbacks += 1
        if self.fallback is not None:
            decision = self.fallback.route(context)
        else:
            from manuora.generation.context import ContextRouter
            decision = from_context(context, ContextRouter("rules").decide(context))
        decision.reason, decision.method = f"{why}; fallback", self.method + "_fallback"
        return decision

    def route(self, context: dict) -> RouteDecision:
        try:
            text = self.generate(router_input(context, self.catalog))
        except OSError:  # router server down or unreachable (URLError is an OSError)
            if self.fallback is None and self.backend != "llamacpp":
                raise
            self.unreachable += 1
            return self._fall_back(context, "router unreachable")
        decision = parse_decision(text, method=self.method)
        if decision is None:
            return self._fall_back(context, "router JSON unusable")
        if decision.product:
            decision.product = self.names.get(decision.product.casefold(), decision.product)
        decision.products = [self.names.get(p.casefold(), p) for p in decision.products]
        return decision


class HybridRouter:
    """`first` for a message with no earlier messages, `followup` otherwise (experiment R, arm H).

    A first message rarely needs the conversation, so a fast classifier routes it; the LLM router only runs when
    earlier messages exist. `followup` should carry `first` as its fallback.
    """

    def __init__(self, first, followup):
        self.first, self.followup = first, followup
        self.method = f"hybrid_{getattr(first, 'method', 'first')}_{getattr(followup, 'method', 'followup')}"

    @property
    def fallbacks(self):
        return getattr(self.followup, "fallbacks", None)

    def route(self, context: dict) -> RouteDecision:
        return (self.followup if earlier_messages(context) else self.first).route(context)


def build_router(conversation: dict, documents: list[dict] | None, root):
    """The router these conversation settings describe (the app and the evaluation runner both use this).

    context_router_method: rules | tfidf (the original direct/contextual TF-IDF) | classifier (experiment R:
    joblib artifact at context_router_model_path) | llm (context_router_llm: {backend, model_path, adapter_path,
    shots_path, url, prompt, fallback_classifier, hybrid}). Relative paths are under `root`.
    """
    from pathlib import Path

    from manuora.generation.context import ContextRouter

    import logging

    root = Path(root)
    method = conversation.get("context_router_method", "rules")
    path = conversation.get("context_router_model_path")
    if method == "classifier":
        if not (root / path).is_file():
            logging.getLogger(__name__).warning("Router model %s is missing; using the rules router.", path)
            return ContextRouter("rules")
        return ClassifierRouter(str(root / path))
    if method == "llm":
        import json as _json

        llm = dict(conversation.get("context_router_llm") or {})
        shots_path = llm.pop("shots_path", None)
        shots = ([tuple(s) for s in (_json.loads(line) for line in (root / shots_path).read_text(encoding="utf-8").splitlines()
                                     if line.strip())] if shots_path else [])
        for key in ("model_path", "adapter_path"):
            if llm.get(key) and not Path(llm[key]).is_absolute() and (root / llm[key]).exists():
                llm[key] = str(root / llm[key])
        fallback_path, hybrid = llm.pop("fallback_classifier", None), llm.pop("hybrid", False)
        router = LLMRouter(documents or [], shots=shots, **llm)
        if fallback_path and not (root / fallback_path).is_file():
            # A fresh checkout has no trained models (data/ is local): fall back to the rules instead.
            logging.getLogger(__name__).warning("Fallback classifier %s is missing; the rules are the fallback.",
                                                fallback_path)
            fallback_path = None
        if fallback_path:
            router.fallback = ClassifierRouter(str(root / fallback_path))
            if hybrid:  # the classifier routes first messages; the LLM only messages with earlier ones
                return HybridRouter(router.fallback, router)
        return router
    return ContextRouter(method, str(root / path) if path else None)


def route_with(router, context: dict) -> RouteDecision:
    """A RouteDecision from either router kind (ContextRouter returns its own action dict)."""
    if hasattr(router, "route"):
        return router.route(context)
    return from_context(context, router.decide(context))
