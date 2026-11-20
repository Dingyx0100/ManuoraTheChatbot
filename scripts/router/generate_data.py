"""Synthetic routing examples for experiment R (router comparison), written by the local chat model.

    python scripts/router/generate_data.py --name pilot_v0_1 --per-scenario 10      # pilot (~10 min)
    python scripts/router/generate_data.py --name train_v0_1 --per-scenario 200     # full set (2-3 h)

Each example is one routing decision. The scenario that produced it fixes the label (action, product,
versions, needs); the model only writes the wording, seeded with a real manual section so the questions use
the manuals' vocabulary. Output: data/router/<name>/examples.jsonl (+ dropped.jsonl, run.json).

Splits, fixed by product and style so the test sets measure what no arm was tuned on:
  test_unseen   KiCad and FreeCAD (never used for training)
  test_style    the "typos" style on training products (never used for training)
  train / val   Nextcloud and QGIS, other styles, 90 / 10 by seed section
Examples too close to any benchmark question (TF-IDF cosine >= 0.8) are dropped, so nothing leaks into the
benchmark's dev or holdout split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from manuora.api.grounding import PRODUCT_NAMES
from manuora.evaluation import bench
from manuora.evaluation.common import call, settings
from manuora.generation.router import normalize_version
from manuora.retrieval.outlines import sections

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/router"
TRAIN_PRODUCTS = ("nextcloud", "qgis")
UNSEEN_PRODUCTS = ("kicad", "freecad")
STYLES = {
    "terse": "Write it the way a busy user types: short, few words, no greeting.",
    "chatty": "Write it conversationally, with a sentence of backstory about what the user is doing.",
    "non_native": "Write it as a non-native English speaker would: understandable, slightly unusual grammar.",
    "multi_sentence": "Write it as two or three sentences: the situation first, then the question.",
    "typos": "Write it quickly with two or three realistic typos and little punctuation (keep product names spelled right).",
}
TRAIN_STYLES = [s for s in STYLES if s != "typos"]
SYSTEM = ("You write realistic messages that users send to a support chatbot for product manuals. You never answer "
          "them. Follow the instructions exactly and reply with one JSON object only. /no_think")

# --phrasing: the test_phrasing split. Same scenarios and labels, but the prompts lose every example phrase
# (the wording the train split copied), the writer persona changes, and examples close to any train question
# are dropped, so the split measures new wording for the same intent.
PHRASING_SYSTEM = ("You play many different real customers of desktop and server software, each with their own "
                   "vocabulary, writing to a help-desk chat. Never reuse stock chatbot phrasing; every message "
                   "should sound like a different person. You never answer them. Reply with one JSON object only. "
                   "/no_think")
PHRASING_NOTE = (" Use your own wording: do not copy any phrasing from these instructions, and avoid stock "
                 "phrases (for example 'what about', 'same there', 'can you explain', 'how do I').")
EXAMPLE_PARENS = re.compile(r"\s*\([^()]*\"[^()]*\)")
EXAMPLE_INLINE = re.compile(r",?\s*\b(?:e\.g\.|for example)\s*\"[^\"]*\"(?:\s*or\s*\"[^\"]*\")*")


def without_examples(prompt: str) -> str:
    """The prompt with its quoted example phrasings removed (JSON format lines kept)."""
    body, _, fmt = prompt.rpartition("\nJSON: ")
    if not body:
        return prompt
    return EXAMPLE_INLINE.sub("", EXAMPLE_PARENS.sub("", body)) + PHRASING_NOTE + "\nJSON: " + fmt


# --------------------------------------------------------------------------- seeds
def documents(sqlite_path: Path) -> list[dict]:
    db = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    rows = db.execute("SELECT DISTINCT product_id, product_version, document_id FROM chunks").fetchall()
    db.close()
    return [{"product_id": p, "product_version": v, "document_id": d} for p, v, d in rows]


def load_sections(sqlite_path: Path, docs: list[dict], products) -> list[dict]:
    ids = [d["document_id"] for d in docs if d["product_id"] in products and d["product_version"]]
    out = []
    for s in sections(sqlite_path, ids, level=3, max_chars=1800):
        text = " ".join(c["text"] for c in s["chunks"]).strip()
        if len(text) < 200 or not s["headings"]:
            continue  # contents pages, stubs
        out.append({**s, "text": text[:900], "chapter": s["headings"][0]})
    return out


def passage(s: dict) -> str:
    name = PRODUCT_NAMES.get(s["product_id"], s["product_id"])
    return f"[{name} {s['product_version']} manual] {' > '.join(s['headings'])}\n{s['text']}"


def name(product: str) -> str:
    return PRODUCT_NAMES.get(product, product)


# --------------------------------------------------------------------------- scenarios
# Each returns (prompt, finish) where finish(json_reply) -> list of examples (or [] if unusable).
def label(action, product=None, versions=(), needs=(), query=None):
    return {"action": action, "product": product, "versions": list(versions), "needs": list(needs), "query": query}


# The standalone is the rewrite target a learned router imitates. The app's rewrite must not add search terms
# the user never wrote, so neither may the label: the passage is there only to make the messages realistic.
STANDALONE_RULE = ("The standalone may only use words and facts from the two user messages: do not add any detail "
                   "from the passage that the user did not say.")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
                word = word[:-1]  # setting -> set, logged -> log
            return word
    return word


# Conversational words a rewrite may add without adding anything to search for.
FILLER = {"help", "need", "try", "trying", "want", "understand", "understanding", "use", "using", "used", "explain",
          "know", "like", "question", "steps", "step", "way", "does", "work", "works", "working", "manual", "tell",
          "possible", "able", "currently", "still", "actually", "same", "instead"}


def added_terms(query: str, messages: list[str], product: str) -> list[str]:
    """Content words of the standalone that appear in neither message (product name and versions allowed)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    def words(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z][a-z0-9+#-]*", text.casefold().replace("’", "'"))
                if w not in ENGLISH_STOP_WORDS and w not in FILLER and len(w) > 2]

    allowed = {_stem(w) for m in messages for w in words(m)}
    allowed |= {_stem(w) for w in words(f"{product} {name(product)} version versions")}
    # Prefix match tolerates inflections the stemmer misses (configure / configuration, setup / setting).
    def known(w: str) -> bool:
        s = _stem(w)
        return s in allowed or (len(s) >= 5 and any(len(a) >= 5 and a[:5] == s[:5] for a in allowed))
    return sorted({w for w in words(query) if not known(w)})


def s_single(rng, pool, by_key, style):
    s = rng.choice(pool)
    prompt = (f"Here is a passage from a manual.\n\n{passage(s)}\n\nWrite one question a user could ask that this "
              f"passage answers. Mention {name(s['product_id'])} {s['product_version']} in the question about half the "
              f"time. {STYLES[style]}\nAlso give `need`: a short search phrase (3-8 words, no full sentence) for what must be found.\n"
              'JSON: {"question": "...", "need": "..."}')

    def finish(r):
        return [dict(question=r["question"], history=[], seed=[s["section_id"]],
                     label=label("direct", s["product_id"], [s["product_version"]], [r["need"]]),
                     ui=_ui(rng, s, r["question"]))]
    return prompt, finish


def related(s: dict, candidates: list[dict], k: int = 8) -> list[dict]:
    """The k candidates whose text is closest to s (TF-IDF), so a multi-step goal joins sections that belong together."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [" ".join(c["headings"]) + " " + c["text"] for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(texts + [s["text"]])
    scores = cosine_similarity(vectorizer.transform([" ".join(s["headings"]) + " " + s["text"]]),
                               vectorizer.transform(texts))[0]
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i] for i in order[:k] if scores[i] > 0.05]


def s_multistep(rng, pool, by_key, style):
    s = rng.choice(pool)
    same_doc = [t for t in pool if t["document_id"] == s["document_id"] and t["chapter"] != s["chapter"]]
    near = related(s, same_doc) if len(same_doc) >= 2 else []
    if len(near) < 2:
        return None
    parts = [s] + rng.sample(near, rng.choice([1, 2]))
    text = "\n\n".join(f"Passage {i}: {passage(p)}" for i, p in enumerate(parts, 1))
    prompt = (f"Here are {len(parts)} passages from different chapters of one manual.\n\n{text}\n\nWrite one question "
              f"about a single goal a user wants to reach that needs ALL of these passages to answer (for example a "
              f"task whose steps are in different chapters). It must be one goal a real user would have, not two "
              f"unrelated tasks joined with \"and\"; if the passages do not fit one goal, set \"question\" to null. "
              f"Do not list the passages or chapter names. {STYLES[style]}\n"
              f"Also give `needs`: one short search phrase per passage, in the order the user would do them.\n"
              'JSON: {"question": "...", "needs": ["...", "..."]}')

    def finish(r):
        needs = [n for n in r.get("needs") or [] if isinstance(n, str) and n.strip()]
        if not r.get("question") or len(needs) < 2:
            return []
        return [dict(question=r["question"], history=[], seed=[p["section_id"] for p in parts],
                     label=label("direct", s["product_id"], [s["product_version"]], needs[:4]),
                     ui=_ui(rng, s, r["question"]))]
    return prompt, finish


def _counterparts(s: dict, by_key: dict) -> list[dict]:
    return [t for t in by_key.get((s["product_id"], tuple(h.casefold() for h in s["headings"])), [])
            if t["product_version"] != s["product_version"]]


def s_compare(rng, pool, by_key, style):
    # Seed only from sections that exist in another version too (most random seeds have no counterpart).
    for _ in range(50):
        s = rng.choice(pool)
        others = _counterparts(s, by_key)
        if others:
            break
    else:
        return None
    t = rng.choice(others)
    a, b = sorted([s, t], key=lambda x: [int(p) for p in re.findall(r"\d+", x["product_version"])])
    form = rng.choice([f"{name(a['product_id'])} {a['product_version']} and {b['product_version']}",
                       f"from {name(a['product_id'])} {a['product_version']} to {b['product_version']}",
                       f"{name(a['product_id'])} {b['product_version']} compared with {a['product_version']}",
                       f"{name(a['product_id'])} {a['product_version']} vs {name(a['product_id'])} {b['product_version']}"])
    prompt = (f"Here is the same section in two versions of a manual.\n\nOlder: {passage(a)}\n\nNewer: {passage(b)}\n\n"
              f"Write one question a user upgrading (or choosing between versions) would ask about how this feature "
              f"or task differs between the two versions of the software. Ask about the software, never about the "
              f"manuals, their sections or their wording. The question must "
              f"name both versions, for example \"{form}\" (you may phrase it differently but both version numbers "
              f"must appear). {STYLES[style]}\nAlso give `need`: a short search phrase.\n"
              'JSON: {"question": "...", "need": "..."}')

    def finish(r):
        q = r["question"]

        def named(v):  # "6.0" may be written "6"
            return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d]|\.\d)", q)
                       for f in [v] + ([v[:-2]] if v.endswith(".0") else []))
        if not (named(a["product_version"]) and named(b["product_version"])) or ABOUT_MANUAL.search(q):
            return []
        return [dict(question=q, history=[], seed=[a["section_id"], b["section_id"]],
                     label=label("direct", a["product_id"], [a["product_version"], b["product_version"]], [r["need"]]),
                     ui={"product_id": None, "product_version": None})]
    return prompt, finish


def s_followup(rng, pool, by_key, style):
    s = rng.choice(pool)
    prompt = (f"Here is a passage from a manual.\n\n{passage(s)}\n\nWrite a two-message exchange. `first`: the user's "
              f"first message, naming {name(s['product_id'])} {s['product_version']} and the topic. `follow_up`: their "
              f"next message, which only makes sense after the first one: it refers back with a pronoun (it, that, "
              f"them, those, one) or leaves the subject out (\"and on Linux?\", \"what about the admin side?\"). "
              f"`standalone`: the follow-up rewritten as a self-contained question that names the product, version "
              f"and subject. {STANDALONE_RULE} {STYLES[style]} (apply the style to follow_up)\n"
              'JSON: {"first": "...", "follow_up": "...", "standalone": "...", "need": "..."}')

    def finish(r):
        return [dict(question=r["follow_up"], history=[{"role": "user", "content": r["first"]}], seed=[s["section_id"]],
                     label=label("rewrite", s["product_id"], [s["product_version"]], [r["need"]], r["standalone"]),
                     ui=rng.choice([{"product_id": None, "product_version": None},
                                    {"product_id": s["product_id"], "product_version": s["product_version"]}]))]
    return prompt, finish


def s_self_contained(rng, pool, by_key, style):
    s = rng.choice(pool)
    word = rng.choice(["it", "that", "them", "instead", "those", "this"])
    prompt = (f"Here is a passage from a manual.\n\n{passage(s)}\n\nWrite one question this passage answers that "
              f"names its own subject first and later refers back to it with the word \"{word}\" in the SAME "
              f"question, e.g. \"Can X do Y? How do I set that up?\" or \"My X is broken, how do I fix it?\". "
              f"It must make sense with no earlier conversation. {STYLES[style]}\nAlso give `need`.\n"
              'JSON: {"question": "...", "need": "..."}')

    def finish(r):
        if not re.search(rf"\b{word}\b", r["question"], re.I):
            return []
        return [dict(question=r["question"], history=[], seed=[s["section_id"]],
                     label=label("direct", s["product_id"], [s["product_version"]], [r["need"]]),
                     ui=_ui(rng, s, r["question"]))]
    return prompt, finish


def s_missing_context(rng, pool, by_key, style):
    prompt = ("Write 5 different short messages a user might send as the FIRST message to a support chatbot, "
              "that cannot be understood without earlier context: they point at something never named (\"How do I "
              "turn it off?\", \"Where is that setting?\", \"Why does it keep failing?\", \"Can I do the same for "
              "the others?\"). The thing asked about must stay unnamed: refer to it only as it, that, this, those, "
              "them, the same or the others. Do not name a product, app, device, file, feature, error or any other "
              "subject. At most 15 words each. Vary the wording. " + STYLES[style] + '\nJSON: {"messages": ["...", ...]}')

    def finish(r):
        # A message that names its subject ("the app crashes when I upload files") is a product-less question,
        # which the other scenarios label as a search over everything, not a clarification.
        return [dict(question=m, history=[], seed=[], label=label("clarify"),
                     ui={"product_id": None, "product_version": None})
                for m in (r.get("messages") or [])[:5]
                if isinstance(m, str) and m.strip() and len(m.split()) <= 15 and not NAMED_SUBJECT.search(m)
                and REFERENCE.search(m)]
    return prompt, finish


REFERENCE = re.compile(r"\b(it|that|this|those|these|them|same|others?|there)\b", re.I)
NAMED_SUBJECT = re.compile(r"\b(apps?|application|device|software|system|files?|uploads?|network|server|error "
                           r"message|installation|install(ed)?|calibration|password|manual|instructions|program)\b",
                           re.I)


def s_edit(rng, pool, by_key, style):
    s = rng.choice(pool)
    prompt = (f"Here is a passage from a manual.\n\n{passage(s)}\n\n`first`: a question a user asks about it (name "
              f"{name(s['product_id'])} {s['product_version']}). `edit`: the user's next message after getting an "
              f"answer, asking ONLY to change the form of that answer (shorter, a numbered list, simpler words, more "
              f"detail on what it said, a table, two sentences...) without asking anything new. The edit must not "
              f"mention the topic, the product or the manual again; it only refers to \"that\", \"it\" or \"your "
              f"answer\" (e.g. \"can you make that a numbered list?\", \"shorter pls\"). {STYLES[style]} "
              '(apply the style to edit)\nJSON: {"first": "...", "edit": "..."}')

    def finish(r):
        return [dict(question=r["edit"], history=[{"role": "user", "content": r["first"]}], seed=[s["section_id"]],
                     label=label("edit"), ui={"product_id": None, "product_version": None}),
                # The same request with nothing answered yet has nothing to edit.
                dict(question=r["edit"], history=[], seed=[s["section_id"]], label=label("clarify"),
                     ui={"product_id": None, "product_version": None}, variant="edit_without_answer")]
    return prompt, finish


def s_correction(rng, pool, by_key, style):
    s = rng.choice(pool)
    versions = sorted({t["product_version"] for t in pool if t["product_id"] == s["product_id"]} - {s["product_version"]})
    if not versions:
        return None
    other = rng.choice(versions)
    prompt = (f"Here is a passage from a manual.\n\n{passage(s)}\n\n`first`: the user's question about one specific "
              f"thing in it, naming {name(s['product_id'])} {s['product_version']} and that thing (not just the "
              f"manual or a chapter title). `correction`: their next message saying they actually "
              f"use version {other} instead (e.g. \"sorry, I meant {other}\", \"actually we're still on {other}, same "
              f"there?\"); it must contain \"{other}\" and must not repeat the whole question. `standalone`: the "
              f"original question rewritten for version {other}. {STANDALONE_RULE} {STYLES[style]} (apply the style "
              f"to correction)\n"
              '`need`: a 3-8 word search phrase.\n'
              'JSON: {"first": "...", "correction": "...", "standalone": "...", "need": "..."}')

    def finish(r):
        if other not in r["correction"]:
            return []
        return [dict(question=r["correction"], history=[{"role": "user", "content": r["first"]}], seed=[s["section_id"]],
                     label=label("rewrite", s["product_id"], [other], [r["need"]], r["standalone"]),
                     ui={"product_id": None, "product_version": None})]
    return prompt, finish


def s_switch(rng, pool, by_key, style, everything=None):
    s = rng.choice(pool)
    others = [t for t in everything if t["product_id"] != s["product_id"]]
    t = rng.choice(others)
    prompt = (f"Two passages from manuals of different products.\n\nA: {passage(t)}\n\nB: {passage(s)}\n\n`first`: a "
              f"question about how to do something that A answers; it must contain the text \"{name(t['product_id'])} "
              f"{t['product_version']}\". `question`: the user's next message, switching to a new, self-contained "
              f"question about how to do something that B answers; it must contain the text \"{name(s['product_id'])} "
              f"{s['product_version']}\" (it may start with \"new "
              f"question:\", \"different thing:\" or nothing). Both are about using the software, never about the "
              f"manual, its sections or its wording. {STYLES[style]} (apply the style to question)\n"
              f"Also give `need`: a 3-8 word search phrase for B.\n"
              'JSON: {"first": "...", "question": "...", "need": "..."}')

    def finish(r):
        q = r["question"]
        if name(s["product_id"]).casefold() not in q.casefold() or s["product_version"] not in q:
            return []
        if ABOUT_MANUAL.search(q) or ABOUT_MANUAL.search(r["first"]) or len(str(r.get("need", "")).split()) < 2:
            return []
        return [dict(question=q, history=[{"role": "user", "content": r["first"]}], seed=[t["section_id"], s["section_id"]],
                     label=label("direct", s["product_id"], [s["product_version"]], [r["need"]]),
                     ui={"product_id": None, "product_version": None})]
    return prompt, finish


ABOUT_MANUAL = re.compile(r"\b(manual|passage|section \d|chapter \d)\b|\bcalled\b", re.I)


def s_two_products(rng, pool, by_key, style, everything=None):
    s = rng.choice(pool)
    t = rng.choice([x for x in everything if x["product_id"] != s["product_id"]])
    prompt = (f"Write one question that involves BOTH {name(s['product_id'])} and {name(t['product_id'])} by name "
              f"(e.g. moving data between them, or which of the two to use). Topics for inspiration: "
              f"\"{' > '.join(s['headings'])}\" and \"{' > '.join(t['headings'])}\". {STYLES[style]}\n"
              'JSON: {"question": "..."}')

    def finish(r):
        q = r["question"].casefold()
        if name(s["product_id"]).casefold() not in q or name(t["product_id"]).casefold() not in q:
            return []
        return [dict(question=r["question"], history=[], seed=[s["section_id"], t["section_id"]],
                     label=two_product_label(r["question"]), ui={"product_id": None, "product_version": None})]
    return prompt, finish


def two_product_label(question: str) -> dict:
    """Round 3 (2026-11-20): a question naming several products is searched in each, in the order named."""
    q = question.casefold()
    found = {}
    for p in PRODUCT_NAMES:
        hits = [i for i in (q.find(name(p).casefold()), q.find(p)) if i >= 0]
        if hits:
            found[p] = min(hits)
    named = sorted(found, key=found.get)
    return label("direct", named if len(named) > 1 else (named[0] if named else None))


def s_version_reply(rng, pool, by_key, style):
    """Round 3: a question with no version, then the user's version as the next message ("30", "im on kicad 9")."""
    s = rng.choice(pool)
    v = s["product_version"]
    prompt = (f"Here is a passage from a manual.\n\n{passage(s)}\n\n`first`: a question a user asks that this "
              f"passage answers. It names {name(s['product_id'])} but no version number. The assistant then asks "
              f"which version they use. `reply`: the user's answer, giving version {v} the way a real user would "
              f"(sometimes just the number, sometimes a few words); it must contain \"{v}\" and must not repeat the "
              f"question. `standalone`: the first question rewritten for {name(s['product_id'])} {v}. {STANDALONE_RULE} "
              f"{STYLES[style]} (apply the style to reply)\n`need`: a 3-8 word search phrase.\n"
              'JSON: {"first": "...", "reply": "...", "standalone": "...", "need": "..."}')

    def finish(r):
        if v not in r["reply"] or re.search(rf"(?<![\d.]){re.escape(v)}(?![\d]|\.\d)", r["first"]):
            return []
        if name(s["product_id"]).casefold() not in r["first"].casefold():
            return []
        return [dict(question=r["reply"], history=[{"role": "user", "content": r["first"]}], seed=[s["section_id"]],
                     label=label("rewrite", s["product_id"], [v], [r["need"]], r["standalone"]),
                     ui={"product_id": None, "product_version": None})]
    return prompt, finish


def s_version_form(rng, pool, by_key, style, docs=None):
    s = rng.choice(pool)
    stored = s["product_version"]
    forms = [f"v{stored}", f"version {stored}"]
    if stored.endswith(".0"):
        forms.append(stored[:-2])
    form = rng.choice([f for f in forms if normalize_version(s["product_id"], f.removeprefix("version "), docs) == [stored]])
    written = f"{name(s['product_id'])} {form}"
    prompt = (f"Here is a passage from a manual.\n\n{passage(s)}\n\nWrite one question this passage answers. It must "
              f"contain exactly the text \"{written}\". {STYLES[style]}\nAlso give `need`.\n"
              'JSON: {"question": "...", "need": "..."}')

    def finish(r):
        if written.casefold() not in r["question"].casefold():
            return []
        return [dict(question=r["question"], history=[], seed=[s["section_id"]],
                     label=label("direct", s["product_id"], [stored], [r["need"]]),
                     ui={"product_id": None, "product_version": None})]
    return prompt, finish


def s_chat(rng, pool, by_key, style):
    kind = rng.choice(["greetings or thanks (\"hi there\", \"thanks, that worked!\")",
                       "questions about the assistant itself (\"who are you?\", \"what can you do?\")",
                       "off-topic requests (weather, jokes, sports scores, news, homework)"])
    prompt = (f"Write 5 different messages a user might send to a product-manual support chatbot that are {kind}. "
              f"They must not ask about any product feature. {STYLES[style]}\n" + 'JSON: {"messages": ["...", ...]}')

    def finish(r):
        return [dict(question=m, history=[], seed=[], label=label("chat"), ui={"product_id": None, "product_version": None})
                for m in (r.get("messages") or [])[:5] if isinstance(m, str) and m.strip()]
    return prompt, finish


# Added 2026-11-19 (training data only; aug_v0_1). F confused "reformat the answer" with "find more" when the
# edit named its topic, and complaints about the assistant with missing-context clarifications.
EDIT_FORMS = ["as numbered steps", "shorter", "in simpler words", "as a table", "as bullet points",
              "in one or two sentences", "as a summary of the key points", "as a checklist",
              "with less jargon", "in a different order, most important first"]
# An edit must say how to change the answer; without one of these it may be a new question, so it is dropped.
EDIT_CUE = re.compile(r"\b(steps?|numbered|numbers|list|short(er)?|brief(ly)?|concise|simpl\w*|plain|table|"
                      r"bullets?|points?|sentences?|summar\w*|checklist|jargon|order|again|repeat|reformat|format|"
                      r"tl;?dr|condense|rephrase|reword)\b", re.I)
MORE_ANGLES = ["a related setting", "what to do when it goes wrong", "the task that usually comes next",
               "a limit or requirement", "how to undo or reverse it", "a different option for the same goal"]


def s_edit_named(rng, pool, by_key, style):
    s = rng.choice(pool)
    form, angle = rng.choice(EDIT_FORMS), rng.choice(MORE_ANGLES)
    prompt = (f"Here is a passage from a manual.\n\n{passage(s)}\n\n`first`: a question a user asks about it (name "
              f"{name(s['product_id'])} {s['product_version']}). After the user gets an answer, write two possible "
              f"next messages.\n`edit`: asks ONLY to get that same answer again {form}. It mentions the topic of the "
              f"answer in a few words of the user's own (so it could be mistaken for a new question), but asks for "
              f"nothing the answer did not already say. Write it the way this user would; do not reuse the wording "
              f"of these instructions.\n`more`: asks for something NEW about the same topic that the answer "
              f"would not contain ({angle}), without naming the product.\n"
              f"`standalone`: `more` rewritten as one self-contained question with the product and version. "
              f"{STANDALONE_RULE} `need`: a 3-8 word search phrase for `more`. {STYLES[style]} (apply the style to "
              'edit and more)\nJSON: {"first": "...", "edit": "...", "more": "...", "standalone": "...", "need": "..."}')

    def finish(r):
        history = [{"role": "user", "content": r["first"]}]
        none = {"product_id": None, "product_version": None}
        edit = [dict(question=r["edit"], history=history, seed=[s["section_id"]], label=label("edit"), ui=none)]
        return (edit if EDIT_CUE.search(r["edit"]) else []) + [
                dict(question=r["more"], history=history, seed=[s["section_id"]], ui=none, variant="more",
                     label=label("rewrite", s["product_id"], [s["product_version"]], [r["need"]], r["standalone"]))]
    return prompt, finish


def s_assistant_feedback(rng, pool, by_key, style):
    kind = rng.choice(["complaints that the assistant did not answer or did not help",
                       "remarks that an answer was wrong, useless or too long, without saying what it was about",
                       "questions about why the assistant behaves as it does or what it can and cannot do",
                       "frustrated or confused messages about using the chatbot itself"])
    prompt = (f"Write 5 different messages a user might send to a product-manual support chatbot that are {kind}. "
              f"They are about the chatbot and its answers, never about a product, a feature, a file, a step that "
              f"failed, or finding or downloading a manual. "
              f"{STYLES[style]}\n" + 'JSON: {"messages": ["...", ...]}')

    def finish(r):
        return [dict(question=m, history=[], seed=[], label=label("chat"), ui={"product_id": None, "product_version": None})
                for m in (r.get("messages") or [])[:5] if isinstance(m, str) and m.strip() and clearly_about_assistant(m)]
    return prompt, finish

# Round 3: "chat is stuck", "keeps giving errors" read like an unnamed product problem (clarify), so a complaint
# is kept as chat only when it plainly addresses the assistant and names no failure.
PROBLEM_WORDS = re.compile(r"\b(stuck|errors?|crash\w*|freez\w*|hang\w*|broken|bug\w*|fail\w*|not (?:working|"
                           r"responding|loading|saving)|(?:does|did|is|was)n'?t (?:work|respond|load|save)\w*|"
                           r"blank|missing|gone|disappear\w*|lag\w*|slow)\b", re.I)
ADDRESSES_ASSISTANT = re.compile(r"\b(you|your|you're|bot|chatbot|assistant|ai|answer\w*|repl(?:y|ies)|responses?)\b",
                                 re.I)


def clearly_about_assistant(message: str) -> bool:
    return bool(ADDRESSES_ASSISTANT.search(message)) and not PROBLEM_WORDS.search(message)



ORIGINAL_SCENARIOS = ["single", "multistep", "compare", "followup", "self_contained", "missing_context", "edit",
                      "correction", "switch", "two_products", "version_form", "chat"]
SCENARIOS = {"single": s_single, "multistep": s_multistep, "compare": s_compare, "followup": s_followup,
             "self_contained": s_self_contained, "missing_context": s_missing_context, "edit": s_edit,
             "correction": s_correction, "switch": s_switch, "two_products": s_two_products,
             "version_form": s_version_form, "chat": s_chat,
             "edit_named": s_edit_named, "assistant_feedback": s_assistant_feedback,
             "version_reply": s_version_reply}
# Batch scenarios return 5 examples per call; the others 1 (edit 2).
PER_CALL = {"missing_context": 5, "chat": 5, "assistant_feedback": 5}


def _ui(rng, s, question):
    """Page selection: sometimes the user picked product and version on the page, sometimes nothing."""
    if rng.random() < 0.4:
        return {"product_id": s["product_id"], "product_version": s["product_version"]}
    return {"product_id": None, "product_version": None}


NO_PRODUCT = {"missing_context", "chat", "assistant_feedback"}
# An edit request that retells the topic is not an edit; keep edits short.
SCENARIO_STYLES = {"edit": ["terse", "non_native", "typos"], "edit_named": ["terse", "non_native", "typos"]}


def observable(example: dict) -> dict:
    """Keep only the scope a router could know: named in the messages or selected on the page.

    The seed section fixes the product and version the question is about, but a question that names neither
    (with nothing selected) can only be searched across all versions. Such a label gets no version, and no
    product when the product isn't named either.
    """
    lab = example["label"]
    if lab["action"] not in ("direct", "rewrite") or not lab["product"] or isinstance(lab["product"], list):
        return example
    text = " ".join([example["question"]] + [h["content"] for h in example.get("history") or []]).casefold()
    ui = example.get("ui") or {}
    product = lab["product"]
    named = product in text or name(product).casefold() in text or ui.get("product_id") == product
    if not named:
        return {**example, "label": {**lab, "product": None, "versions": []}}

    def visible(v: str) -> bool:
        forms = [v] + ([v[:-2]] if v.endswith(".0") else [])
        return ui.get("product_version") == v or any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d]|\.\d)", text) for f in forms)

    return {**example, "label": {**lab, "versions": [v for v in lab["versions"] if visible(v)]}}


def split_of(unseen: bool, style: str, key: str) -> str:
    """unseen: seeded from KiCad/FreeCAD. key: stable text for the 90/10 train/val cut."""
    if unseen:
        return "test_unseen"
    if style == "typos":
        return "test_style"
    return "val" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 == 0 else "train"


def leak_filter(examples: list[dict], against: list[str] | None = None, threshold: float = 0.8,
                key: str = "max_bench_similarity") -> tuple[list[dict], list[dict]]:
    """Drop examples whose question is too close to a benchmark question (any split), or to `against`."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    questions = against if against is not None else [c["question"] for c in bench.load(split=None, private=True, statuses=None)]
    if not examples:
        return [], []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(questions + [e["question"] for e in examples])
    similarity = cosine_similarity(vectorizer.transform([e["question"] for e in examples]), vectorizer.transform(questions))
    keep, dropped = [], []
    for example, row in zip(examples, similarity):
        (dropped if row.max() >= threshold else keep).append({**example, key: round(float(row.max()), 3)})
    return keep, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--per-scenario", type=int, default=10, help="model calls per scenario (batch ones give 5 each)")
    parser.add_argument("--scenarios", nargs="*", default=ORIGINAL_SCENARIOS, choices=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--unseen-share", type=float, default=0.25, help="share of calls seeded from KiCad/FreeCAD")
    parser.add_argument("--phrasing", metavar="TRAIN_SET",
                        help="write the test_phrasing split: prompts without example phrases, a different persona, "
                             "and nothing with TF-IDF cosine >= 0.7 to a question of TRAIN_SET's train split")
    args = parser.parse_args()
    system = PHRASING_SYSTEM if args.phrasing else SYSTEM

    from manuora.generation.model_control import is_loaded, load_model, unload_model

    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    sqlite_path = ROOT / settings()["sparse_index"]
    docs = [d for d in documents(sqlite_path) if d["product_id"] in TRAIN_PRODUCTS + UNSEEN_PRODUCTS]
    seen_pool = load_sections(sqlite_path, docs, TRAIN_PRODUCTS)
    unseen_pool = load_sections(sqlite_path, docs, UNSEEN_PRODUCTS)
    everything = seen_pool + unseen_pool
    by_key: dict = {}
    for s in everything:
        by_key.setdefault((s["product_id"], tuple(h.casefold() for h in s["headings"])), []).append(s)
    print(f"sections: {len(seen_pool)} training products, {len(unseen_pool)} unseen products")

    generation = settings()["generation"]
    loaded_here = not is_loaded(generation["model"])
    if loaded_here:
        load_model(generation["model"], generation["url"])
    examples, ungrounded, failures, started = [], [], 0, time.perf_counter()
    try:
        for scenario in args.scenarios:
            maker = SCENARIOS[scenario]
            # A batch call gives several examples; fewer calls keep the scenarios about the same size.
            for n in range(max(1, args.per_scenario // PER_CALL.get(scenario, 1))):
                unseen = scenario not in NO_PRODUCT and rng.random() < args.unseen_share
                pool = unseen_pool if unseen else seen_pool
                # Unseen-product tests keep the training styles, so product and style effects stay separate.
                allowed = SCENARIO_STYLES.get(scenario, list(STYLES))
                style = rng.choice([s for s in allowed if s != "typos"]) if unseen else rng.choice(allowed)
                # A training example never sees an unseen product, even as the "other" product of a switch.
                extra = ({"everything": everything if unseen else seen_pool} if scenario in ("switch", "two_products")
                         else {"docs": docs} if scenario == "version_form" else {})
                made = maker(rng, pool, by_key, style, **extra)
                if made is None:
                    failures += 1
                    continue
                prompt, finish = made
                if args.phrasing:
                    prompt = without_examples(prompt)
                try:
                    reply = call([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                 max_tokens=400, temperature=1.0 if args.phrasing else 0.8)
                    text = re.sub(r"<think>.*?</think>", "", reply["text"], flags=re.S)
                    data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                    made_examples = finish(data)
                except (ValueError, AttributeError, KeyError, TypeError, OSError) as exc:
                    failures += 1
                    print(f"  {scenario} #{n}: unusable reply ({type(exc).__name__})")
                    continue
                for e in made_examples:
                    if not str(e.get("question", "")).strip():
                        continue
                    key = "|".join(e["seed"]) or e["question"]
                    e = observable(e)
                    e["label"]["needs"] = [n for n in e["label"]["needs"] if len(n.split()) <= 12] or e["label"]["needs"][:1]
                    e.update(scenario=scenario, style=style,
                             split="test_phrasing" if args.phrasing else split_of(unseen, style, key))
                    if args.phrasing:
                        e["seeded_unseen"] = unseen
                    query = e["label"].get("query")
                    added = added_terms(query, [e["question"]] + [h["content"] for h in e["history"]],
                                        e["label"]["product"] or "") if query else []
                    if added:
                        ungrounded.append({**e, "drop_reason": "standalone adds " + ", ".join(added)})
                        continue
                    examples.append(e)
            print(f"{scenario}: {sum(e['scenario'] == scenario for e in examples)} examples "
                  f"({time.perf_counter() - started:.0f} s)")
    finally:
        if loaded_here:
            unload_model(generation["model"])
    keep, dropped = leak_filter(examples)
    for e in dropped:
        e["drop_reason"] = f"benchmark similarity {e['max_bench_similarity']}"
    if args.phrasing:
        train = [r["question"] for r in bench.read_file(OUT / args.phrasing / "examples.jsonl") if r["split"] == "train"]
        keep, near = leak_filter(keep, against=train, threshold=0.7, key="max_train_similarity")
        for e in near:
            e["drop_reason"] = f"train similarity {e['max_train_similarity']}"
        dropped += near
    for i, e in enumerate(keep, 1):
        e["id"] = f"{args.name}-{i:05d}"
    bench.write_file(out / "examples.jsonl", keep)
    bench.write_file(out / "dropped.jsonl", dropped + ungrounded)
    summary = {"name": args.name, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "per_scenario": args.per_scenario,
               "seed": args.seed, "model": generation["model"], "examples": len(keep), "dropped_as_leak": len(dropped),
               "dropped_ungrounded_standalone": len(ungrounded),
               "failed_calls": failures, "seconds": round(time.perf_counter() - started),
               "by_split": {s: sum(e["split"] == s for e in keep) for s in ("train", "val", "test_unseen", "test_style", "test_phrasing")},
               "by_scenario": {s: sum(e["scenario"] == s for e in keep) for s in SCENARIOS}}
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
