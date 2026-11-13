"""Task outlines: offline links from a user task to the manual passages that hold its steps (experiment Q).

Built before serving, one model at a time:
1. `sections` cuts each manual into section windows of consecutive chunks.
2. The chat model reads one window and writes a card (`outline_messages` / `parse_outline`):
   user goals the section serves, its steps with the passage that states each one, and what the user must
   already have done elsewhere in the manual ("needs").
3. Needs are resolved offline by retrieval + rerank in the same manual version (`resolve_need`).
   Every link must pass `backed_by`: the step or need's content words occur in the linked passage,
   so a card can't point at a passage that doesn't say what the card claims.

At question time (`retrieval.outlines` in config/assistant.yaml, off by default) the question is matched
to card goals with the app's embedding model, and the matched cards' passages join the evidence right
after the best reranked passage (`merge_outline`), like experiment L's oracle arm.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

MAX_SECTION_CHARS = 5000
MAX_PASSAGE_CHARS = 1800
MAX_GOALS, MAX_STEPS, MAX_NEEDS = 3, 6, 3

OUTLINE_PROMPT = (
    "You index one section of the {product} manual for a help assistant. The section is split into numbered "
    "passages. Write a card for the user tasks this section helps with.\n"
    "- goals: 1 to 3 questions a user could ask that this section answers, in the user's own words "
    "(e.g. \"How do I share a folder with a group?\").\n"
    "- steps: the actions this section describes, in order (at most 6). Each step is a short phrase in the "
    "manual's wording, with the number of the passage that states it.\n"
    "- needs: at most 3 things the user must already have done or set up before these steps, that the section "
    "mentions or clearly assumes but does not explain itself (e.g. \"enable the app\", \"create a vector layer\"). "
    "Use the manual's wording. Leave the list empty if there are none.\n"
    "Use only what the passages say. If the section is not about doing something (a reference table, release "
    "notes, a glossary, legal text, a table of contents), reply {{\"goals\": []}}.\n"
    "Reply with JSON only: {{\"goals\": [\"...\"], \"steps\": [{{\"text\": \"...\", \"passage\": 1}}], "
    "\"needs\": [\"...\"]}} /no_think"
)

_STOP = {
    "about", "after", "again", "also", "before", "being", "between", "both", "each", "every", "first", "from",
    "have", "here", "into", "make", "more", "must", "need", "only", "other", "same", "should", "some", "such",
    "that", "their", "them", "then", "there", "these", "this", "those", "through", "under", "using", "very",
    "when", "where", "which", "while", "will", "with", "within", "without", "your", "user", "users", "want",
    "open", "click", "select", "choose", "enter", "button", "dialog", "menu", "option", "options", "settings",
    "setting", "file", "files", "manual", "section", "step", "steps", "able", "allow", "allows", "used", "uses",
}


def _ordinal(chunk_id: str) -> int:
    tail = chunk_id.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def sections(sqlite_path, document_ids, *, level: int = 3, max_chars: int = MAX_SECTION_CHARS) -> list[dict]:
    """Windows of consecutive chunks that share a heading path down to `level`, at most `max_chars` of text."""
    db = sqlite3.connect(f"file:{Path(sqlite_path).as_posix()}?mode=ro", uri=True)
    marks = ",".join("?" for _ in document_ids)
    rows = db.execute(f"select chunk_id, document_id, product_id, product_version, headings, text from chunks"
                      f" where document_id in ({marks})", list(document_ids)).fetchall()
    db.close()
    rows.sort(key=lambda r: (r[1], _ordinal(r[0])))
    out: list[dict] = []
    current, size = None, 0
    for chunk_id, doc, product, version, headings, text in rows:
        heads = [str(h) for h in json.loads(headings or "[]")]
        key = (doc, tuple(heads[:level]))
        text = (text or "")[:MAX_PASSAGE_CHARS]
        if current is None or current["_key"] != key or size + len(text) > max_chars:
            current = dict(_key=key, section_id=f"{doc}#{len(out):05d}", document_id=doc, product_id=product,
                           product_version=version, headings=list(key[1]), chunks=[])
            out.append(current); size = 0
        current["chunks"].append(dict(chunk_id=chunk_id, headings=heads, text=text))
        size += len(text)
    for s in out:
        del s["_key"]
    return out


def outline_messages(section: dict, product_name: str) -> list[dict]:
    version = section.get("product_version")
    product = product_name + (f" {version}" if version not in (None, "", "None") else "")
    lines = ["Section: " + " > ".join(section["headings"]), ""]
    for n, chunk in enumerate(section["chunks"], 1):
        sub = chunk["headings"][len(section["headings"]):]
        lines.append(f"[{n}]" + (f" ({' > '.join(sub)})" if sub else "") + " " + chunk["text"])
    return [{"role": "system", "content": OUTLINE_PROMPT.format(product=product)},
            {"role": "user", "content": "\n".join(lines)}]


def parse_outline(text: str, section: dict) -> dict | None:
    """The model's card, with step passages turned into chunk ids. None when the reply is unusable or empty."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    def strings(values, limit):
        return [v.strip() for v in values if isinstance(v, str) and 3 <= len(v.strip()) <= 240][:limit] \
            if isinstance(values, list) else []

    goals = strings(data.get("goals"), MAX_GOALS)
    if not goals:
        return None
    chunks = section["chunks"]
    steps = []
    for step in (data.get("steps") or [])[:MAX_STEPS] if isinstance(data.get("steps"), list) else []:
        if not isinstance(step, dict) or not isinstance(step.get("text"), str):
            continue
        try:
            n = int(step.get("passage"))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(chunks):
            steps.append(dict(text=step["text"].strip(), chunk_id=chunks[n - 1]["chunk_id"]))
    return dict(section_id=section["section_id"], document_id=section["document_id"],
                product_id=section["product_id"], product_version=section["product_version"],
                headings=section["headings"], goals=goals, steps=steps, needs=strings(data.get("needs"), MAX_NEEDS))


def _content(text: str) -> list[str]:
    words = re.sub(r"[^a-z0-9]+", " ", text.casefold()).split()
    return list(dict.fromkeys(w[:-1] if len(w) > 4 and w.endswith("s") else w
                              for w in words if len(w) >= 4 and w not in _STOP))


def backed_by(phrase: str, passage: str, share: float = 0.5) -> bool:
    """At least `share` of the phrase's content words (4+ letters, plural-insensitive) occur in the passage."""
    words = _content(phrase)
    if not words:
        return False
    hay = " " + re.sub(r"[^a-z0-9]+", " ", passage.casefold()) + " "
    flat = re.sub(r"[^a-z0-9]+", "", passage.casefold())
    found = sum(1 for w in words if f" {w}" in hay or (len(w) >= 6 and w in flat))
    return found / len(words) >= share


def passage_text(hit: dict) -> str:
    return " / ".join(str(h) for h in (hit.get("headings") or [])) + "\n" + str(hit.get("text") or "")


def verify_steps(card: dict, passages: dict[str, dict]) -> dict:
    """Keep steps whose phrase is backed by the passage they point to."""
    kept = [s for s in card["steps"] if s["chunk_id"] in passages and backed_by(s["text"], passage_text(passages[s["chunk_id"]]))]
    return {**card, "steps": kept, "dropped_steps": len(card["steps"]) - len(kept)}


def resolve_need(need: str, ranked: list[dict], own: set[str], *, min_score: float = 0.0) -> str | None:
    """Best reranked passage outside the card's own section that backs the need."""
    for hit in ranked:
        if hit["chunk_id"] in own:
            continue
        if hit.get("rerank_score", 0.0) < min_score:
            return None
        if backed_by(need, passage_text(hit)):
            return hit["chunk_id"]
    return None


def card_chunks(card: dict) -> list[str]:
    """The card's passages in task order: its own steps, then what must be done first."""
    needs = [n["chunk_id"] for n in card.get("linked_needs", []) if n.get("chunk_id")]
    return list(dict.fromkeys(needs + [s["chunk_id"] for s in card["steps"]]))


def merge_outline(base: list[dict], outline_hits: list[dict], max_passages: int) -> list[dict]:
    """Best reranked passage first, then the outline's passages, then the rest of the selection."""
    out, seen = [], set()
    for hit in base[:1] + outline_hits + base[1:]:
        if hit["chunk_id"] not in seen:
            seen.add(hit["chunk_id"]); out.append(hit)
    return out[:max_passages]


TASK_PROMPT = (
    "Below is an index of sections of the {product} manual: one line per section, with its id, heading and a "
    "question it answers. Write the user tasks that need SEVERAL of these sections, for example setting something "
    "up and then using it, or a procedure whose steps are described in different places.\n"
    "For each task give 1 to 3 questions a user would ask for it, in the user's own words, and the ids of the "
    "sections the user needs, in the order they would use them (2 to 6 ids). Only use ids from the index. "
    "Write at most {limit} tasks.\n"
    "Reply with JSON only: {{\"tasks\": [{{\"goals\": [\"...\"], \"sections\": [\"S1\", \"S4\"]}}]}} /no_think"
)
PER_SECTION = 2  # step passages a task takes from each of its sections


def task_messages(cards: list[dict], product_name: str, limit: int = 25) -> tuple[list[dict], dict[str, str]]:
    """Index lines for a list of section cards; returns the messages and the short id -> section_id map."""
    ids, lines = {}, []
    for n, card in enumerate(cards, 1):
        short = f"S{n}"
        ids[short] = card["section_id"]
        lines.append(f"{short}: {' > '.join(card['headings'][-2:])}: {card['goals'][0]}")
    version = cards[0].get("product_version") if cards else None
    product = product_name + (f" {version}" if version not in (None, "", "None") else "")
    return ([{"role": "system", "content": TASK_PROMPT.format(product=product, limit=limit)},
             {"role": "user", "content": "\n".join(lines)}], ids)


def parse_tasks(text: str, ids: dict[str, str], cards: dict[str, dict], tag: str) -> list[dict]:
    """Task cards in the section-card format, so OutlineIndex and card_chunks work unchanged.

    Their passages are the verified step passages of the sections they list, so a task links only to
    passages a section card already backed.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        tasks = json.loads(m.group(0)).get("tasks", []) if m else []
    except (ValueError, AttributeError):
        return []
    out = []
    for task in tasks if isinstance(tasks, list) else []:
        if not isinstance(task, dict):
            continue
        goals = [g.strip() for g in task.get("goals") or [] if isinstance(g, str) and 3 <= len(g.strip()) <= 240][:MAX_GOALS]
        sections = [ids[s] for s in dict.fromkeys(task.get("sections") or []) if isinstance(s, str) and s in ids]
        if not goals or len(sections) < 2:
            continue
        steps = []
        for sid in sections[:MAX_STEPS]:
            own = list(dict.fromkeys(s["chunk_id"] for s in cards[sid]["steps"]))[:PER_SECTION]
            steps += [dict(text=" > ".join(cards[sid]["headings"][-1:]), chunk_id=c) for c in own]
        if len({s["chunk_id"] for s in steps}) < 2:
            continue
        first = cards[sections[0]]
        out.append(dict(section_id=f"{tag}#T{len(out):03d}", document_id=first["document_id"],
                        product_id=first["product_id"], product_version=first["product_version"],
                        headings=["task"], goals=goals, steps=steps, needs=[], linked_needs=[], sections=sections))
    return out


def seed_links(seeds: list[str], section_of: dict[str, str], cards: dict[str, dict],
               tasks_by_section: dict[str, list[dict]], use=("card", "task")) -> list[str]:
    """Passages linked offline to the sections of the seed passages: the section's own card (its steps and
    resolved needs) and every task that lists the section. No question-to-card matching."""
    out: list[str] = []
    for seed in seeds:
        sid = section_of.get(seed)
        if sid is None:
            continue
        if "card" in use and sid in cards:
            out += card_chunks(cards[sid])
        if "task" in use:
            out += [c for task in tasks_by_section.get(sid, []) for c in card_chunks(task)]
    return [c for c in dict.fromkeys(out) if c not in seeds]


class OutlineIndex:
    """Cards with one embedding per goal, matched to a question within its product and version."""

    def __init__(self, path, encoder):
        import hashlib

        import numpy as np

        self.cards = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.cards = [c for c in self.cards if card_chunks(c)]
        goals = [(i, g) for i, c in enumerate(self.cards) for g in c["goals"]]
        self.owner = np.array([i for i, _ in goals], dtype=int)
        digest = hashlib.sha1("\n".join(g for _, g in goals).encode("utf-8")).hexdigest()[:16]
        cache = Path(path).with_suffix(f".{digest}.npy")
        if cache.exists():
            self.vectors = np.load(cache)
        else:
            self.vectors = np.array(encoder.embed_queries([g for _, g in goals]), dtype="float32")
            np.save(cache, self.vectors)

    def match(self, vector, product_id: str | None, versions, *, top_k: int = 1, min_score: float = 0.0):
        """[(card, cosine)] best first, one score per card (its best goal), at least `min_score`."""
        import numpy as np

        wanted = {str(v) for v in versions or [] if v not in (None, "", "None")}
        scores = self.vectors @ np.asarray(vector, dtype="float32")
        best: dict[int, float] = {}
        for owner, score in zip(self.owner, scores):
            card = self.cards[owner]
            if card["product_id"] != product_id or (wanted and str(card["product_version"]) not in wanted):
                continue
            if score >= min_score and score > best.get(owner, -1.0):
                best[int(owner)] = float(score)
        ranked = sorted(best.items(), key=lambda x: -x[1])[:top_k]
        return [(self.cards[i], s) for i, s in ranked]
