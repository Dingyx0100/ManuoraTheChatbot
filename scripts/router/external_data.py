"""Public datasets as routing examples (experiment R, mix arm), with their own splits kept apart.

    python scripts/router/external_data.py --out external_v0_1 --catalog data/router/fictional_v0_1

Sources (data/router/external/, downloaded by hand, licences checked 2026-11-20):
- CANARD (CC BY-SA 4.0; from QuAC): question-in-context rewrites.
- QReCC (CC BY-SA 3.0), Natural Questions conversations only: the TREC CAsT part traces back to MS MARCO
  (non-commercial terms) and the QuAC part repeats CANARD's dialogs.
- CLINC150 (CC BY 3.0), small-talk intents only: its other intents and out-of-scope queries are general
  requests that the manual-derived data would label differently, which would teach a contradiction.

Mapping (one turn per conversation, conversations assigned to one role at random):
- rewrite: a follow-up whose reference rewrite differs from it; `query` is the rewrite. Kept only when the
  rewrite adds no content word missing from the message and the two earlier user messages the router sees.
- direct (first): the rewrite of a first question, alone. direct (follow-up): a later question that already
  equals its rewrite, with its history (a self-contained question in an ongoing chat).
- clarify: a short follow-up with a reference word ("it", "he", "that") whose rewrite adds a name, sent with no
  history: nothing to resolve it against.
- chat: CLINC150 small talk.
Splits: each dataset's train -> train; its dev -> val (QReCC has none: 10% of train conversations by hash);
its test -> test_external. Caps per source and role keep any one source or action from dominating.
Training and validation examples carry a catalog of invented products plus Nextcloud, so "no product" isn't
learned as a property of one catalog; test_external uses the library's catalog (none stored).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_data as g  # noqa: E402
from manuora.evaluation import bench  # noqa: E402

EXTERNAL = g.OUT / "external"
SMALL_TALK = {"greeting", "goodbye", "thank_you", "tell_joke", "where_are_you_from", "how_old_are_you",
              "what_is_your_name", "who_made_you", "what_can_i_ask_you", "what_are_your_hobbies", "do_you_have_pets",
              "are_you_a_bot", "meaning_of_life", "who_do_you_work_for", "fun_fact"}
CAPS = {  # (source, role): (train, val, test)
    ("canard", "rewrite"): (300, 35, 150), ("canard", "direct_first"): (60, 8, 40),
    ("canard", "direct_followup"): (80, 10, 40), ("canard", "clarify"): (120, 15, 60),
    ("qrecc_nq", "rewrite"): (150, 18, 80), ("qrecc_nq", "direct_first"): (60, 8, 20),
    ("qrecc_nq", "direct_followup"): (40, 5, 20), ("qrecc_nq", "clarify"): (80, 10, 30),
    ("clinc150", "chat"): (150, 18, 100),
}
PRONOUN = re.compile(r"\b(he|she|it|they|him|her|them|his|its|their|this|that|these|those)\b", re.I)
NAME = re.compile(r"\b[A-Z][a-z]+")
ABOUT_SOURCE = re.compile(r"\b(article|section|passage|paragraph)\b", re.I)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.casefold()).strip()


def roles(question: str, rewrite: str, earlier: list[str], first: bool) -> list[str]:
    """Which roles this turn can play (a conversation then gets one of them)."""
    if first:
        # CANARD's rewrites of first questions sometimes name the source ("Does the article about X say ...").
        return [] if ABOUT_SOURCE.search(rewrite) else ["direct_first"]
    if norm(question) == norm(rewrite):
        return ["direct_followup"]
    out = []
    if not g.added_terms(rewrite, earlier[-2:] + [question], ""):
        out.append("rewrite")
    added_names = set(NAME.findall(rewrite)) - set(NAME.findall(question))
    if PRONOUN.search(question) and len(question.split()) <= 12 and added_names:
        out.append("clarify")
    return out


def example(source: str, role: str, question: str, rewrite: str, earlier: list[str], split: str, key: str) -> dict:
    action = {"direct_first": "direct", "direct_followup": "direct"}.get(role, role)
    text = rewrite if role == "direct_first" else question
    history = [{"role": "user", "content": m} for m in earlier] if role in ("rewrite", "direct_followup") else []
    return {"question": text.strip(), "history": history, "seed": [],
            "label": g.label(action, None, [], [], rewrite.strip() if action == "rewrite" else None),
            "ui": {"product_id": None, "product_version": None}, "scenario": f"ext_{source}_{role}",
            "style": "external", "split": split, "source": source, "source_key": key}


def conversations(source: str):
    """(split, conversation id, [turns]) with each turn (question, rewrite, earlier user messages, first)."""
    if source == "canard":
        for name, split in (("train", "train"), ("dev", "val"), ("test", "test_external")):
            rows = json.loads((EXTERNAL / "canard/CANARD_Release" / f"{name}.json").read_text(encoding="utf-8"))
            by_dialog = defaultdict(list)
            for r in rows:
                history = r["History"] if isinstance(r["History"], list) else ast.literal_eval(r["History"])
                earlier = history[2::2]  # after title and section: question, answer, question, ...
                by_dialog[r["QuAC_dialog_id"]].append((r["Question"], r["Rewrite"], earlier, int(r["Question_no"]) == 1))
            for dialog, turns in by_dialog.items():
                yield split, dialog, turns
    elif source == "qrecc_nq":
        for name in ("train", "test"):
            rows = json.loads((EXTERNAL / "qrecc" / f"qrecc_{name}.json").read_text(encoding="utf-8"))
            by_conv = defaultdict(list)
            for r in rows:
                if r["Conversation_source"] != "nq":
                    continue
                context = r["Context"] if isinstance(r["Context"], list) else ast.literal_eval(r["Context"] or "[]")
                by_conv[r["Conversation_no"]].append((r["Question"], r["Rewrite"] or r["Question"], context[0::2],
                                                      int(r["Turn_no"]) == 1))
            for conv, turns in by_conv.items():
                split = "test_external" if name == "test" else (
                    "val" if int(hashlib.sha1(str(conv).encode()).hexdigest(), 16) % 10 == 0 else "train")
                yield split, f"{name}-{conv}", turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--catalog", required=True, help="fictional_products.py output folder")
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    from manuora.evaluation.common import settings
    from manuora.generation.router import known_versions

    fictional = json.loads((Path(args.catalog) / "catalog.json").read_text(encoding="utf-8"))["catalog"]
    nextcloud = known_versions(g.documents(g.ROOT / settings()["sparse_index"])).get("nextcloud", [])
    catalog = {**fictional, "nextcloud": nextcloud}

    pools: dict = defaultdict(list)
    for source in ("canard", "qrecc_nq"):
        for split, conv, turns in conversations(source):
            by_role = defaultdict(list)
            for t in turns:
                for role in roles(*t):
                    by_role[role].append(t)
            if not by_role:
                continue
            # A role first (evenly among those this conversation offers), then a turn: picking a turn first would
            # favour clarify, which most follow-ups qualify for.
            role = rng.choice(sorted(by_role))
            question, rewrite, earlier, _ = rng.choice(by_role[role])
            pools[(source, role, split)].append(example(source, role, question, rewrite, earlier, split, f"{source}:{conv}"))
    clinc = json.loads((EXTERNAL / "clinc150/data_full.json").read_text(encoding="utf-8"))
    for name, split in (("train", "train"), ("val", "val"), ("test", "test_external")):
        for i, (text, intent) in enumerate(clinc[name]):
            if intent in SMALL_TALK:
                pools[("clinc150", "chat", split)].append({
                    "question": text, "history": [], "seed": [], "label": g.label("chat"),
                    "ui": {"product_id": None, "product_version": None}, "scenario": "ext_clinc150_chat",
                    "style": "external", "split": split, "source": "clinc150", "source_key": f"clinc150:{name}:{i}"})

    rows, counts = [], Counter()
    for (source, role), caps in CAPS.items():
        for split, cap in zip(("train", "val", "test_external"), caps):
            pool = pools[(source, role, split)]
            chosen = rng.sample(pool, min(cap, len(pool)))
            for e in chosen:
                if split != "test_external":
                    e["catalog"] = catalog
            rows += chosen
            counts[f"{source}/{role}/{split}"] = len(chosen)
    # Nothing that also appears in the benchmark (any split) or twice.
    rows, dropped = g.leak_filter(rows)
    seen, unique = set(), []
    for e in rows:
        if (e["question"].casefold(), e["split"]) not in seen:
            seen.add((e["question"].casefold(), e["split"]))
            unique.append(e)
    for i, e in enumerate(unique, 1):
        e["id"] = f"{args.out}-{i:05d}"
    out = g.OUT / args.out
    out.mkdir(parents=True, exist_ok=False)
    bench.write_file(out / "examples.jsonl", unique)
    summary = {"examples": len(unique), "dropped_as_leak": len(dropped), "by_cell": dict(sorted(counts.items())),
               "available": {f"{s}/{r}/{sp}": len(v) for (s, r, sp), v in sorted(pools.items())},
               "by_split": dict(Counter(e["split"] for e in unique)),
               "by_action": {sp: dict(Counter(e["label"]["action"] for e in unique if e["split"] == sp))
                             for sp in ("train", "val", "test_external")}}
    (out / "run.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("examples", "dropped_as_leak", "by_split", "by_action")}, indent=1))


if __name__ == "__main__":
    main()
