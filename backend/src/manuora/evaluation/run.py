"""Run the benchmark through the production chat path and write a scorecard.

    python -m manuora.evaluation.run --tier routing                        # router decisions only, seconds, no models
    python -m manuora.evaluation.run --tier retrieval                      # dev split, no LLM
    python -m manuora.evaluation.run --tier answer                         # also generates answers
    python -m manuora.evaluation.run --tier retrieval --set evidence.gate=relative --set evidence.relative_margin=8
    python -m manuora.evaluation.run --tier answer --split holdout --release   # release gate only (logged)
    python -m manuora.evaluation.run label                                 # label shown answers to negatives

Each case goes through manuora.api.app exactly as a chat request from the page would (routing, scope,
retrieval, rerank, evidence selection, packing, generation, grounding checks), in a fresh conversation
whose history is a throwaway database. `--set key=value` changes settings for this run only, the same
keys as config/assistant.yaml. Results: data/eval_runs/<run>/ (records.jsonl, scorecard.json, scorecard.md).

The routing tier replays each case's earlier turns into a fresh memory and scores only the router's decision
(action, product, versions, needs); it scores every category, including conversation cases. The retrieval
and answer tiers do not replay history yet, so they leave conversation and routing-only cases out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

from manuora.evaluation import bench

ROOT = bench.ROOT
RUNS = ROOT / "data/eval_runs"
LABELS = ROOT / "data/eval_labels.jsonl"
HOLDOUT_LOG = ROOT / "eval/holdout_log.jsonl"
STAGES = ("candidates", "reranked", "selected", "packed")
NOT_SCORED = {"conversation"}
# Replies that keep a negative question from being answered from the manuals.
DECLINED = {"abstain", "chat", "clarify"}


# --------------------------------------------------------------------------- settings
def parse_sets(pairs: list[str]) -> dict:
    """["evidence.max_passages=12", "grounding.term_check=null"] -> nested dict (values parsed as YAML)."""
    out: dict = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set needs key=value, got {pair!r}")
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


def _digest(value) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- one case
def request_for(case: dict) -> dict:
    """What the chat page would send: its selection (bench.ui_of) and the question."""
    return {"question": case["question"], **bench.ui_of(case)}


# --------------------------------------------------------------------------- routing tier
def make_router(conversation: dict, documents: list[dict] | None = None):
    """The decider the app would build from these conversation settings (generation/router.build_router)."""
    from manuora.generation.router import build_router

    return build_router(conversation, documents, ROOT)


def replay(case: dict, documents: list[dict], router, rules: str = "v1", assistant_name: str = "Manuora"):
    """Replay the case's earlier turns into a fresh memory; return decide(question) -> (context, decision).

    Each earlier turn is routed by the same router; a turn it would not answer (clarify, chat) does not
    become task memory, exactly as in the app. Answers are placeholders: only routing is measured.
    """
    from manuora.api.routing import early_response
    from manuora.generation.context import prepare_context, remember_reply
    from manuora.generation.router import chat_decision, checked, known_versions, route_with

    products = {d["product_id"] for d in documents}
    ui = bench.ui_of(case)
    state = {"memory": {}, "last": None, "earlier": []}

    def decide(question: str):
        if early_response(question, assistant_name, products, rules=rules) is not None:
            return None, chat_decision()
        context = prepare_context(question, ui, documents, state["memory"], thread_id=case["id"], rules=rules)
        context["last_displayed_reply"] = state["last"]
        context["ui"] = ui
        context["earlier"] = list(state["earlier"])
        context["catalog"] = known_versions(documents)  # what a learned router is told the page offers
        return context, checked(route_with(router, context), context, documents)

    for turn, earlier in enumerate(h for h in case.get("history") or [] if h.get("role") == "user"):
        context, decision = decide(earlier["content"])
        answer = f"(earlier answer {turn + 1})"
        kind = {"direct": "answer", "rewrite": "answer"}.get(decision.action, decision.action)
        if context is not None:
            state["memory"] = remember_reply(state["memory"], context, answer, [], turn_id=f"{case['id']}:{turn}",
                                             kind=kind)
        state["last"] = {"content": answer, "message_id": f"{case['id']}:{turn}", "sources": [], "truncated": False}
        state["earlier"].append(earlier["content"])
    return decide


def route_case(case: dict, documents: list[dict], router, rules: str = "v1", assistant_name: str = "Manuora") -> dict:
    """Route one case after replaying its history (see `replay`). No retrieval, no answer model."""
    decide = replay(case, documents, router, rules, assistant_name)
    tick = time.perf_counter()
    _, decision = decide(case["question"])
    ms = (time.perf_counter() - tick) * 1000
    return score_route(case, decision, ms)


def score_route(case: dict, decision, ms: float) -> dict:
    expected = bench.expected_route(case)
    got = decision.action
    record = {"id": case["id"], "category": case["category"], "expect": case["expect"], "private": case["_private"],
              "seconds": round(ms / 1000, 5), "ms": round(ms, 3), "action": got, "expected": sorted(expected["actions"]),
              "reason": decision.reason, "product": decision.product, "versions": decision.versions,
              "products": list(getattr(decision, "products", None) or []),
              "needs": len(decision.needs), "action_ok": got in expected["actions"],
              "false_clarify": got == "clarify" and "clarify" not in expected["actions"],
              "missed_clarify": "clarify" in expected["actions"] and got != "clarify",
              "wrong_direct": not expected["actions"] & {"direct", "rewrite"} and got in ("direct", "rewrite")}
    if expected["actions"] & {"direct", "rewrite"} and got in ("direct", "rewrite"):
        wanted = expected["product"]
        if isinstance(wanted, list):  # several products: every one, searched separately
            product_ok = set(getattr(decision, "products", None) or []) == set(wanted)
        else:
            product_ok = (decision.product or None) == wanted or wanted is None
        record["scope_ok"] = product_ok and (not expected["versions"] or set(decision.versions) == expected["versions"])
        record["needs_ok"] = (len(decision.needs) >= 2) == (expected["needs"] == "many")
    record["route_ok"] = record["action_ok"] and record.get("scope_ok", True)
    return record


def documents_for(effective: dict) -> list[dict]:
    """The (product, version, document) rows the page offers, from the BM25 index these settings use."""
    import sqlite3
    from manuora.api.app import resolve_path

    path = resolve_path(effective["sparse_index"])
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT DISTINCT product_id, product_version, document_id FROM chunks").fetchall()
    finally:
        db.close()
    return [{"product_id": p, "product_version": v, "document_id": d} for p, v, d in rows]


def evaluate_routing(cases: list[dict], effective: dict, progress=print) -> list[dict]:
    conversation = effective.get("conversation") or {}
    documents = documents_for(effective)
    router = make_router(conversation, documents)
    rules = conversation.get("context_rules", "v1")
    return [route_case(c, documents, router, rules, effective.get("assistant_name", "Manuora")) for c in cases]


def score_case(case: dict, body: dict, seconds: float, tier: str = "answer") -> dict:
    """Per-case record: which stages hold the gold, and how the reply ended."""
    trace = body.get("trace") or {}
    stages = trace.get("stages") or {}
    record = {"id": case["id"], "category": case["category"], "expect": case["expect"],
              "private": not bench.is_public(case), "outcome": trace.get("outcome"),
              "kind": body.get("kind"), "seconds": round(seconds, 2), "stage_ids": {
                  s: [p["chunk_id"] for p in stages.get(s, [])] for s in STAGES if s in stages}}
    if stages.get("reranked"):  # the best reranker score: how far above or below the gate the case sits
        record["top_score"] = stages["reranked"][0].get("rerank_score")
    if bench.has_gold(case) and case["expect"] == "answer":
        scores = {s: bench.evidence_score(case, stages.get(s, [])) for s in STAGES}
        record["complete"] = {s: bool(v["complete"]) for s, v in scores.items()}
        record["packed_recall"] = round(scores["packed"]["recall"], 3)
    if tier == "answer":
        answered = body.get("kind") == "answer"
        record["answered"] = answered
        record["answer"] = body.get("answer", "")
        if answered and record.get("complete") is not None:
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", body.get("answer", ""))}
            packed = stages.get("packed", [])
            record["cited_complete"] = bool(bench.evidence_score(case, [p for i, p in enumerate(packed, 1) if i in cited])["complete"])
        if case["expect"] in ("abstain", "premise"):
            record["declined"] = body.get("kind") in DECLINED
            if answered:
                record["label_key"] = label_key(case, body.get("answer", ""))
    return record


def label_key(case: dict, answer: str) -> str:
    return hashlib.sha1((case["id"] + "\n" + case["question"] + "\n" + answer).encode()).hexdigest()[:16]


def read_labels() -> dict[str, str]:
    return {r["key"]: r["label"] for r in bench.read_file(LABELS)}


# --------------------------------------------------------------------------- aggregation
def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def aggregate(records: list[dict], labels: dict[str, str] | None = None) -> dict:
    """Per-category numbers. Counts, not rates, so a change of one case is visible."""
    labels = labels or {}
    out: dict[str, dict] = {}
    groups = defaultdict(list)
    for r in records:
        groups[r["category"]].append(r)
        groups["all"].append(r)
    for category, rs in sorted(groups.items()):
        row: dict = {"n": len(rs)}
        gold = [r for r in rs if "complete" in r]
        if gold:
            row["with_gold"] = len(gold)
            for s in STAGES:
                row[f"{s}_complete"] = sum(r["complete"].get(s, False) for r in gold)
        answered = [r for r in rs if "answered" in r]
        if answered:
            row["answered"] = sum(r["answered"] for r in answered)
            row["abstained"] = sum(r["kind"] == "abstain" for r in answered)
            cited = [r for r in answered if "cited_complete" in r]
            if gold:
                row["cited_complete"] = sum(r["cited_complete"] for r in cited)
            negatives = [r for r in answered if "declined" in r]
            if negatives:
                shown = [r for r in negatives if not r["declined"]]
                row["declined"] = sum(r["declined"] for r in negatives)
                row["shown"] = len(shown)
                row["made_up"] = sum(labels.get(r["label_key"]) == "made_up" for r in shown)
                row["unlabelled"] = sum(r["label_key"] not in labels for r in shown)
        routed = [r for r in rs if "action_ok" in r]
        if routed:
            for key in ("route_ok", "action_ok", "false_clarify", "missed_clarify", "wrong_direct"):
                row[key] = sum(r[key] for r in routed)
            for key in ("scope_ok", "needs_ok"):
                scoped = [r for r in routed if key in r]
                if scoped:
                    row[key] = sum(r[key] for r in scoped)
            ms = [r["ms"] for r in routed]
            row["p50_ms"], row["p95_ms"] = _p(ms, 0.5), _p(ms, 0.95)
        else:
            seconds = [r["seconds"] for r in rs]
            row["p50_seconds"], row["p95_seconds"] = _p(seconds, 0.5), _p(seconds, 0.95)
        out[category] = row
    return out


def scorecard(records: list[dict], meta: dict) -> dict:
    labels = read_labels()
    return {"meta": meta,
            "public": aggregate([r for r in records if not r["private"]], labels),
            "private": aggregate([r for r in records if r["private"]], labels)}


COLUMNS = ("n", "with_gold", "candidates_complete", "reranked_complete", "selected_complete", "packed_complete",
           "answered", "cited_complete", "abstained", "declined", "shown", "made_up", "unlabelled",
           "route_ok", "action_ok", "false_clarify", "missed_clarify", "wrong_direct", "scope_ok", "needs_ok",
           "p50_seconds", "p95_seconds", "p50_ms", "p95_ms")


def markdown(card: dict) -> str:
    meta = card["meta"]
    lines = [f"# Scorecard: {meta['name']}", "",
             f"- tier `{meta['tier']}`, split `{meta['split']}`, {meta['cases']} cases, {meta['started']}",
             f"- commit `{meta['git']['commit']}`{' (uncommitted changes)' if meta['git']['dirty'] else ''}, "
             f"settings `{meta['settings_digest']}`, bench `{meta['bench_digest']}`",
             f"- overrides: `{json.dumps(meta['overrides'])}`" if meta["overrides"] else "- overrides: none",
             f"- not scored: {', '.join(sorted(meta['not_scored'])) or 'none'}", ""]
    for section in ("public", "private"):
        rows = card[section]
        if not rows:
            continue
        cols = [c for c in COLUMNS if any(c in r for r in rows.values())]
        lines += [f"## {section}", "", "| category | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        lines += [f"| {cat} | " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |"
                  for cat, r in rows.items()]
        lines.append("")
    if meta["tier"] == "routing":
        lines += ["`route_ok`: right action and, for questions that reach retrieval, the right product and versions. "
                  "`false_clarify`: asked for clarification although the question should be searched. `missed_clarify`: "
                  "should have asked. `wrong_direct`: searched a question that should be clarified or edited. "
                  "`scope_ok`/`needs_ok`: of the questions searched, right scope / right number of needs (one vs several)."]
    else:
        lines += ["`*_complete`: cases whose gold evidence is all present after that stage. `cited_complete`: the answer's "
                  "citations alone cover the gold. `declined`: a negative that was not answered from the manuals. "
                  "`made_up`/`unlabelled`: shown answers to negatives, by your labels (`python -m manuora.evaluation.run label`)."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- run
def _reset_app(api) -> None:
    for cached in (api.settings, api.conversation_generator, api.pipeline, api.version_index, api.outline_index):
        cached.cache_clear()


def effective_settings(overrides: dict) -> dict:
    """The settings the app would use with these overrides."""
    from manuora.api import app as api

    api.OVERRIDES = overrides
    api.settings.cache_clear()
    try:
        return api.settings()
    finally:
        api.OVERRIDES = {}
        api.settings.cache_clear()


def evaluate(cases: list[dict], tier: str, overrides: dict, progress=print) -> list[dict]:
    """Every case as a fresh chat request; conversations go to a throwaway database."""
    from fastapi import HTTPException
    from manuora.api import app as api

    scratch = Path(tempfile.mkdtemp(prefix="manuora_eval_"))
    conversation = {**(overrides.get("conversation") or {}), "db_path": str(scratch / "conversations.sqlite")}
    api.OVERRIDES = {**overrides, "conversation": conversation}
    _reset_app(api)
    records = []
    try:
        for i, case in enumerate(cases, 1):
            request = api.ChatRequest(**request_for(case), conversation_id=uuid4(), debug=True,
                                      retrieval_only=tier == "retrieval")
            tick = time.perf_counter()
            try:
                body = api._chat(request)
            except HTTPException as exc:  # e.g. the question does not fit the context budget
                body = {"kind": "error", "answer": "", "trace": {"outcome": f"error: {exc.detail}"}}
            records.append(score_case(case, body, time.perf_counter() - tick, tier))
            if progress and (i % 10 == 0 or i == len(cases)):
                progress(f"  {i}/{len(cases)}")
    finally:
        if api.conversation_generator.cache_info().currsize:
            api.conversation_generator().close()
        api.OVERRIDES = {}
        _reset_app(api)
    return records


def select_cases(tier: str, split: str, private: bool, filters: dict, limit: int | None,
                 drafts: bool = False) -> tuple[list[dict], list[str]]:
    """(cases this tier scores, categories it leaves out). The routing tier scores every category;
    the other tiers leave out conversation cases (no history replay yet) and routing-only cases."""
    statuses = bench.SCORED | ({"draft"} if drafts else set())
    cases = bench.load(split, private=private, statuses=statuses, **filters)
    if tier == "routing":
        return cases[:limit], []
    skipped = sorted({c["category"] for c in cases if c["category"] in NOT_SCORED})
    keep = [c for c in cases if c["category"] not in NOT_SCORED and "routing_only" not in (c.get("tags") or [])]
    return keep[:limit], skipped


def run(tier: str, split: str, overrides: dict, name: str | None, release: bool, private: bool,
        filters: dict, limit: int | None, drafts: bool = False) -> Path:
    if split == "holdout" and not release:
        raise SystemExit("The holdout is for release checks only: add --release (every holdout run is logged).")
    cases, skipped = select_cases(tier, split, private, filters, limit, drafts)
    if not cases:
        raise SystemExit("no cases match")
    effective = effective_settings(overrides)
    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    name = (name or (_digest(overrides) if overrides else "production")) + ("-drafts" if drafts else "")
    out = RUNS / f"{datetime.now():%Y%m%d_%H%M%S}_{tier}_{split}_{name}"
    out.mkdir(parents=True, exist_ok=False)
    print(f"{len(cases)} cases -> {out.relative_to(ROOT)}")
    if tier == "answer":
        records = _with_model(effective, overrides, lambda: evaluate(cases, tier, overrides))
    elif tier == "routing":
        records = evaluate_routing(cases, effective)
    else:
        records = evaluate(cases, tier, overrides)
    meta = {"name": name, "tier": tier, "split": split, "cases": len(records), "started": started,
            "git": git_state(), "overrides": overrides, "settings_digest": _digest(effective),
            "bench_digest": _digest(sorted(c["id"] + c["status"] for c in cases)), "not_scored": skipped,
            "dense_collection": effective.get("dense_collection"), "sparse_index": effective.get("sparse_index"),
            "generation_model": effective["generation"]["model"], "reranker": effective["reranker"]["model_id"]}
    bench.write_file(out / "records.jsonl", records)
    card = scorecard(records, meta)
    (out / "scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "scorecard.md").write_text(markdown(card), encoding="utf-8")
    if split == "holdout":
        HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with HOLDOUT_LOG.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"date": started, "tier": tier, "run": out.name, "commit": meta["git"]["commit"],
                                  "overrides": overrides, "cases": len(records)}) + "\n")
    print(markdown(card))
    return out


def _with_model(effective: dict, overrides: dict, work):
    """Run answer generation with the chat model at the context size these settings expect.

    Production context: load the model if needed and unload it afterwards only if this run loaded it.
    A different context (an arm that sets conversation.context_window_tokens): reconfigure Docker Model
    Runner for the run and restore it after (manuora.evaluation.common.model_context), which refuses
    while the live app has the model loaded.
    """
    from manuora.evaluation.common import model_context
    from manuora.generation.model_control import is_loaded, load_model, unload_model

    generation = effective["generation"]
    if (overrides.get("conversation") or {}).get("context_window_tokens"):
        with model_context(generation["model"], int(effective["conversation"]["context_window_tokens"])):
            return work()
    loaded_here = not is_loaded(generation["model"])
    if loaded_here:
        load_model(generation["model"], generation["url"])
    try:
        return work()
    finally:
        if loaded_here:
            unload_model(generation["model"])


# --------------------------------------------------------------------------- labels
def label_shown(run_dir: Path | None) -> None:
    """Ask for a label for every shown answer to a negative that has none yet (each answer once, ever)."""
    run_dir = run_dir or max((p for p in RUNS.glob("*_answer_*") if (p / "records.jsonl").exists()), default=None)
    if run_dir is None:
        raise SystemExit("no answer-tier run found")
    labels = read_labels()
    cases = {c["id"]: c for c in bench.load(split=None, statuses=None)}
    todo = [r for r in bench.read_file(run_dir / "records.jsonl") if r.get("label_key") and r["label_key"] not in labels]
    print(f"{len(todo)} unlabelled shown answer(s) to negatives in {run_dir.name}")
    for r in todo:
        case = cases[r["id"]]
        print("\n" + "=" * 80 + f"\n{r['id']} ({case['product_id']} {', '.join(case['versions'])})\nQ: {case['question']}")
        if case.get("expected_behavior"):
            print(f"expected: {case['expected_behavior']}")
        print(f"\nA: {r['answer']}\n")
        choice = input("[m]ade up / [o]k (grounded, e.g. a correct premise correction) / [s]kip / [q]uit: ").strip().lower()
        if choice.startswith("q"):
            break
        if choice[:1] in ("m", "o"):
            LABELS.parent.mkdir(parents=True, exist_ok=True)
            with LABELS.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": r["label_key"], "case": r["id"], "label": "made_up" if choice[0] == "m" else "ok",
                                    "date": datetime.now().strftime("%Y-%m-%d")}) + "\n")
    card_path = run_dir / "scorecard.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card = scorecard(bench.read_file(run_dir / "records.jsonl"), card["meta"])
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (run_dir / "scorecard.md").write_text(markdown(card), encoding="utf-8")


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["label"]:
        label_shown(Path(argv[1]) if len(argv) > 1 else None)
        return
    parser = argparse.ArgumentParser(prog="python -m manuora.evaluation.run")
    parser.add_argument("--tier", choices=("routing", "retrieval", "answer"), required=True)
    parser.add_argument("--split", choices=sorted(bench.SPLITS), default="dev")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--name", help="short label for the run folder (default: production or a digest of --set)")
    parser.add_argument("--category", choices=sorted(bench.CATEGORIES))
    parser.add_argument("--product")
    parser.add_argument("--public-only", action="store_true", help="leave out the local-only cases")
    parser.add_argument("--limit", type=int, help="first N cases (smoke test)")
    parser.add_argument("--release", action="store_true", help="required for --split holdout")
    parser.add_argument("--drafts", action="store_true",
                        help="also score draft cases (exploration only; never promote such a run)")
    args = parser.parse_args(argv)
    filters = {k: v for k, v in (("category", args.category), ("product_id", args.product)) if v}
    run(args.tier, args.split, parse_sets(args.sets), args.name, args.release, not args.public_only, filters,
        args.limit, args.drafts)


if __name__ == "__main__":
    main()
