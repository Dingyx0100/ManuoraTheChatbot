"""The Manuora benchmark: one maintained set of questions with gold evidence.

Two files, same schema (one JSON object per line):
- eval/bench/cases.jsonl           tracked in git; public manuals only (PUBLIC_PRODUCTS)
- data/bench_private/cases.jsonl   local only; manuals that must not be published

A case:
    id          globally unique, never reused (e.g. "nc-w014")
    category    single | workflow | version | negative | conversation
    split       dev (experiments, tuning) | holdout (release gate only)
    status      draft (gold unresolved) | checked (gold anchors verified by tooling)
                | reviewed (a person checked it) | retired (kept for history, never scored)
    product_id, versions, question, history (conversation turns, optional)
    expect      answer | abstain | premise (abstain or correct the question's assumption) | clarify | edit
    gold        {"span_groups": [[{"id": chunk_id, "quote": str}, ...], ...]}   every group needs one anchor
                or {"required_facts": [...], "accepted_alternatives": [...]}   version facts (quote + version)
                or null (negatives)
    absence_checks  [{"find": [regex], "matches": 0}] for negatives
    tags, source (legacy file/id it came from), added (date), expected_behavior (free text, optional)
    ui          optional {"product_id", "product_version"}: what the chat page had selected. Without it, single-turn
                cases select the case's product (and its version when there is exactly one); cases with history
                select nothing, so scope must come from the conversation.
    route       optional expected routing decision, overriding the one derived from the fields above:
                {"action": "direct" | "rewrite" | "clarify" | "edit" | "chat" (or a list of acceptable actions),
                 "product": id or null, "versions": [...]}
Cross-product conversation cases (scope comes only from history, or a deliberate conflict) use product_id "".
Tag "routing_only" marks a case that only the routing tier scores (it needs no gold).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from manuora.evaluation.coverage import quote_coverage, span_coverage

ROOT = Path(__file__).resolve().parents[4]
PUBLIC = ROOT / "eval/bench/cases.jsonl"
PRIVATE = ROOT / "data/bench_private/cases.jsonl"

PUBLIC_PRODUCTS = {"nextcloud", "qgis", "kicad", "freecad"}
CATEGORIES = {"single", "workflow", "version", "negative", "conversation"}
SPLITS = {"dev", "holdout"}
STATUSES = {"draft", "checked", "reviewed", "retired"}
SCORED = {"checked", "reviewed"}
EXPECTS = {"answer", "abstain", "premise", "clarify", "edit"}
REQUIRED = ("id", "category", "split", "status", "product_id", "versions", "question", "expect")
ACTIONS = {"direct", "rewrite", "clarify", "edit", "chat"}


def is_public(case: dict) -> bool:
    """Public manuals, or a cross-product case ("") that names only public products."""
    return case["product_id"] in PUBLIC_PRODUCTS or case["product_id"] == ""


def versions_of(case: dict) -> list[str]:
    return [str(v) for v in case.get("versions") or [] if v not in ("None", "", None)]


def ui_of(case: dict) -> dict:
    """What the chat page had selected for this case (see the module docstring)."""
    if "ui" in case:
        return {k: case["ui"].get(k) for k in ("product_id", "product_version")}
    if case.get("history"):
        return {"product_id": None, "product_version": None}
    versions = versions_of(case)
    return {"product_id": case["product_id"] or None, "product_version": versions[0] if len(versions) == 1 else None}


def expected_route(case: dict) -> dict:
    """{"actions": set of acceptable actions, "product", "versions" (set), "needs": "one" | "many"}.

    Derived unless the case gives `route`: a question that should be answered, declined or corrected must
    reach retrieval (direct; rewrite when it continues earlier turns), clarify and edit cases expect those.
    """
    given = case.get("route") or {}
    if "action" in given:
        actions = set(given["action"] if isinstance(given["action"], list) else [given["action"]])
    elif case["expect"] in ("clarify", "edit"):
        actions = {case["expect"]}
    elif case.get("history"):
        standalone = {"explicit_switch", "standalone_after_topic"} & set(case.get("tags") or [])
        actions = {"direct"} if standalone else {"rewrite"}
    else:
        actions = {"direct"}
    groups = len((case.get("gold") or {}).get("span_groups") or [])
    return {"actions": actions,
            "product": given.get("product", case["product_id"] or None),
            "versions": set(given.get("versions", versions_of(case))),
            "needs": given.get("needs") or ("many" if case["category"] == "workflow" and groups >= 2 else "one")}


def read_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_file(path: Path, cases: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")


def load(split: str | None = "dev", private: bool = True, statuses=SCORED, **filters) -> list[dict]:
    """Scorable cases. filters match case fields exactly, e.g. category="workflow", product_id="qgis"."""
    cases = read_file(PUBLIC) + (read_file(PRIVATE) if private else [])
    for case in cases:
        case["_private"] = not is_public(case)
    return [c for c in cases
            if (split is None or c["split"] == split) and (statuses is None or c["status"] in statuses)
            and all(c.get(k) == v for k, v in filters.items())]


def validate(cases: list[dict], public: bool) -> list[str]:
    """Schema problems; an empty list means the file is valid."""
    errors = []
    for i, c in enumerate(cases, 1):
        where = f"line {i} ({c.get('id', '?')})"
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            errors.append(f"{where}: missing {missing}")
            continue
        for key, allowed in (("category", CATEGORIES), ("split", SPLITS), ("status", STATUSES), ("expect", EXPECTS)):
            if c[key] not in allowed:
                errors.append(f"{where}: {key}={c[key]!r} not in {sorted(allowed)}")
        if public and not is_public(c):
            errors.append(f"{where}: product {c['product_id']!r} belongs in the private bench file")
        if c["product_id"] == "" and c["category"] != "conversation":
            errors.append(f"{where}: only conversation cases may leave product_id empty")
        if not isinstance(c["versions"], list):
            errors.append(f"{where}: versions must be a list")
        route = c.get("route") or {}
        actions = route.get("action", [])
        if any(a not in ACTIONS for a in (actions if isinstance(actions, list) else [actions])):
            errors.append(f"{where}: route.action must be among {sorted(ACTIONS)}")
        gold = c.get("gold")
        routing_only = "routing_only" in (c.get("tags") or [])
        if c["expect"] == "answer" and c["status"] in SCORED and not gold and not routing_only:
            errors.append(f"{where}: an answerable scored case needs gold (or tag routing_only)")
        if gold and not (gold.get("span_groups") or gold.get("required_facts")):
            errors.append(f"{where}: gold needs span_groups or required_facts")
    for case_id, n in Counter(c.get("id") for c in cases).items():
        if n > 1:
            errors.append(f"duplicate id {case_id!r} ({n}x)")
    return errors


def has_gold(case: dict) -> bool:
    return bool(case.get("gold"))


def evidence_score(case: dict, passages: list[dict]) -> dict:
    """{"complete", "recall"} of passages ({chunk_id, text, product_version}) against the case's gold."""
    gold = case["gold"]
    if gold.get("span_groups"):
        return span_coverage(gold, passages)
    return quote_coverage(gold, passages)


# --------------------------------------------------------------------------- maintenance CLI
def _file_of(case_id: str) -> Path:
    for path in (PUBLIC, PRIVATE):
        if any(c["id"] == case_id for c in read_file(path)):
            return path
    raise SystemExit(f"unknown case id {case_id!r}")


def mark(status: str, ids: list[str], note: str | None = None) -> None:
    """Set the status of cases in whichever file holds them (reviewed after you checked them, retired to drop)."""
    if status not in STATUSES:
        raise SystemExit(f"status must be one of {sorted(STATUSES)}")
    for path in {_file_of(i) for i in ids}:
        cases = read_file(path)
        for c in cases:
            if c["id"] in ids:
                c["status"] = status
                if note:
                    c["review_note"] = note
        write_file(path, cases)
    print(f"{len(ids)} case(s) -> {status}")


def stats() -> str:
    cases = read_file(PUBLIC) + read_file(PRIVATE)
    table = Counter((c["split"], c["category"], c["status"], is_public(c)) for c in cases)
    lines = ["split    category      status    public  n"]
    lines += [f"{s:8} {cat:13} {st:9} {'yes' if pub else 'no':6}  {n}" for (s, cat, st, pub), n in sorted(table.items())]
    return "\n".join(lines)


def _chunk_texts(ids: set[str]) -> dict[str, dict]:
    import sqlite3
    import yaml

    config = yaml.safe_load((ROOT / "config/assistant.yaml").read_text(encoding="utf-8"))
    path = ROOT / config["sparse_index"]
    if not ids or not path.is_file():
        return {}
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    out = {}
    for chunk_id, headings, text in db.execute(
            f"select chunk_id, headings, text from chunks where chunk_id in ({','.join('?' * len(ids))})", sorted(ids)):
        out[chunk_id] = {"headings": " / ".join(json.loads(headings or "[]")), "text": text}
    return out


def review_page(out: Path, split: str | None = None, category: str | None = None, limit: int = 40) -> int:
    """One HTML page of cases nobody has reviewed yet, with their gold passages, and the command to mark them."""
    from html import escape

    cases = [c for c in read_file(PUBLIC) + read_file(PRIVATE)
             if c["status"] in ("draft", "checked") and (split is None or c["split"] == split)
             and (category is None or c["category"] == category)][:limit]
    anchors = {a["id"] for c in cases for g in (c.get("gold") or {}).get("span_groups", []) for a in g}
    texts = _chunk_texts(anchors)
    blocks = []
    for c in cases:
        gold = c.get("gold") or {}
        parts, labels = [], gold.get("step_labels") or []
        for n, group in enumerate(gold.get("span_groups", []), 1):
            label = labels[n - 1] if n <= len(labels) else None
            options = []
            for a in group:
                t = texts.get(a["id"], {})
                options.append(f"<div class=opt><code>{escape(a['id'])}</code> <span class=h>{escape(t.get('headings', ''))}</span>"
                               + (f"<div class=q>“{escape(a['quote'])}”</div>" if a.get("quote") else "")
                               + f"<div class=t>{escape((t.get('text') or '(chunk not in the current index)')[:700])}</div></div>")
            parts.append(f"<li>{escape(label or f'requirement {n}')}{''.join(options)}</li>")
        for f in gold.get("required_facts", []):
            quotes = "; ".join(f"v{e['version']}: “{e['quote']}”" for e in f["evidence"])
            parts.append(f"<li>{escape(f.get('fact', f['fact_id']))}<div class=q>{escape(quotes)}</div></li>")
        if c.get("absence_checks"):
            parts.append("<li>must not be in the manual: " + escape(", ".join(p for a in c["absence_checks"] for p in a["find"])) + "</li>")
        history = "".join(f"<div class=hist><b>{escape(h['role'])}:</b> {escape(h['content'])}</div>" for h in c.get("history", []))
        blocks.append(f"<section><label><input type=checkbox value='{escape(c['id'])}'> <b>{escape(c['id'])}</b></label> "
                      f"<span class=meta>{escape(c['category'])} · {escape(c['split'])} · {escape(c['product_id'] or 'any product')} "
                      f"{escape(', '.join(c['versions']))} · expect {escape(c['expect'])}"
                      + (f" · route {escape(json.dumps(c['route']))}" if c.get("route") else "")
                      + (f" · page selection {escape(json.dumps(c['ui']))}" if c.get("ui") else "")
                      + f" · {escape(c.get('source', ''))}</span>"
                      f"{history}<p class=question>{escape(c['question'])}</p>"
                      + (f"<p class=meta>{escape(c['expected_behavior'])}</p>" if c.get("expected_behavior") else "")
                      + f"<ol>{''.join(parts)}</ol></section>")
    page = f"""<!doctype html><meta charset=utf-8><title>Bench review</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#222}}
section{{border-top:1px solid #ddd;padding:1rem 0}} .meta{{color:#666;font-size:13px}} .question{{font-size:17px;margin:.4rem 0}}
.opt{{margin:.3rem 0 .6rem}} .h{{color:#666;font-size:13px}} .q{{background:#fff6d5;padding:2px 4px}} .t{{color:#444;font-size:13px;white-space:pre-wrap}}
.hist{{color:#555;font-size:14px}} #bar{{position:sticky;top:0;background:#f4f4f4;padding:.6rem;border-radius:6px}} code{{font-size:12px}}
</style><div id=bar>{len(cases)} unreviewed case(s). Tick the ones that are right, then run:<br>
<code id=cmd>python -m manuora.evaluation.bench mark reviewed</code> <button onclick="navigator.clipboard.writeText(document.getElementById('cmd').textContent)">copy</button></div>
{''.join(blocks)}
<script>document.addEventListener('change',()=>{{const ids=[...document.querySelectorAll('input:checked')].map(i=>i.value);
document.getElementById('cmd').textContent='python -m manuora.evaluation.bench mark reviewed '+ids.join(' ');}});</script>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return len(cases)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m manuora.evaluation.bench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats", help="case counts by split, category and status")
    sub.add_parser("check", help="validate both files")
    review = sub.add_parser("review", help="write an HTML page of unreviewed cases")
    review.add_argument("--split", choices=sorted(SPLITS))
    review.add_argument("--category", choices=sorted(CATEGORIES))
    review.add_argument("--limit", type=int, default=40)
    review.add_argument("--out", type=Path, default=ROOT / "data/bench_review.html")
    marker = sub.add_parser("mark", help="set the status of cases")
    marker.add_argument("status", choices=sorted(STATUSES))
    marker.add_argument("ids", nargs="+")
    marker.add_argument("--note")
    args = parser.parse_args(argv)
    if args.command == "stats":
        print(stats())
    elif args.command == "check":
        errors = validate(read_file(PUBLIC), public=True) + validate(read_file(PRIVATE), public=False)
        print("\n".join(errors) or "ok")
        raise SystemExit(1 if errors else 0)
    elif args.command == "review":
        n = review_page(args.out, args.split, args.category, args.limit)
        print(f"{n} case(s) -> {args.out}")
    else:
        mark(args.status, args.ids, args.note)


if __name__ == "__main__":
    main()
