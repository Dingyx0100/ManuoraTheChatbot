"""Spot-review synthetic routing examples (experiment R).

    python scripts/router/review_page.py page data/router/pilot_v0_1 [--sample 100]   # writes review.html there
    python scripts/router/review_page.py mark data/router/pilot_v0_1 <ids> [--note "..."]   # record bad examples

The page shows each example's conversation and the label its scenario gave it. Tick the ones that are wrong
(label wrong, or the wording is not something a user would write); the copy button gives the `mark` command.
`mark` appends to review.jsonl; `report` prints the bad share per scenario (fix a template above 10%).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.is_file() else []


def stratified(examples: list[dict], n: int, seed: int = 3) -> list[dict]:
    """About n examples, spread evenly over scenario x split."""
    groups = defaultdict(list)
    for e in examples:
        groups[(e["scenario"], e["split"])].append(e)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    picked, i = [], 0
    while len(picked) < min(n, len(examples)):
        for g in groups.values():
            if i < len(g) and len(picked) < n:
                picked.append(g[i])
        i += 1
    return sorted(picked, key=lambda e: (e["scenario"], e["split"], e["id"]))


def page(folder: Path, sample: int) -> Path:
    examples = read(folder / "examples.jsonl")
    shown = stratified(examples, sample)
    blocks = []
    for e in shown:
        lab = e["label"]
        scope = " ".join(filter(None, [lab.get("product"), ", ".join(lab.get("versions") or [])])) or "none"
        ui = e.get("ui") or {}
        page_sel = " ".join(filter(None, [ui.get("product_id"), ui.get("product_version")])) or "nothing selected"
        history = "".join(f"<div class=hist>earlier: {escape(h['content'])}</div>" for h in e.get("history") or [])
        needs = "".join(f"<li>{escape(n)}</li>" for n in lab.get("needs") or [])
        query = f"<div class=q>standalone: {escape(lab['query'])}</div>" if lab.get("query") else ""
        blocks.append(
            f"<section><label><input type=checkbox value='{escape(e['id'])}'> <b>{escape(e['id'])}</b></label> "
            f"<span class=meta>{escape(e['scenario'])} · {escape(e['style'])} · {escape(e['split'])} · page: {escape(page_sel)}</span>"
            f"{history}<p class=question>{escape(e['question'])}</p>"
            f"<div class=label><b>{escape(lab['action'])}</b> · scope {escape(scope)}{query}"
            + (f"<ol>{needs}</ol>" if needs else "") + "</div></section>")
    counts = Counter(e["scenario"] for e in examples)
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    html = f"""<!doctype html><meta charset=utf-8><title>Router examples review</title><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#222;background:#fff}}
section{{border-top:1px solid #ddd;padding:.8rem 0}} .meta{{color:#666;font-size:13px}} .question{{font-size:17px;margin:.35rem 0}}
.hist{{color:#555;font-size:14px}} .label{{background:#eef4ff;padding:.35rem .6rem;border-radius:4px;font-size:14px}}
.label ol{{margin:.2rem 0 0 1.2rem;padding:0}} .q{{color:#333}} #bar{{position:sticky;top:0;background:#f4f4f4;padding:.6rem;border-radius:6px}}
code{{font-size:12px}} input[type=text]{{width:60%}}
@media (prefers-color-scheme: dark){{body{{background:#1b1b1b;color:#ddd}} .label{{background:#23304a}} #bar{{background:#2a2a2a}}
.meta,.hist{{color:#aaa}} section{{border-color:#333}} .q{{color:#ccc}}}}
</style><div id=bar><b>{len(shown)}</b> of {len(examples)} examples ({escape(summary)}).<br>
Tick the examples that are <b>wrong</b>: the label doesn't fit the message, or no user would write it. Note (optional):
<input type=text id=note placeholder="e.g. follow-up names its subject"><br>
<code id=cmd></code> <button onclick="navigator.clipboard.writeText(document.getElementById('cmd').textContent)">copy</button></div>
{''.join(blocks)}
<script>
const folder={json.dumps(str(folder))};
function update(){{const ids=[...document.querySelectorAll('input[type=checkbox]:checked')].map(i=>i.value);
const note=document.getElementById('note').value;
document.getElementById('cmd').textContent='python scripts/router/review_page.py mark "'+folder+'" '+(ids.join(' ')||'(none)')
+(note?' --note "'+note.replace(/"/g,"'")+'"':'')+' --reviewed '+{len(shown)};}}
document.addEventListener('change',update);document.addEventListener('input',update);update();
</script>"""
    out = folder / "review.html"
    out.write_text(html, encoding="utf-8")
    (folder / "review_sample.json").write_text(json.dumps([e["id"] for e in shown]) + "\n", encoding="utf-8")
    return out


def mark(folder: Path, ids: list[str], note: str | None, reviewed: int | None) -> None:
    ids = [i for i in ids if i != "(none)"]
    with (folder / "review.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "bad": ids, "note": note,
                            "reviewed": reviewed}) + "\n")
    print(f"{len(ids)} bad of {reviewed or '?'} reviewed -> {folder / 'review.jsonl'}")
    report(folder)


def report(folder: Path) -> None:
    examples = {e["id"]: e for e in read(folder / "examples.jsonl")}
    sample_path = folder / "review_sample.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8")) if sample_path.is_file() else list(examples)
    bad = {i for r in read(folder / "review.jsonl") for i in r["bad"]}
    reviewed = Counter(examples[i]["scenario"] for i in sample if i in examples)
    wrong = Counter(examples[i]["scenario"] for i in bad if i in examples)
    for scenario, n in sorted(reviewed.items()):
        share = wrong[scenario] / n
        print(f"{scenario:16} {wrong[scenario]:3}/{n:<3} bad  {'FIX TEMPLATE' if share > 0.10 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("page"); p.add_argument("folder", type=Path); p.add_argument("--sample", type=int, default=100)
    m = sub.add_parser("mark"); m.add_argument("folder", type=Path); m.add_argument("ids", nargs="*")
    m.add_argument("--note"); m.add_argument("--reviewed", type=int)
    r = sub.add_parser("report"); r.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command == "page":
        print(page(args.folder, args.sample))
    elif args.command == "mark":
        mark(args.folder, args.ids, args.note, args.reviewed)
    else:
        report(args.folder)


if __name__ == "__main__":
    main()
