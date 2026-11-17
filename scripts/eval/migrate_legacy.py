"""One-off: fold the legacy question sets in data/evaluation/ into the benchmark (manuora.evaluation.bench).

    python scripts/eval/migrate_legacy.py            # report only
    python scripts/eval/migrate_legacy.py --write    # write eval/bench/cases.jsonl and the private file

Holdout = the confirmation sets manuora_test_v1..v4 (never tuned on). Everything else is dev.
A question that appears in several sets is kept once; holdout copies win, so the holdout never
shares a question with dev. Products outside bench.PUBLIC_PRODUCTS go to the private file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend/src"))
from manuora.evaluation import bench  # noqa: E402

EVAL = ROOT / "data/evaluation"
ADDED = "2026-11-17"
PREFIX = {"nextcloud": "nc", "qgis": "qgis", "kicad": "kc", "freecad": "fc"}
LETTER = {"single": "s", "workflow": "w", "version": "v", "negative": "n", "conversation": "c"}


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def key(text):
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def versions(v):
    v = v if isinstance(v, list) else [v]
    return [str(x) for x in v]


def from_built(folder, split):
    """Sets made by experiments/gold/build_test_set.py (anchors resolved at build time)."""
    out = []
    for q in jsonl(EVAL / folder / "questions.jsonl"):
        negative = q["type"] == "negative"
        expect = "premise" if q.get("negative_kind") == "wrong_version" else "abstain" if negative else "answer"
        out.append(dict(category=q["type"], split=split, status="checked", product_id=q["product_id"],
                        versions=versions(q["versions"]), question=q["question"], expect=expect,
                        gold=None if negative else q["gold"], absence_checks=q.get("absence_checks", []),
                        tags=[q["negative_kind"]] if negative else [], expected_behavior=q.get("expected_behavior"),
                        source=f"{folder}/{q['id']}"))
    return out


def from_gold_v02():
    out = []
    for q in jsonl(EVAL / "manuora_gold_v0_2/queries.verified.resolved.jsonl"):
        if not q.get("relevant_chunk_ids"):
            continue
        out.append(dict(category="single", split="dev", status="checked", product_id=q["product"],
                        versions=versions(q["version"]), question=q["query"], expect="answer",
                        gold={"span_groups": [[{"id": c, "quote": ""} for c in q["relevant_chunk_ids"]]]},
                        tags=[q["intent"], q["version_relation"]], source=f"manuora_gold_v0_2/{q['id']}"))
    return out


def from_version_comparison():
    out = []
    for q in jsonl(EVAL / "version_comparison_v0_1/questions_resolved.jsonl"):
        s = q["scope"]
        out.append(dict(category="version", split="dev", status="checked", product_id=s["product"].casefold(),
                        versions=[str(s["old_version"]), str(s["new_version"])], question=q["question"],
                        expect="premise" if q["category"] == "insufficient_misleading_premise" else "answer",
                        gold={"required_facts": q["required_facts"], "accepted_alternatives": q["accepted_alternatives"],
                              "expected_conclusion": q["expected_conclusion"]},
                        tags=[q["category"]], source=f"version_comparison_v0_1/{q['question_id']}"))
    return out


def from_packing():
    out = []
    for c in json.loads((EVAL / "workflow_packing_v0_1/inputs.json").read_text(encoding="utf-8")):
        first = c["arms"]["four_full"]["evidence"][0]
        out.append(dict(category="workflow", split="dev", status="checked", product_id=first["product_id"],
                        versions=versions(first["product_version"]), question=c["query"], expect="answer",
                        gold=c["gold"], source=f"workflow_packing_v0_1/{c['id']}"))
    return out


def from_pilot():
    out = []
    for q in jsonl(EVAL / "workflow_pilot_v0_1/workflow_gold.jsonl"):
        groups = [[{"id": s["chunk_id"], "quote": ""} for s in g["accepted_sources"]] for g in q["required_groups"]]
        out.append(dict(category="workflow", split="dev", status="checked", product_id=q["product_id"],
                        versions=versions(q["product_version"]), question=q["query"], expect="answer",
                        gold={"span_groups": groups, "step_labels": [g["claim"] for g in q["required_groups"]]},
                        tags=[q.get("question_type") or ""], source=f"workflow_pilot_v0_1/{q['id']}"))
    return out


def from_conversation():
    out = []
    expects = {"retrieve": "answer", "clarify": "clarify", "edit": "edit"}
    data = json.loads((EVAL / "conversation_retrieval_v0_3/cases.json").read_text(encoding="utf-8"))
    for c in data["cases"] if isinstance(data, dict) else data:
        if c.get("set") != "contextual":
            continue
        scope = c.get("expected_scope") or {}
        gold = {"span_groups": [[{"id": i, "quote": ""} for i in c["relevant_chunk_ids"]]]} if c.get("relevant_chunk_ids") else None
        out.append(dict(category="conversation", split="dev", status="checked" if gold or c["expected_action"] != "retrieve" else "draft",
                        product_id=scope.get("product_id") or "", versions=versions(scope.get("product_version")),
                        question=c["question"], history=[{"role": h["role"], "content": h["content"]} for h in c["history"]],
                        expect=expects[c["expected_action"]], gold=gold, tags=[c.get("category") or ""],
                        source=f"conversation_retrieval_v0_3/{c['id']}"))
    return out


def collect():
    holdout = []
    for folder in ("manuora_test_v1", "manuora_test_v2", "manuora_test_v3", "manuora_test_v4"):
        holdout += from_built(folder, "holdout")
    dev = (from_built("workflow_dev_v1", "dev") + from_built("grounding_dev_v1", "dev") + from_built("version_dev_v1", "dev")
           + from_packing() + from_pilot() + from_gold_v02() + from_version_comparison() + from_conversation())
    kept, seen, merged = [], {}, Counter()
    for case in holdout + dev:
        k = (case["product_id"], key(case["question"]))
        if k in seen:
            seen[k].setdefault("also_in", []).append(case["source"])
            merged[f"{case['split']} copy of {seen[k]['split']}"] += 1
            continue
        seen[k] = case
        kept.append(case)
    counters = Counter()
    for case in kept:
        prefix = PREFIX.get(case["product_id"], case["product_id"][:4] or "x")
        counters[(prefix, case["category"])] += 1
        case["id"] = f"{prefix}-{LETTER[case['category']]}{counters[(prefix, case['category'])]:03d}"
        case["added"] = ADDED
        case.setdefault("history", [])
        for k in ("absence_checks", "tags"):
            case.setdefault(k, [])
        case["tags"] = [t for t in case["tags"] if t]
        if case.get("expected_behavior") is None:
            case.pop("expected_behavior", None)
    order = ("id", "category", "split", "status", "product_id", "versions", "question", "history", "expect",
             "gold", "absence_checks", "expected_behavior", "tags", "source", "also_in", "added")
    kept = [{k: c[k] for k in order if k in c} for c in kept]
    return kept, merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cases, merged = collect()
    public = [c for c in cases if c["product_id"] in bench.PUBLIC_PRODUCTS]
    private = [c for c in cases if c["product_id"] not in bench.PUBLIC_PRODUCTS]
    private_names = {c["product_id"].casefold() for c in private if c["product_id"]}
    leaks = [c["id"] for c in public if any(n in json.dumps(c, ensure_ascii=False).casefold() for n in private_names)]
    print(f"{len(cases)} cases: {len(public)} public, {len(private)} private; duplicates merged: {dict(merged)}")
    for name, group in (("public", public), ("private", private)):
        table = Counter((c["split"], c["category"], c["status"]) for c in group)
        print(f"\n{name}:")
        for (split, cat, status), n in sorted(table.items()):
            print(f"  {split:8} {cat:13} {status:8} {n}")
    errors = bench.validate(public, public=True) + bench.validate(private, public=False)
    if leaks:
        errors.append(f"public cases mention a private product: {leaks}")
    if errors:
        print("\nERRORS:\n  " + "\n  ".join(errors[:40]))
        sys.exit(1)
    if args.write:
        bench.write_file(bench.PUBLIC, public)
        bench.write_file(bench.PRIVATE, private)
        print(f"\nwrote {bench.PUBLIC.relative_to(ROOT)} and {bench.PRIVATE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
