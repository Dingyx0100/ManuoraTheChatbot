"""Compare a run with the baseline, and apply the release gate.

    python -m manuora.evaluation.compare data/eval_runs/<run>                 # vs the baseline of its tier and split
    python -m manuora.evaluation.compare data/eval_runs/<run> data/eval_runs/<other>   # two runs (experiment arms)
    python -m manuora.evaluation.compare promote data/eval_runs/<run>         # make a run the new baseline

Baselines live in eval/baselines/<tier>_<split>.json (tracked in git, public cases only, no answer text).
Only cases scored in both runs are compared, so adding or retiring bench cases never looks like a
regression. Local-only cases are compared when both sides are run folders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from manuora.evaluation import bench
from manuora.evaluation.run import aggregate, read_labels

ROOT = bench.ROOT
BASELINES = ROOT / "eval/baselines"
GATES = ROOT / "eval/gates.yaml"


def baseline_for(meta: dict) -> Path:
    return BASELINES / f"{meta['tier']}_{meta['split']}.json"
SHOWN = ("packed_complete", "cited_complete", "answered", "declined", "made_up", "unlabelled", "p50_seconds")


def load_side(path: Path) -> tuple[dict, list[dict]]:
    """(meta, per-case records) from a run folder or a baseline file."""
    if path.is_dir():
        card = json.loads((path / "scorecard.json").read_text(encoding="utf-8"))
        return card["meta"], bench.read_file(path / "records.jsonl")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["meta"], data["records"]


def check(rule: dict, base: dict, cand: dict) -> tuple[bool | None, str]:
    """(passed or None when not applicable, explanation)."""
    b, c = (base.get(rule["category"]) or {}).get(rule["metric"]), (cand.get(rule["category"]) or {}).get(rule["metric"])
    if c is None or (b is None and "max" not in rule):
        return None, "not measured"
    if "max_drop" in rule:
        return c >= b - rule["max_drop"], f"{b} -> {c} (may drop {rule['max_drop']})"
    if "max_rise" in rule:
        return c <= b + rule["max_rise"], f"{b} -> {c} (may rise {rule['max_rise']})"
    if "max_ratio" in rule:
        ok = b in (0, None) or c <= b * rule["max_ratio"]
        return ok, f"{b} -> {c} (at most x{rule['max_ratio']})"
    return c <= rule["max"], f"{c} (at most {rule['max']})"


def compare(base_path: Path, cand_path: Path, gates: Path = GATES) -> tuple[str, bool]:
    base_meta, base_records = load_side(base_path)
    cand_meta, cand_records = load_side(cand_path)
    problems = [f"{k} differs: {base_meta[k]} vs {cand_meta[k]}" for k in ("split", "tier") if base_meta[k] != cand_meta[k]]
    common = {r["id"] for r in base_records} & {r["id"] for r in cand_records}
    labels = read_labels()
    lines = [f"# {cand_meta['name']} vs {base_meta['name']}", "",
             f"{len(common)} cases in both (baseline {len(base_records)}, candidate {len(cand_records)}); "
             f"split `{cand_meta['split']}`, tier `{cand_meta['tier']}`", ""]
    if problems:
        lines += ["**Not comparable:** " + "; ".join(problems), ""]
    verdict = not problems
    for section, private in (("public", False), ("private", True)):
        b = aggregate([r for r in base_records if r["id"] in common and r["private"] == private], labels)
        c = aggregate([r for r in cand_records if r["id"] in common and r["private"] == private], labels)
        if not c:
            continue
        cols = [m for m in SHOWN if any(m in row for row in c.values())]
        lines += [f"## {section}", "", "| category | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for category, row in c.items():
            cells = []
            for m in cols:
                old, new = (b.get(category) or {}).get(m), row.get(m)
                cells.append("" if new is None else f"{old} → **{new}**" if old is not None and old != new else str(new))
            lines.append(f"| {category} | {row['n']} | " + " | ".join(cells) + " |")
        lines.append("")
        if section == "public" and not problems:
            rules = (yaml.safe_load(gates.read_text(encoding="utf-8")) or {}).get("rules", [])
            lines += ["### Gate", ""]
            for rule in rules:
                passed, why = check(rule, b, c)
                if passed is None:
                    continue
                verdict &= passed
                lines.append(f"- {'PASS' if passed else '**FAIL**'} {rule['category']} {rule['metric']}: {why}")
            lines.append("")
    lines.append(f"**Verdict: {'PASS' if verdict else 'FAIL'}**")
    return "\n".join(lines) + "\n", verdict


def promote(run_dir: Path) -> None:
    """Write eval/baseline.json from a run: its meta and its public per-case records (no answer text)."""
    meta, records = load_side(run_dir)
    keep = ("id", "category", "expect", "private", "outcome", "kind", "seconds", "complete", "packed_recall",
            "answered", "cited_complete", "declined", "label_key")
    public = [{k: r[k] for k in keep if k in r} for r in records if not r["private"]]
    target = baseline_for(meta)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"meta": {**meta, "run": run_dir.name}, "records": public}, ensure_ascii=False,
                                 indent=1) + "\n", encoding="utf-8")
    print(f"{target.relative_to(ROOT)} <- {run_dir.name} ({len(public)} public cases)")


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["promote"] and len(argv) == 2:
        promote(Path(argv[1]))
        return
    if len(argv) not in (1, 2):
        raise SystemExit(__doc__)
    if len(argv) == 1:
        cand = Path(argv[0])
        base = baseline_for(load_side(cand)[0])
        if not base.exists():
            raise SystemExit(f"no baseline yet at {base.relative_to(ROOT)}: create one with "
                             f"python -m manuora.evaluation.compare promote <run>")
    else:
        base, cand = Path(argv[0]), Path(argv[1])
    report, ok = compare(base, cand)
    print(report)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
