"""Combine generated sets into one training set (experiment R).

    python scripts/router/combine_sets.py --out train_v0_2 --base train_v0_1 --add aug_v0_1

Keeps every split of --base unchanged and only the train and val rows of each --add set, so the base's test
splits stay exactly as scored before. Writes data/router/<out>/examples.jsonl and run.json.
"""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/router"


def read(name: str) -> list[dict]:
    with open(OUT / name / "examples.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--add", nargs="+", required=True)
    args = parser.parse_args()
    rows = read(args.base)
    for name in args.add:
        rows += [e for e in read(name) if e["split"] in ("train", "val")]
    assert len({e["id"] for e in rows}) == len(rows), "duplicate ids"
    out = OUT / args.out
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "examples.jsonl", "w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {"name": args.out, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "base": args.base, "added": args.add,
               "examples": len(rows), "by_split": dict(collections.Counter(e["split"] for e in rows)),
               "train_by_scenario": dict(collections.Counter(e["scenario"] for e in rows if e["split"] == "train"))}
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
