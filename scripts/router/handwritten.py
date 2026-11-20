"""Hand-written routing slice (experiment R): YAML -> routing examples, as data/router/<name>/examples.jsonl.

    python scripts/router/handwritten.py --name handwritten_v0_1

Reads every *.yaml in data/router/<name>/ (messages.yaml, drafts_claude.yaml). Each entry becomes one example with
split "test_handwritten", scored like the synthetic ones. Scenario is the expected action; `author` and
`products` are kept so results can be split by who wrote the message and whether it names several products.
A list of products (round 3 convention) is scored as "every one of them, searched separately".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = {"direct", "rewrite", "clarify", "edit", "chat"}


def example(entry: dict, source: str) -> dict:
    expect = entry["expect"]
    assert expect in ACTIONS, (entry["id"], expect)
    products = entry.get("product")
    products = products if isinstance(products, list) else ([products] if products else [])
    page = (entry.get("page") or "").split()
    several = bool(entry.get("several_needs"))
    answers = expect in ("direct", "rewrite")
    return {"id": f"hw-{entry['id']}", "question": entry["message"],
            "history": [{"role": "user", "content": m} for m in entry.get("earlier") or []],
            "seed": [], "ui": {"product_id": page[0] if page else None,
                               "product_version": page[1] if len(page) > 1 else None},
            "label": {"action": expect, "product": (products[0] if len(products) == 1 else products or None) if answers else None,
                      "versions": [str(v) for v in entry.get("versions") or []] if answers else [],
                      "needs": ["need 1", "need 2"] if several else ["need"], "query": None},
            "scenario": expect, "style": "handwritten", "split": "test_handwritten",
            "author": entry.get("author") or "user", "products": products, "source": source,
            "note": entry.get("note") or ""}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="handwritten_v0_1")
    args = parser.parse_args()
    folder = ROOT / "data/router" / args.name
    rows, seen = [], set()
    for path in sorted(folder.glob("*.yaml")):
        for entry in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            if str(entry["id"]).startswith("example-"):
                continue
            assert entry["id"] not in seen, f"duplicate id {entry['id']}"
            seen.add(entry["id"])
            rows.append(example(entry, path.name))
    (folder / "examples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                           encoding="utf-8")
    print(f"wrote {len(rows)} examples to {(folder / 'examples.jsonl').relative_to(ROOT).as_posix()}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
