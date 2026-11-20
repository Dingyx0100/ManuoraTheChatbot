"""Round-3 conventions applied to an existing synthetic set (experiment R, 2026-11-20).

    python scripts/router/relabel_round3.py --base train_v0_2 --out train_v0_3_base

- two_products (every split): clarify -> direct with both products, no versions (generate_data.two_product_label).
- assistant_feedback (every split): kept only when it plainly addresses the assistant and names no failure
  (generate_data.clearly_about_assistant); "chat is stuck" read like an unnamed product problem in round 2.
Ids, splits and everything else are unchanged. Writes examples.jsonl and relabel.json (counts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_data as g  # noqa: E402
from manuora.evaluation import bench  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = bench.read_file(g.OUT / args.base / "examples.jsonl")
    kept, counts = [], {"relabelled_two_products": 0, "dropped_feedback": 0, "two_products_unparsed": 0}
    dropped = []
    for r in rows:
        if r["scenario"] == "two_products":
            new = g.two_product_label(r["question"])
            if not isinstance(new["product"], list):
                counts["two_products_unparsed"] += 1
                dropped.append({**r, "drop_reason": "two products not both found"})
                continue
            r = {**r, "label": new}
            counts["relabelled_two_products"] += 1
        if r["scenario"] == "assistant_feedback" and not g.clearly_about_assistant(r["question"]):
            counts["dropped_feedback"] += 1
            dropped.append({**r, "drop_reason": "complaint reads like a product problem"})
            continue
        kept.append(r)
    out = g.OUT / args.out
    out.mkdir(parents=True, exist_ok=False)
    bench.write_file(out / "examples.jsonl", kept)
    bench.write_file(out / "dropped.jsonl", dropped)
    counts |= {"base": args.base, "examples": len(kept),
               "by_split": {s: sum(r["split"] == s for r in kept) for s in ("train", "val", "test_unseen", "test_style")}}
    (out / "relabel.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
