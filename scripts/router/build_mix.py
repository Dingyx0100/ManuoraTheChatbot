"""The licence-clean training mix (experiment R, arm M): no QGIS-derived example is trained on.

    python scripts/router/build_mix.py --out mix_v0_1 --synthetic train_v0_3 --fictional fict_v0_1 --external external_v0_1

Training and validation rows:
- synthetic (train_v0_3 train/val): only rows written from Nextcloud manuals (CC BY 3.0), and rows written from
  no manual at all (greetings, complaints, unnamed references); nothing that names QGIS.
- fictional (fict_v0_1 train/val): invented products, no manual text.
- external (external_v0_1 train/val): CANARD, QReCC (Natural Questions part), CLINC150 small talk.
Test rows, never trained on: train_v0_3's QGIS-derived train/val rows become `test_qgis`; its test_unseen
(KiCad, FreeCAD) and test_style (typos) stay; fict_v0_1's typo rows become `test_style_fictional`; the external
test splits are `test_external`. Every row keeps a `source`. A cap per (source, action) on the training rows
keeps the public data under `--external-share` of training.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_data as g  # noqa: E402
from manuora.evaluation import bench  # noqa: E402


def manual_of(seed: str) -> str:
    return seed.split("_", 1)[0]


def names_qgis(row: dict) -> bool:
    text = " ".join([row["question"]] + [h["content"] for h in row.get("history") or []] +
                    [str(row["label"].get("query") or "")]).casefold()
    product = row["label"].get("product")
    products = product if isinstance(product, list) else [product]
    return "qgis" in text or "qgis" in products


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--fictional", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--external-share", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    read = lambda name: bench.read_file(g.OUT / name / "examples.jsonl")  # noqa: E731

    rows = []
    for r in read(args.synthetic):
        manuals = {manual_of(s) for s in r.get("seed") or []}
        if r["split"] in ("train", "val"):
            if "qgis" in manuals or (not manuals and names_qgis(r)):
                rows.append({**r, "split": "test_qgis", "source": "synthetic_qgis"})
            elif manuals <= {"nextcloud"}:
                rows.append({**r, "source": "synthetic_nextcloud" if manuals else "synthetic_no_manual"})
        else:
            rows.append({**r, "source": "synthetic"})
    for r in read(args.fictional):
        split = {"test_style": "test_style_fictional"}.get(r["split"], r["split"])
        rows.append({**r, "split": split, "source": "fictional"})
    external = read(args.external)
    rows += external

    # Keep the public data at most `external_share` of training: trim it evenly across its (source, action) cells.
    train = [r for r in rows if r["split"] == "train"]
    own = sum(not r["source"].startswith(("canard", "qrecc", "clinc")) for r in train)
    ext_train = [r for r in train if r["source"].startswith(("canard", "qrecc", "clinc"))]
    allowed = int(args.external_share / (1 - args.external_share) * own)
    if len(ext_train) > allowed:
        keep_ids = {r["id"] for r in rng.sample(ext_train, allowed)}
        rows = [r for r in rows if not (r in ext_train and r["id"] not in keep_ids)]

    for i, r in enumerate(rows, 1):
        r["origin_id"], r["id"] = r.get("id"), f"{args.out}-{i:05d}"
    out = g.OUT / args.out
    out.mkdir(parents=True, exist_ok=False)
    bench.write_file(out / "examples.jsonl", rows)
    by_split = Counter(r["split"] for r in rows)
    train = [r for r in rows if r["split"] == "train"]
    summary = {"by_split": dict(by_split),
               "train_by_source": dict(Counter(r["source"] for r in train)),
               "train_by_action": dict(Counter(r["label"]["action"] for r in train)),
               "train_by_source_action": {f"{s}/{a}": n for (s, a), n in
                                          sorted(Counter((r["source"], r["label"]["action"]) for r in train).items())},
               "external_share": round(sum(r["source"].startswith(("canard", "qrecc", "clinc")) for r in train)
                                       / max(1, len(train)), 3)}
    (out / "mix.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
