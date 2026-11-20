"""Random topic cards from the manuals, for writing the hand-written routing slice (experiment R).

    python scripts/router/topic_cards.py --out data/router/handwritten_v0_1/topics.md

Gives the writer real product facts so the wording can be their own. Sections are drawn at random (seeded),
not picked by hand, so the cards are not steered toward any router's known errors.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_data as g  # noqa: E402
from manuora.evaluation.common import settings  # noqa: E402

PRODUCTS = ("nextcloud", "qgis", "kicad", "freecad")
SKIP = re.compile(r"\b(contents|index|license|copyright|glossary|release notes?|changelog|acknowledg\w*|credits)\b", re.I)


def snippet(text: str, n: int = 380) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def card(s: dict, versions: dict) -> str:
    others = [v for v in versions[s["product_id"]] if v != s["product_version"]]
    return (f"**{g.name(s['product_id'])} {s['product_version']}**: {' > '.join(s['headings'])}\n\n"
            f"> {snippet(s['text'])}\n\n"
            f"Other versions in the index: {', '.join(others) if others else 'none'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    sqlite_path = g.ROOT / settings()["sparse_index"]
    docs = [d for d in g.documents(sqlite_path) if d["product_id"] in PRODUCTS]
    pools = {p: [s for s in g.load_sections(sqlite_path, docs, (p,))
                 if len(s["headings"]) >= 2 and len(s["text"]) >= 300 and not SKIP.search(" ".join(s["headings"]))]
             for p in PRODUCTS}
    versions = {p: sorted({s["product_version"] for s in pools[p]}, key=lambda v: [int(x) if x.isdigit() else x
                                                                             for x in re.split(r"[.]", v)])
                for p in PRODUCTS}
    by_key: dict = {}
    for p in PRODUCTS:
        for s in pools[p]:
            by_key.setdefault((p, tuple(h.casefold() for h in s["headings"])), []).append(s)

    lines = ["# Topic cards for the hand-written slice", "",
             "Random sections from the indexed manuals (seed %d). Use them for facts only: write each message in "
             "your own words, as you would really type it. You don't need to use every card, and you can ask "
             "about anything the snippet suggests, not only its heading." % args.seed, "",
             "Messages that need no product knowledge (reformat requests, \"that didn't work\", greetings, "
             "complaints, two products in one question) don't need a card.", ""]

    lines += ["## A. Single topics (16)", "",
              "Use for: new questions (direct); a first question plus a follow-up (rewrite); a first question plus a "
              "version correction (rewrite, pick a version from \"Other versions\"); a first question plus a "
              "reformat request (edit); two cards from different products for a topic switch.", ""]
    n = 0
    for p in PRODUCTS:
        for s in rng.sample(pools[p], 4):
            n += 1
            lines += [f"### A{n}", card(s, versions)]

    lines += ["## B. Pairs for a multi-step goal (5)", "",
              "Two sections from the same manual that one task could need together. Write one message asking for "
              "the whole task (several_needs: true).", ""]
    for i in range(5):
        p = PRODUCTS[i % len(PRODUCTS)]
        s = rng.choice(pools[p])
        near = [t for t in g.related(s, [t for t in rng.sample(pools[p], min(400, len(pools[p])))
                                          if t["document_id"] == s["document_id"] and t["chapter"] != s["chapter"]])]
        if not near:
            continue
        lines += [f"### B{i + 1}", card(s, versions), "and", "", card(near[0], versions)]

    lines += ["## C. Same section in two versions, for a comparison (4)", "",
              "Write one message asking how this differs between the two versions (several_needs: true).", ""]
    found = 0
    for s in rng.sample([s for p in PRODUCTS for s in pools[p]], 4000):
        others = g._counterparts(s, by_key)
        if not others or found == 4:
            continue
        t = rng.choice(others)
        found += 1
        lines += [f"### C{found}", f"**{g.name(s['product_id'])} {s['product_version']} vs {t['product_version']}**: "
                  f"{' > '.join(s['headings'])}", "", f"> {s['product_version']}: {snippet(s['text'], 250)}", "",
                  f"> {t['product_version']}: {snippet(t['text'], 250)}", ""]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: A {n}, C {found}")


if __name__ == "__main__":
    main()
