"""Invented products with short manual-like sections, for routing data that copies no real manual (experiment R).

    python scripts/router/fictional_products.py --out data/router/fictional_v0_1

Names, categories and versions are fixed here so their variety is controlled: several versions or one, plain
numbers, minors and year-based versions. The local chat model (Qwen3-8B on Docker Model Runner) invents the
features and writes each section; it is told not to describe a real product. Every version of a product gets the
same sections, so comparison questions have counterparts. Writes sections.jsonl (the generate_data.py section
format) and catalog.json ({product: [versions]}).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manuora.evaluation.common import call, settings  # noqa: E402

PRODUCTS = {
    # id: (display name, what it is, versions)
    "vaultwell": ("Vaultwell", "a self-hosted file sync and sharing server with calendars and user management",
                  ["7", "8", "9", "10", "11"]),
    "geoloom": ("Geoloom", "a desktop GIS for maps, vector and raster layers and spatial analysis",
                ["1.8", "2.0", "2.4", "3.1"]),
    "tracewright": ("Tracewright", "an electronics design suite for schematics and printed circuit boards",
                    ["4.0", "5", "6"]),
    "solidmoss": ("Solidmoss", "a parametric 3D CAD modeller with sketches, parts and assemblies", ["0.21"]),
    "quillnest": ("Quillnest", "a note-taking and knowledge-base app with sync and plugins", ["2023.1", "2024.2"]),
    "fleetmint": ("Fleetmint", "a delivery route planning and fleet tracking web service", ["3.0", "4.0", "4.5"]),
    "lumacraft": ("Lumacraft", "a raster image editor with layers, masks and filters", ["12.5", "13.0", "14.1"]),
    "tablery": ("Tablery", "a database administration tool for SQL servers", ["5.2"]),
}
AREAS = ["installation and setup", "everyday tasks", "settings and preferences", "troubleshooting",
         "administration and permissions", "import, export and integration", "advanced features"]

SYSTEM = ("You write documentation for invented software products. The product and its features are fictional: "
          "never describe a real product, never name a real company or product, and do not copy any real manual.")


def prompt(product: str, area: str, n: int) -> str:
    display, what, _ = PRODUCTS[product]
    return (f"{display} is {what}. Invent {n} different features or tasks of {display} in the area of {area}, "
            f"and write the manual section for each: `chapter` (a short chapter title), `section` (the section "
            f"title), and `text` (70 to 110 words in a plain manual style: what it is, the menu or command path, "
            f"two or three concrete steps or options with invented names, and one caveat). Use the product name "
            f"{display} at most once per section.\nJSON: {{\"sections\": [{{\"chapter\": \"...\", \"section\": "
            f"\"...\", \"text\": \"...\"}}, ...]}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-call", type=int, default=6)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()
    from manuora.generation.model_control import is_loaded, load_model, unload_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    generation = settings()["generation"]
    loaded_here = not is_loaded(generation["model"])
    if loaded_here:
        load_model(generation["model"], generation["url"])
    sections, failures, started = [], 0, time.perf_counter()
    try:
        for product, (display, _, versions) in PRODUCTS.items():
            features = []
            for area in AREAS:
                try:
                    reply = call([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": prompt(product, area, args.per_call) + " /no_think"}],
                                 max_tokens=1800, temperature=0.9)
                    text = re.sub(r"<think>.*?</think>", "", reply["text"], flags=re.S)
                    data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                except (ValueError, AttributeError, OSError) as exc:
                    failures += 1
                    print(f"  {product} / {area}: unusable reply ({type(exc).__name__})")
                    continue
                for s in data.get("sections") or []:
                    if all(isinstance(s.get(k), str) and s[k].strip() for k in ("chapter", "section", "text")):
                        features.append({"chapter": s["chapter"].strip(), "section": s["section"].strip(),
                                         "text": s["text"].strip(), "area": area})
            for version in versions:
                doc = f"{product}_manual_{version}"
                for i, f in enumerate(features):
                    chapter_no = AREAS.index(f["area"]) + 1
                    sections.append({
                        "section_id": f"{doc}#{i:04d}", "document_id": doc, "product_id": product,
                        "product_version": version, "chapter": f"{chapter_no} {f['chapter']}",
                        "headings": [f"{chapter_no} {f['chapter']}", f["section"]], "text": f["text"]})
            print(f"{display}: {len(features)} sections x {len(versions)} versions ({time.perf_counter() - started:.0f} s)")
    finally:
        if loaded_here:
            unload_model(generation["model"])
    rng.shuffle(sections)
    (out / "sections.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sections),
                                        encoding="utf-8")
    catalog = {p: v for p, (_, _, v) in PRODUCTS.items()}
    names = {p: d for p, (d, _, _) in PRODUCTS.items()}
    (out / "catalog.json").write_text(json.dumps({"catalog": catalog, "names": names, "failed_calls": failures,
                                                  "sections": len(sections)}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(sections)} sections, {failures} failed calls")


if __name__ == "__main__":
    main()
