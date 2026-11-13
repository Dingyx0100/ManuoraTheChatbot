"""Sample visual-region decisions and render their source pages for review."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from hashlib import sha256
from html import escape
from pathlib import Path
from urllib.parse import quote

import pypdfium2 as pdfium


def load_regions(sources: Path) -> list[tuple[Path, dict]]:
    records = []
    for path in sorted(sources.glob("**/visual_regions.jsonl")):
        with path.open(encoding="utf-8") as source:
            records.extend((path.parent, json.loads(line)) for line in source)
    return records


# Regions every review must show, in this order. Optional keys narrow a pin: label (substring of
# context_text) and route. Pins for local-only manuals go in data/review_pins.json (same format, appended).
PINS = [
    {"product_id": "qgis", "page": 39, "kind": "icon_table_row", "label": "Delete Selected"},
    {"product_id": "nextcloud", "page": 281, "kind": "icon_table_row", "label": "Paused"},
    {"product_id": "freecad", "page": 105, "kind": "picture"},
    {"product_id": "qgis", "page": 272, "kind": "picture"},
    {"product_id": "nextcloud", "page": 191, "kind": "picture"},
    {"product_id": "freecad", "page": 106, "kind": "picture"},
]
LOCAL_PINS = Path(__file__).resolve().parents[4] / "data" / "review_pins.json"


def pins() -> list[dict]:
    extra = json.loads(LOCAL_PINS.read_text(encoding="utf-8")) if LOCAL_PINS.is_file() else []
    return PINS + extra


def _pin_index(region: dict, pin_list: list[dict]) -> int | None:
    for i, pin in enumerate(pin_list):
        if (region["product_id"], region["page"], region["kind"]) != (pin["product_id"], pin["page"], pin["kind"]):
            continue
        if "label" in pin and pin["label"] not in region["context_text"]:
            continue
        if "route" in pin and region.get("route") != pin["route"]:
            continue
        return i
    return None


def is_pinned(region: dict, pin_list: list[dict] | None = None) -> bool:
    return _pin_index(region, pins() if pin_list is None else pin_list) is not None


def sample_regions(records: list[tuple[Path, dict]], limit: int = 96,
                   pin_list: list[dict] | None = None) -> list[tuple[Path, dict]]:
    """Pinned checks plus deterministic round-robin sampling of every stratum."""
    pin_list = pins() if pin_list is None else pin_list
    if limit < len(pin_list):
        raise ValueError(f"region review sample size must be at least {len(pin_list)} (the pinned regions)")
    pinned = sorted((item for item in records if is_pinned(item[1], pin_list)),
                    key=lambda item: _pin_index(item[1], pin_list))
    chosen = list(pinned[:limit])
    seen = {item[1]["region_id"] for item in chosen}
    strata: dict[tuple, list[tuple[Path, dict]]] = defaultdict(list)
    for item in records:
        region = item[1]
        if region["region_id"] in seen:
            continue
        key = (region["document_id"], region["kind"], region["visual_priority"])
        strata[key].append(item)
    rng = random.Random(20261017)
    for items in strata.values():
        rng.shuffle(items)
    keys = sorted(strata)
    while len(chosen) < limit and any(strata.values()):
        for key in keys:
            if strata[key] and len(chosen) < limit:
                chosen.append(strata[key].pop())
    return chosen


def source_pdf(output_dir: Path, name: str) -> Path:
    folder = output_dir.parent.parent if output_dir.parent.name == "processed" else output_dir
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_image(image, path: Path, quality: int) -> str:
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)
    return quote(path.relative_to(path.parents[2]).as_posix(), safe="/")


def crop_box(region: dict, image) -> tuple[int, int, int, int]:
    left, bottom, right, top = region["bbox_pdf"]
    page_width, page_height = region["page_size_pdf"]
    xscale, yscale = image.width / page_width, image.height / page_height
    coords = (max(0, round(left*xscale)), max(0, round((page_height-top)*yscale)),
              min(image.width, round(right*xscale)),
              min(image.height, round((page_height-bottom)*yscale)))
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError(f"empty rendered crop: {region['region_id']}")
    return coords


def review_card(region: dict, page_data: str, crop_data: str,
                pdf_href: str, pinned: bool) -> str:
    width, height = region["page_size_pdf"]
    left, bottom, right, top = region["bbox_pdf"]
    box_style = (f"left:{100*left/width:.3f}%;top:{100*(height-top)/height:.3f}%;"
                 f"width:{100*(right-left)/width:.3f}%;height:{100*(top-bottom)/height:.3f}%")
    section = region.get("section") or {}
    section_path = " › ".join(section.get("heading_path") or []) or "No section match"
    route = region["route"]
    priority = region["visual_priority"]
    region_id = region["region_id"]
    return f"""
<article class="region-card" id="region-{escape(region_id, quote=True)}"
 data-id="{escape(region_id, quote=True)}"
 data-doc="{escape(region['document_id'], quote=True)}"
 data-kind="{escape(region['kind'], quote=True)}"
 data-priority="{escape(priority, quote=True)}" data-pinned="{int(pinned)}">
 <div class="region-head"><h3>{escape(region['product_id'] or region['document_id'])} · PDF page {region['page']}</h3>
 <span>{escape(region['kind'])} · {escape(priority)} priority · {escape(route)} route
 {"· known example" if pinned else ""}</span></div>
 <p class="region-section">{escape(section_path)}</p>
 <div class="region-images"><div><h4>Source page · proposed box in red</h4>
  <div class="region-page"><img loading="lazy" src="{page_data}" alt="PDF page {region['page']}">
  <span class="region-outline" style="{box_style}"></span></div></div>
  <div><h4>Proposed crop</h4><img class="region-crop" loading="lazy"
    src="{crop_data}" alt="Region crop on page {region['page']}"></div></div>
 <p><strong>Nearby text:</strong> {escape(region['context_text'] or 'None found')}</p>
 <p><strong>Rule:</strong> {escape(region['reason'])} ·
 <a href="{escape(pdf_href, quote=True)}#page={region['page']}">Open source PDF</a></p>
 <div class="region-answers">
  <label>Box contains the intended content<select data-field="box"><option value="">Choose…</option>
   <option value="yes">Yes</option><option value="no">No</option><option value="unsure">Unsure</option></select></label>
  <label>Crop has enough context<select data-field="context"><option value="">Choose…</option>
   <option value="yes">Yes</option><option value="no">No</option><option value="unsure">Unsure</option></select></label>
  <label>Best route<select data-field="route"><option value="">Choose…</option>
   <option value="text">Text</option><option value="visual">Visual</option>
   <option value="both">Both</option><option value="skip">Skip</option>
   <option value="unsure">Unsure</option></select></label>
  <label class="region-note">Optional note<input data-field="note" type="text"
   placeholder="What should change?"></label></div>
 <details><summary>Region metadata</summary><code>{escape(region_id)}</code> ·
  source refs {escape(', '.join(region['source_refs']))} ·
  crop box {escape(str(region['bbox_pdf']))} PDF points</details>
</article>"""


def render_region_review(sources: Path, html_path: Path,
                         sample_size: int = 96) -> dict:
    records = load_regions(sources)
    selected = sample_regions(records, sample_size)
    if not selected:
        return {"cards": '<p>No visual-region inventories found yet.</p>',
                "sample_id": "none", "count": 0,
                "documents": [], "kinds": [], "priorities": []}
    signature = "\n".join(item[1]["region_id"] + ":" + item[1]["source_pdf_sha256"]
                          for item in selected)
    sample_id = sha256(signature.encode("utf-8")).hexdigest()[:16]
    sample_manifest = {
        "schema_version": 1, "sample_id": sample_id, "sample_size": len(selected),
        "method": "pinned examples plus seeded round-robin by document/kind/priority",
        "region_ids": [region["region_id"] for _, region in selected],
    }
    sample_path = html_path.parent / "region_review_sample.json"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(sample_manifest, indent=2) + "\n", encoding="utf-8")
    assets = html_path.parent / "region_review_assets" / sample_id
    assets.mkdir(parents=True, exist_ok=True)
    pdfs = {}
    page_images = {}
    cards = []
    try:
        for folder, region in selected:
            source = source_pdf(folder, region["source_pdf"])
            if source not in pdfs:
                pdfs[source] = pdfium.PdfDocument(str(source))
            key = (source, region["page"])
            if key not in page_images:
                image = pdfs[source][region["page"]-1].render(scale=0.85).to_pil()
                page_name = f"page-{region['document_id']}-{region['page']:04d}.jpg"
                page_images[key] = save_image(image, assets / page_name, quality=62)
            image = pdfs[source][region["page"]-1].render(scale=1.6).to_pil()
            crop = image.crop(crop_box(region, image))
            crop_name = "crop-" + sha256(region["region_id"].encode("utf-8")).hexdigest()[:16] + ".jpg"
            crop_href = save_image(crop, assets / crop_name, quality=78)
            pdf_href = quote(Path("..", *source.relative_to(sources.parent).parts).as_posix(), safe="/")
            cards.append(review_card(region, page_images[key], crop_href,
                                     pdf_href, is_pinned(region)))
    finally:
        for pdf in pdfs.values():
            pdf.close()
    return {"cards": "\n".join(cards), "sample_id": sample_id,
            "count": len(selected),
            "documents": sorted({r["document_id"] for _, r in selected}),
            "kinds": sorted({r["kind"] for _, r in selected}),
            "priorities": sorted({r["visual_priority"] for _, r in selected})}
