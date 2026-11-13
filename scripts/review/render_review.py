"""Build one offline page for retrieval chunks and corrected table review."""

import argparse
import base64
import json
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import pypdfium2
from manuora.ingestion.region_review import render_region_review


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "debug" / "review.html"


def source_pdf_for_output(output_dir: Path, name: str) -> Path:
    """Locate the PDF for either isolated or legacy beside-PDF artifacts."""
    source_dir = output_dir.parent.parent if output_dir.parent.name == "processed" else output_dir
    pdf_path = source_dir / name
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    return pdf_path

STYLE = """
:root { color-scheme: light; font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  background: #f5f7f8; color: #17242e; }
* { box-sizing: border-box; }
body { margin: 0; }
header { background: #183c4a; color: white; padding: 2rem max(1.5rem, calc((100vw - 1100px)/2)); }
header h1 { margin: 0 0 .35rem; font-size: 1.8rem; }
header p { margin: 0; max-width: 55rem; color: #d8e8ec; line-height: 1.5; }
main { max-width: 1600px; margin: auto; padding: 1.25rem; }
.tabs { display: flex; gap: .5rem; max-width: 1600px; margin: 1rem auto 0; padding: 0 1.25rem; }
.tabs button { border: 1px solid #adc3cb; border-radius: 8px; background: white;
  color: #173c4b; padding: .65rem 1rem; font: inherit; font-weight: 700; cursor: pointer; }
.tabs button[aria-selected="true"] { background: #176684; color: white; }
.review-view[hidden] { display: none; }
#chunks-view { max-width: 1100px; margin: auto; }
.controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .8rem;
  background: white; border: 1px solid #d9e2e6; border-radius: 12px; padding: 1rem;
  position: sticky; top: 0; z-index: 2; box-shadow: 0 4px 18px #173a4b12; }
label { display: flex; flex-direction: column; gap: .3rem; font-size: .84rem; font-weight: 650; }
input, select { min-width: 0; width: 100%; border: 1px solid #afc1c9; border-radius: 7px;
  background: white; color: #17242e; padding: .55rem .65rem; font: inherit; }
.wide { grid-column: span 2; }
.checkbox { flex-direction: row; align-items: center; gap: .55rem; }
.checkbox input { width: auto; }
.toolbar { display: flex; justify-content: space-between; gap: 1rem; align-items: center;
  flex-wrap: wrap; margin: 1.2rem 0 .8rem; }
.toolbar p { margin: 0; color: #425763; }
.toolbar a { color: #146383; }
.chunk { background: white; border: 1px solid #d9e2e6; border-radius: 11px;
  margin: 0 0 .8rem; padding: 1rem 1.1rem; scroll-margin-top: 230px; }
.chunk[hidden] { display: none; }
.chunk-top { display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem; }
.chunk-number { font-weight: 750; color: #12607a; text-decoration: none; }
.meta { color: #516571; font-size: .84rem; }
.heading { font-weight: 680; line-height: 1.45; margin: .55rem 0; }
.heading span { color: #7b8990; padding: 0 .3rem; }
.none { font-weight: 450; color: #72818a; font-style: italic; }
.badges { display: flex; flex-wrap: wrap; gap: .35rem; margin: .45rem 0; }
.badge { border-radius: 999px; background: #eaf2f5; color: #335869;
  padding: .16rem .5rem; font-size: .76rem; }
.badge.toc { background: #fff0d4; color: #725516; }
.badge.callout { background: #ffe3db; color: #8a3b2b; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; line-height: 1.55;
  margin: .85rem 0 .5rem; }
.context-text { display: none; }
body.show-context .body-text { display: none; }
body.show-context .context-text { display: block; }
details { margin-top: .75rem; color: #536976; font-size: .8rem; }
summary { cursor: pointer; }
.refs { overflow-wrap: anywhere; padding: .4rem 0; }
#empty { display: none; text-align: center; padding: 3rem; color: #516571; }
.table-card { background: white; border: 1px solid #d9e2e6; border-radius: 11px;
  margin: 0 0 1rem; padding: 1.2rem; }
.table-card h2 { margin: 0 0 .45rem; }
.table-card p { color: #425763; line-height: 1.45; }
.table-comparison { display: grid; grid-template-columns: minmax(260px, 40%) minmax(400px, 60%); gap: 1rem; }
.table-comparison img { width: 100%; border: 1px solid #ccd4dd; }
.table-card table { border-collapse: collapse; width: 100%; font-size: .84rem; }
.table-card th, .table-card td { border: 1px solid #ccd4dd; padding: .4rem; vertical-align: top;
  text-align: left; overflow-wrap: anywhere; }
.table-card th { background: #e7eef5; }
.table-card tr:nth-child(even) { background: #f5f7fa; }
.table-card details { overflow-x: auto; }
.region-controls { display: flex; flex-wrap: wrap; gap: .7rem; align-items: end;
  background: white; border: 1px solid #d9e2e6; border-radius: 10px;
  padding: 1rem; position: sticky; top: 0; z-index: 3; }
.region-controls label { min-width: 140px; flex: 1; }
.region-controls button { border: 1px solid #8cabb9; border-radius: 7px;
  padding: .55rem .8rem; background: #176684; color: white; cursor: pointer; }
.region-card { background: white; border: 1px solid #d9e2e6; border-radius: 11px;
  padding: 1.2rem; margin: 1rem 0; scroll-margin-top: 125px; }
.region-card[hidden] { display: none; }
.region-head { display: flex; justify-content: space-between; align-items: baseline;
  flex-wrap: wrap; gap: .5rem; }
.region-head h3 { margin: 0; }
.region-head span, .region-section { color: #516571; }
.region-images { display: grid; grid-template-columns: minmax(260px, 44%) minmax(260px, 56%); gap: 1rem; }
.region-images h4 { margin: .4rem 0; }
.region-page { position: relative; width: 100%; }
.region-page img { display: block; width: 100%; border: 1px solid #ccd4dd; }
.region-outline { position: absolute; border: 3px solid #d32f2f;
  box-shadow: 0 0 0 1px white inset; pointer-events: none; }
.region-crop { max-width: 100%; max-height: 650px; border: 1px solid #ccd4dd; }
.region-answers { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .7rem; margin-top: 1rem; }
.region-note { grid-column: 1 / -1; }
.region-card.reviewed { border-left: 5px solid #2b8b65; }
#region-progress { margin: .8rem 0; font-weight: 700; }
@media (max-width: 680px) {
  header { padding: 1.5rem; } .controls { grid-template-columns: 1fr; position: static; }
  .wide { grid-column: span 1; } main { padding: .8rem; }
  .table-comparison { grid-template-columns: 1fr; }
  .region-images, .region-answers { grid-template-columns: 1fr; }
}
"""

SCRIPT = """
const cards = [...document.querySelectorAll('.chunk')];
const documentSelect = document.getElementById('document');
const methodSelect = document.getElementById('method');
const pageInput = document.getElementById('page');
const labelSelect = document.getElementById('label');
const searchInput = document.getElementById('search');
const count = document.getElementById('count');
const empty = document.getElementById('empty');
const pdfLink = document.getElementById('pdf-link');
function filterCards() {
  const doc = documentSelect.value;
  const method = methodSelect.value;
  const page = pageInput.value.trim();
  const label = labelSelect.value;
  const query = searchInput.value.trim().toLocaleLowerCase();
  let shown = 0;
  for (const card of cards) {
    const visible = card.dataset.document === doc && card.dataset.method === method
      && (!page || card.dataset.pages.split(',').includes(page))
      && (!label || card.dataset.labels.split(',').includes(label))
      && (!query || (card.querySelector('.heading').textContent + ' '
        + card.querySelector('.body-text').textContent + ' '
        + card.querySelector('.refs').textContent).toLocaleLowerCase().includes(query));
    card.hidden = !visible;
    if (visible) shown++;
  }
  count.textContent = `${shown} of ${cards.filter(c => c.dataset.document === doc && c.dataset.method === method).length} chunks shown`;
  empty.style.display = shown ? 'none' : 'block';
  pdfLink.href = documentSelect.selectedOptions[0].dataset.pdf;
}
for (const control of [documentSelect, methodSelect, pageInput, labelSelect, searchInput]) {
  control.addEventListener('input', filterCards);
}
document.getElementById('context').addEventListener('change', event => {
  document.body.classList.toggle('show-context', event.target.checked);
});
for (const button of document.querySelectorAll('.tabs button')) {
  button.addEventListener('click', () => {
    for (const tab of document.querySelectorAll('.tabs button')) {
      tab.setAttribute('aria-selected', String(tab === button));
    }
    document.getElementById('chunks-view').hidden = button.dataset.view !== 'chunks';
    document.getElementById('tables-view').hidden = button.dataset.view !== 'tables';
  });
}
filterCards();
"""

REGION_SCRIPT = """
const regionView = document.getElementById('regions-view');
const regionCards = [...regionView.querySelectorAll('.region-card')];
const sampleId = regionView.dataset.sampleId;
const storageKey = `manuora-region-review:${sampleId}`;
let regionAnswers = {};
try { regionAnswers = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (_) {}
function reviewed(card) {
  const answer = regionAnswers[card.dataset.id] || {};
  return Boolean(answer.box && answer.context && answer.route);
}
function applyAnswers() {
  for (const card of regionCards) {
    const answer = regionAnswers[card.dataset.id] || {};
    for (const control of card.querySelectorAll('[data-field]')) {
      control.value = answer[control.dataset.field] || '';
    }
    card.classList.toggle('reviewed', reviewed(card));
  }
  filterRegions();
}
function filterRegions() {
  const doc = document.getElementById('region-document').value;
  const kind = document.getElementById('region-kind').value;
  const priority = document.getElementById('region-priority').value;
  const status = document.getElementById('region-status').value;
  const pinned = document.getElementById('region-pinned').checked;
  let visible = 0;
  for (const card of regionCards) {
    const done = reviewed(card);
    card.hidden = Boolean((doc && card.dataset.doc !== doc)
      || (kind && card.dataset.kind !== kind)
      || (priority && card.dataset.priority !== priority)
      || (status === 'reviewed' && !done)
      || (status === 'unreviewed' && done)
      || (pinned && card.dataset.pinned !== '1'));
    if (!card.hidden) visible++;
  }
  const doneCount = regionCards.filter(reviewed).length;
  document.getElementById('region-progress').textContent =
    `${doneCount} of ${regionCards.length} reviewed · ${visible} shown`;
}
for (const control of regionView.querySelectorAll('.region-controls select, #region-pinned')) {
  control.addEventListener('change', filterRegions);
}
for (const card of regionCards) {
  for (const control of card.querySelectorAll('[data-field]')) {
    control.addEventListener('change', () => {
      const answer = regionAnswers[card.dataset.id] || {};
      answer[control.dataset.field] = control.value;
      regionAnswers[card.dataset.id] = answer;
      try { localStorage.setItem(storageKey, JSON.stringify(regionAnswers)); } catch (_) {}
      card.classList.toggle('reviewed', reviewed(card));
      filterRegions();
    });
  }
}
document.getElementById('region-export').addEventListener('click', () => {
  const payload = {schema_version: 1, sample_id: sampleId,
    exported_at: new Date().toISOString(), answers: regionAnswers};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = `manuora_region_review_${sampleId}.json`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.getElementById('region-import').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    if (payload.sample_id !== sampleId || typeof payload.answers !== 'object') {
      alert('This review file belongs to a different sample.'); return;
    }
    regionAnswers = payload.answers;
    try { localStorage.setItem(storageKey, JSON.stringify(regionAnswers)); } catch (_) {}
    applyAnswers();
  } catch (error) { alert(`Could not import review: ${error}`); }
  event.target.value = '';
});
applyAnswers();
"""


def render_chunk(chunk: dict, document_number: int, method: str, pdf_href: str) -> str:
    """Render one chunk with escaped text and searchable source metadata."""
    index = chunk["index"]
    pages = ",".join(map(str, chunk["pages"]))
    labels = ",".join(chunk["labels"])
    heading = (
        ' <span>›</span> '.join(escape(part) for part in chunk["headings"])
        if chunk["headings"]
        else '<span class="none">No heading</span>'
    )
    badges = "".join(
        f'<span class="badge{" toc" if label == "document_index" else ""}">{escape(label)}</span>'
        for label in chunk["labels"]
    )
    badges += "".join(
        f'<span class="badge callout">{escape(marker["kind"])} starts here</span>'
        for marker in chunk.get("callout_starts", [])
    )
    refs = ", ".join(escape(ref) for ref in chunk["refs"]) or "No source refs"
    identifier = f"d{document_number}-{method}-{index}"
    page_link = (
        f'<a href="{escape(pdf_href, quote=True)}#page={chunk["pages"][0]}">Open PDF page</a>'
        if chunk["pages"] else ""
    )
    return f"""
<article class="chunk" id="{identifier}" data-document="{document_number}"
  data-method="{method}" data-pages="{pages}" data-labels="{escape(labels, quote=True)}">
  <div class="chunk-top"><a class="chunk-number" href="#{identifier}">Chunk {index + 1}</a>
    <span class="meta">Page {escape(', '.join(map(str, chunk['pages'])) or 'unknown')} · {chunk['context_tokens']} tokens · {page_link}</span></div>
  <div class="heading">{heading}</div>
  <div class="badges">{badges}</div>
  <pre class="body-text">{escape(chunk['text'])}</pre>
  <pre class="context-text">{escape(chunk['contextual_text'])}</pre>
  <details><summary>Source references</summary><div class="refs">{refs}</div></details>
</article>"""


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_table_cards() -> list[str]:
    """Embed source-page images so the table comparison needs no asset folder."""
    cards = []
    for normalized_path in sorted((ROOT / "data" / "sources").glob("**/normalized_tables.jsonl")):
        folder = normalized_path.parent
        parsed = json.loads((folder / "parsed.json").read_text(encoding="utf-8"))
        raw_by_ref = {item["self_ref"]: item for item in parsed["tables"]}
        records = [json.loads(line) for line in normalized_path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            pdf_path = source_pdf_for_output(folder, record["source_pdf"])
            page = record["pdf_page"]
            label = folder.relative_to(ROOT / "data" / "sources").as_posix()
            pdf = pypdfium2.PdfDocument(str(pdf_path))
            try:
                image_bytes = BytesIO()
                pdf[page - 1].render(scale=1.7).to_pil().save(image_bytes, format="PNG")
            finally:
                pdf.close()
            image_data = base64.b64encode(image_bytes.getvalue()).decode("ascii")
            pdf_href = quote(Path("..", *pdf_path.relative_to(ROOT / "data").parts).as_posix(), safe="/")
            json_href = quote(Path("..", *normalized_path.relative_to(ROOT / "data").parts).as_posix(), safe="/")
            raw = raw_by_ref[record["table_ref"]]["data"]
            raw_rows = [[(cell or {}).get("text", "") for cell in row] for row in raw["grid"]]
            keys = (["variable", "default", "meaning"] if record["kind"] == "configuration"
                    else ["state", "status", "mono_tray", "color_tray"])
            corrected = [
                [", ".join(map(str, row["raw_row_indices"])), *(row[key] for key in keys)]
                for row in record["rows"]
            ]
            cards.append(f"""
<article class="table-card">
  <h2>{escape(label)} — PDF page {page}: {escape(record['title'])}</h2>
  <p>{escape(' › '.join(record['section_path']))} · {len(corrected)} corrected rows ·
    raw {raw['num_rows']}×{raw['num_cols']} · {escape(record['review_status'])}</p>
  <p><a href="{escape(pdf_href, quote=True)}#page={page}">Open source PDF at this page</a>
    · <a href="{escape(json_href, quote=True)}">Open normalized JSONL</a>
    · table ref <code>{escape(record['table_ref'])}</code></p>
  <div class="table-comparison"><section><h3>Source page</h3>
    <a href="{escape(pdf_href, quote=True)}#page={page}"><img
      src="data:image/png;base64,{image_data}" alt="PDF page {page}"></a></section>
    <section><h3>Corrected table</h3>
      {html_table(['Raw rows', *record['headers']], corrected)}</section></div>
  <details><summary>Raw Docling grid ({raw['num_rows']}×{raw['num_cols']})</summary>
    {html_table([f'Raw column {i + 1}' for i in range(raw['num_cols'])], raw_rows)}</details>
</article>""")
    return cards


def build_page(include_experimental: bool = False) -> Path:
    """Combine available chunk views and corrected-table comparisons."""
    sources = ROOT / "data" / "sources"
    entries_by_dir = (
        {path.parents[2]: json.loads(path.read_text(encoding="utf-8"))
         for path in sources.glob("**/debug/docling_chunk_review/summary.json")}
        if include_experimental else {}
    )
    for path in sources.glob("**/retrieval_chunks.jsonl"):
        if path.parent not in entries_by_dir:
            with path.open(encoding="utf-8") as source:
                entries_by_dir[path.parent] = json.loads(next(source))
    entries = sorted(entries_by_dir.items(), key=lambda pair: pair[0].as_posix())
    if not entries:
        raise FileNotFoundError("No saved chunks found under data/sources")

    options = []
    cards = []
    labels = set()
    first_assembled_number = None
    for number, (source_dir, summary) in enumerate(entries):
        source_pdf = source_pdf_for_output(source_dir, summary["source_pdf"])
        relative_pdf = source_pdf.relative_to(ROOT)
        pdf_href = quote(Path("..", *relative_pdf.parts[1:]).as_posix(), safe="/")
        source_folder = relative_pdf.parent.relative_to(Path("data/sources")).as_posix()
        title = f"{source_folder} / {summary['document_id']}"
        assembled_path = source_dir / "retrieval_chunks.jsonl"
        if assembled_path.is_file() and first_assembled_number is None:
            first_assembled_number = number
        selected = " selected" if first_assembled_number == number and assembled_path.is_file() else ""
        options.append(
            f'<option value="{number}" data-pdf="{escape(pdf_href, quote=True)}"{selected}>{escape(title)}</option>'
        )
        methods = (
            ("assembled", "adapted_hybrid", "corrected_tables", "hybrid", "hierarchical")
            if include_experimental else ("assembled",)
        )
        for method in methods:
            chunk_path = (assembled_path if method == "assembled" else
                          source_dir / "debug" / "docling_chunk_review" / f"{method}.jsonl")
            if not chunk_path.is_file():
                continue
            with chunk_path.open(encoding="utf-8") as source:
                for index, line in enumerate(source):
                    chunk = json.loads(line)
                    if method == "assembled":
                        chunk["index"] = index
                    labels.update(chunk["labels"])
                    cards.append(render_chunk(chunk, number, method, pdf_href))

    label_options = '<option value="">All labels</option>' + "".join(
        f'<option value="{escape(label, quote=True)}">{escape(label)}</option>'
        for label in sorted(labels)
    )
    method_options = '<option value="assembled">Assembled retrieval chunks</option>'
    if include_experimental:
        method_options += (
            '<option value="adapted_hybrid">Hybrid + normalized headings</option>'
            '<option value="corrected_tables">Corrected table rows</option>'
            '<option value="hybrid">Raw Hybrid</option>'
            '<option value="hierarchical">Raw Hierarchical</option>'
        )
    table_cards = render_table_cards()
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Manuora ingestion review</title><style>{STYLE}</style></head><body>
<header><h1>Manuora ingestion review</h1><p>Inspect retrieval chunks, experimental Docling chunks, and corrected tables against their source PDF pages.</p></header>
<nav class="tabs" aria-label="Review views"><button type="button" data-view="chunks" aria-selected="true">Chunks</button>
  <button type="button" data-view="tables" aria-selected="false">Corrected tables</button></nav>
<main><section id="chunks-view" class="review-view"><section class="controls" aria-label="Chunk filters">
  <label>Document<select id="document">{''.join(options)}</select></label>
  <label>Chunk view<select id="method">{method_options}</select></label>
  <label>Page number<input id="page" type="number" min="1" placeholder="Any page"></label>
  <label>Content label<select id="label">{label_options}</select></label>
  <label class="wide">Search headings, text, or source refs<input id="search" type="search" placeholder="e.g. Installing on Windows"></label>
  <label class="checkbox wide"><input id="context" type="checkbox">Show heading-prefixed text used for embeddings</label>
</section>
<div class="toolbar"><p id="count" role="status"></p><a id="pdf-link" href="#">Open source PDF</a></div>
<div id="empty">No chunks match these filters.</div>
{''.join(cards)}
  </section>
  <section id="tables-view" class="review-view" hidden>
    <p>Compare each corrected table with the source page and the unchanged Docling grid.</p>
    {''.join(table_cards) if table_cards else '<p>No corrected tables found.</p>'}
  </section>
</main><script>{SCRIPT}</script></body></html>
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page, encoding="utf-8")
    return OUTPUT


def build_region_page(sample_size: int = 96) -> Path:
    """Replace the debug review page with a compact sampled-region view."""
    review = render_region_review(ROOT / "data" / "sources", OUTPUT, sample_size)
    def select_options(values: list[str]) -> str:
        return ''.join(f'<option value="{escape(value, quote=True)}">{escape(value)}</option>'
                       for value in values)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Manuora visual region review</title><style>{STYLE}</style></head><body>
<header><h1>Manuora visual region review</h1><p>Inspect the proposed image regions against their source PDF pages.</p></header>
<main><section id="regions-view" data-sample-id="{review['sample_id']}">
  <p>Check the red box and crop, then judge whether it has enough context and whether
  the suggested embedding route is right. Your choices are saved in this browser.
  Export JSON when you finish or want a backup; import it to resume elsewhere.</p>
  <div class="region-controls">
    <label>Document<select id="region-document"><option value="">All</option>{select_options(review['documents'])}</select></label>
    <label>Kind<select id="region-kind"><option value="">All</option>{select_options(review['kinds'])}</select></label>
    <label>Priority<select id="region-priority"><option value="">All</option>{select_options(review['priorities'])}</select></label>
    <label>Progress<select id="region-status"><option value="">All</option><option value="unreviewed">Unreviewed</option><option value="reviewed">Reviewed</option></select></label>
    <label class="checkbox"><input id="region-pinned" type="checkbox">Known examples only</label>
    <button id="region-export" type="button">Export decisions</button>
    <label>Import decisions<input id="region-import" type="file" accept=".json,application/json"></label>
  </div>
  <p id="region-progress" role="status"></p>
  {review['cards']}
</section></main><script>{REGION_SCRIPT}</script></body></html>
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page, encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-chunks", action="store_true",
                        help="Rebuild one document's experimental chunk JSONL before rendering")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--document-id")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--include-experimental", action="store_true",
                        help="Also embed raw and adapted Docling chunk comparisons")
    parser.add_argument("--region-sample-size", type=int, default=96,
                        help="Number of stratified visual-region cards (default: 96)")
    parser.add_argument("--regions-only", action="store_true",
                        help="Render sampled visual regions in the same debug review page")
    args = parser.parse_args()
    if args.regions_only:
        if args.refresh_chunks or args.manifest or args.document_id or args.include_experimental:
            parser.error("--regions-only cannot be combined with chunk-review options")
        print(build_region_page(args.region_sample_size))
        raise SystemExit(0)
    if args.refresh_chunks:
        if args.manifest is None:
            parser.error("--refresh-chunks requires --manifest")
        from manuora.ingestion.chunk_review import refresh_chunk_review

        output_dir = refresh_chunk_review(
            args.manifest, args.document_id, args.max_tokens, args.tokenizer,
        )
        print(f"Chunk review refreshed: {output_dir}")
    elif args.manifest or args.document_id:
        parser.error("--manifest and --document-id require --refresh-chunks")
    print(build_page(args.include_experimental))
