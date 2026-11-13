"""Whole-page native/parsed text quantity checks (not transcription accuracy).

No OCR, model inference or downloads. Tokenization uses a cached, pinned text
tokenizer on CPU. Whitespace is normalized identically before token counting;
characters exclude whitespace. Page ratios are intentionally not capped.
"""

from functools import lru_cache
from pathlib import Path

import pypdfium2 as pdfium

from manuora.ingestion.files import sha256_file
from manuora.ingestion.profile_schemas import TextCoverageSettings


def character_count(text: str) -> int:
    return sum(not char.isspace() for char in text)


@lru_cache(maxsize=4)
def _load_tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True, trust_remote_code=False,
    )


def token_count(tokenizer, text: str) -> int:
    # Actual tokenizer IDs, without prompts, special tokens or truncation.
    return len(tokenizer.encode(
        " ".join(text.split()), add_special_tokens=False, truncation=False,
    ))


def parsed_page_text(document) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Read saved text, including furniture and text nested beneath pictures.

    Table cells contribute once, without Markdown delimiters. Ambiguous
    multipage items make affected pages unmeasurable rather than assigning the
    entire item to every page and overestimating retention.
    """
    fragments = {page: [] for page in document.pages}
    notes = {page: [] for page in document.pages}
    nodes = {item.self_ref: item for item in [*document.texts, *document.groups,
                                             *document.pictures, *document.tables]}

    def inside_table(item):
        if str(item.label) in {"caption", "footnote"}:
            return False
        seen = set()
        parent = item.parent
        while parent and parent.cref not in seen:
            seen.add(parent.cref)
            if parent.cref.startswith("#/tables/"):
                return True
            node = nodes.get(parent.cref)
            parent = node.parent if node else None
        return False

    def cell_text(cell):
        if cell.text or not hasattr(cell, "ref"):
            return cell.text
        # Rich cells can store text exclusively in referenced child nodes.
        parts = []
        pending = [cell.ref.cref]
        seen = set()
        while pending:
            ref = pending.pop()
            if ref in seen or ref not in nodes:
                continue
            seen.add(ref)
            node = nodes[ref]
            if getattr(node, "text", ""):
                parts.append(node.text)
            pending.extend(child.cref for child in reversed(node.children))
        return "\n".join(parts)

    for item in [*document.texts, *document.tables]:
        if inside_table(item):
            continue  # Rich table cells already supply their text below.
        is_table = item.self_ref.startswith("#/tables/")
        value = ("\n".join(cell_text(cell) for cell in item.data.table_cells)
                 if is_table else item.text)
        # Docling keeps list markers outside item.text. The native PDF counts
        # include them; omitting them can fail otherwise intact list-only pages.
        marker = getattr(item, "marker", "") if not is_table else ""
        if marker and not value.lstrip().startswith(marker):
            value = f"{marker} {value}"
        if not value.strip():
            continue
        pages = {prov.page_no for prov in item.prov if prov.page_no in fragments}
        if len(pages) == 1:
            fragments[next(iter(pages))].append(value)
        elif len(pages) > 1:
            spans = sorted(set((prov.charspan[0], prov.charspan[1], prov.page_no)
                               for prov in item.prov if prov.page_no in pages))
            valid = not is_table and all(0 <= a < b <= len(value) for a, b, _ in spans)
            valid = valid and all(b <= c for (_, b, _), (c, _, _) in zip(spans, spans[1:]))
            # Missing portions cannot be silently assigned to an arbitrary page.
            valid = valid and character_count("".join(value[a:b] for a, b, _ in spans)) == character_count(value)
            if valid:
                for start, end, page in spans:
                    fragments[page].append(value[start:end])
            else:
                for page in pages:
                    notes[page].append(f"Unassignable multipage text: {item.self_ref}")
        # Items without provenance remain covered by the separate orphan metric.
    return {page: "\n".join(parts) for page, parts in fragments.items()}, notes


def measure_text_coverage(profile, document, source_path: Path | None,
                          settings: TextCoverageSettings) -> dict:
    if settings.fail_below > settings.warn_below:
        raise ValueError("text coverage fail_below must not exceed warn_below")
    parsed, notes = parsed_page_text(document)
    native_chars = {page.page_number: page.text_char_count for page in profile.page_profiles}
    native_tokens = {}
    parsed_tokens = {}
    messages = []
    tokenizer = None
    token_status = "unavailable_without_source_pdf"
    if source_path is not None:
        if sha256_file(source_path) != profile.source_sha256:
            raise ValueError("Native text coverage source hash differs from profile")
        try:
            tokenizer = _load_tokenizer(settings.tokenizer_id, settings.tokenizer_revision)
            token_status = "measured"
        except (OSError, ImportError, ValueError) as error:
            token_status = "unavailable_tokenizer"
            messages.append(f"Token coverage unavailable: {type(error).__name__}; cache the configured tokenizer")
        pdf = pdfium.PdfDocument(source_path)
        try:
            if len(pdf) != profile.page_count:
                raise ValueError("Native text coverage source page count differs from profile")
            for index in range(len(pdf)):
                page = pdf[index]
                try:
                    text_page = page.get_textpage()
                    try:
                        text = text_page.get_text_range()
                    finally:
                        text_page.close()
                    native_chars[index + 1] = character_count(text)
                    if tokenizer is not None:
                        native_tokens[index + 1] = token_count(tokenizer, text)
                finally:
                    page.close()
        finally:
            pdf.close()
    else:
        messages.append("Token coverage requires source_path; using saved native character counts")
    if tokenizer is not None:
        parsed_tokens = {page: token_count(tokenizer, text) for page, text in parsed.items()}

    rows = {}
    for page in profile.page_profiles:
        number = page.page_number
        n_chars = native_chars[number]
        p_chars = character_count(parsed[number]) if not notes[number] else None
        n_tokens = native_tokens.get(number)
        p_tokens = parsed_tokens.get(number) if not notes[number] else None
        char_ratio = p_chars / n_chars if n_chars and p_chars is not None else None
        tok_ratio = p_tokens / n_tokens if n_tokens and p_tokens is not None else None
        eligible = page.body_text_char_count >= profile.thresholds.min_body_text_chars
        ratios = [value for value in (char_ratio, tok_ratio) if value is not None]
        if not eligible:
            status = "NOT_APPLICABLE"
        elif ratios and min(ratios) < settings.fail_below:
            status = "FAIL"
        elif notes[number] or char_ratio is None or tok_ratio is None:
            status = "UNMEASURED"
        elif min(ratios) < settings.warn_below:
            status = "WARN"
        else:
            status = "PASS"
        rows[number] = dict(
            native_text_chars=n_chars, parsed_coverage_chars=p_chars,
            native_text_tokens=n_tokens, parsed_text_tokens=p_tokens,
            native_character_coverage_ratio=char_ratio,
            native_token_coverage_ratio=tok_ratio,
            coverage_status=status, coverage_notes=notes[number],
        )

    eligible_rows = [rows[page.page_number] for page in profile.page_profiles
                     if page.body_text_char_count >= profile.thresholds.min_body_text_chars]

    def weighted_ratio(native_key, parsed_key):
        if not eligible_rows or any(row[native_key] is None or row[parsed_key] is None
                                    for row in eligible_rows):
            return None
        total = sum(row[native_key] for row in eligible_rows)
        # Excess text on one page cannot compensate for missing text elsewhere.
        return (sum(min(row[parsed_key], row[native_key]) for row in eligible_rows) / total
                if total else None)

    return dict(
        pages=rows,
        native_character_coverage_ratio=weighted_ratio("native_text_chars", "parsed_coverage_chars"),
        native_token_coverage_ratio=weighted_ratio("native_text_tokens", "parsed_text_tokens"),
        text_coverage_settings=settings,
        text_coverage_scope="whole_page_saved_parse_including_furniture_picture_text_and_list_markers; counts_not_content_recall",
        token_counting_status=token_status,
        text_coverage_failed_pages=[page for page, row in rows.items() if row["coverage_status"] == "FAIL"],
        text_coverage_warning_pages=[page for page, row in rows.items() if row["coverage_status"] == "WARN"],
        text_coverage_unmeasured_pages=[page for page, row in rows.items() if row["coverage_status"] == "UNMEASURED"],
        text_coverage_notes=messages,
    )
