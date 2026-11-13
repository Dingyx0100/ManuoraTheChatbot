"""Reference-constrained source-to-chunk preservation, without model inference."""

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from time import perf_counter

from manuora.ingestion.table_normalization import format_corrected_table_row


METRIC_VERSION = "2"
CONTENT_RECOVERY_FAIL_BELOW = 0.95
EXCLUDED_LABELS = {
    "document_index": "document index intentionally filtered by retrieval assembly",
    "page_header": "running page header excluded from searchable body",
    "page_footer": "running page footer excluded from searchable body",
}


def normalized_tokens(text: str) -> list[str]:
    """Lexical matching tokens, NOT embedding tokens or a code-fidelity score.

    Ignore layout punctuation/Markdown, normalize case and Unicode ligatures,
    and repair only line-break hyphenation. Preserve numbers and word-internal
    hyphens, dots, slashes plus comparison/arithmetic operators.
    """
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    return re.findall(r"\w+(?:[-./]\w+)*|[<>=+%]+", text)


def source_inventory(document, tables: list[dict], exclusions: dict[str, str] | None = None):
    """Inventory all saved blocks, including children omitted by chunk traversal.

    No exclusion is inferred from failure to chunk a block. Table text is
    represented once at its owning table; repaired tables use canonical rows.
    """
    exclusions = exclusions or {}
    nodes = {item.self_ref: item for item in [*document.texts, *document.tables,
                                             *document.pictures, *document.groups]}
    if set(exclusions) - nodes.keys() or any(not reason.strip() for reason in exclusions.values()):
        raise ValueError("Every explicit exclusion needs a known source block ID and a reason")
    repairs = {table["table_ref"]: table for table in tables}
    if len(repairs) != len(tables) or set(repairs) - {table.self_ref for table in document.tables}:
        raise ValueError("Corrected tables must have unique, existing source references")
    blocks, excluded = [], []

    def table_owner(item):
        seen = set()
        parent = item.parent
        while parent and parent.cref not in seen:
            seen.add(parent.cref)
            if parent.cref.startswith("#/tables/"):
                return parent.cref
            parent = getattr(nodes.get(parent.cref), "parent", None)
        return None

    def plain_cell(cell):
        if cell.text or not hasattr(cell, "ref"):
            return cell.text
        pending, seen, parts = [cell.ref.cref], set(), []
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

    def add(block_id, text, label, pages, **extra):
        tokens = normalized_tokens(text)
        if tokens:
            blocks.append(dict(block_id=block_id, tokens=tokens, label=label, pages=pages, **extra))
        else:
            excluded.append(dict(block_id=block_id, reason="no lexical text in text-only metric", pages=pages))

    for item in [*document.texts, *document.tables, *document.pictures]:
        ref, label = item.self_ref, str(item.label)
        pages = sorted({p.page_no for p in item.prov})
        owner = table_owner(item)
        reason = exclusions.get(ref) or EXCLUDED_LABELS.get(label)
        if not reason and item.content_layer.value == "furniture":
            reason = "furniture content layer intentionally excluded"
        if not reason and owner and label not in {"caption", "footnote"}:
            reason = f"text represented by owning table {owner}"
        if not reason and label == "picture":
            reason = "visual asset outside text-only preservation; text children evaluated separately"
        if reason:
            excluded.append(dict(block_id=ref, reason=reason, pages=pages))
            continue
        if ref in repairs:
            table = repairs[ref]
            excluded.append(dict(block_id=ref, reason="raw table superseded by canonical corrected rows", pages=pages))
            for index, row in enumerate(table["rows"]):
                add(f"{ref}/normalized_rows/{index}", format_corrected_table_row(table, row),
                    "corrected_table_row", pages, parent_block_id=ref,
                    raw_row_indices=row["raw_row_indices"])
        else:
            value = ("\n".join(plain_cell(cell) for cell in item.data.table_cells)
                     if ref.startswith("#/tables/") else getattr(item, "text", ""))
            add(ref, value, label, pages)
    return blocks, excluded


def _ranges(counts, predicate, tokens):
    spans, start = [], None
    for index in range(len(counts) + 1):
        selected = index < len(counts) and predicate(counts[index])
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            spans.append(dict(token_start=start, token_end=index,
                              excerpt=" ".join(tokens[start:index])[:240]))
            start = None
    return spans


def _align(source, target, counts, minimum_match_tokens, *, uncovered_only):
    """Credit ordered exact token runs; count each target position only once.

    Prefer uncovered source spans, then measure repeated source occurrences.
    Mask matched positions rather than deleting them (no artificial adjacency).
    """
    remaining = list(target)
    used = set()
    minimum = min(minimum_match_tokens, len(source))
    while True:
        available = [token if not uncovered_only or counts[i] == 0 else object()
                     for i, token in enumerate(source)]
        matches = [match for match in SequenceMatcher(None, available, remaining, autojunk=False).get_matching_blocks()
                   if match.size >= minimum]
        if not matches:
            break
        for match in matches:
            for offset in range(match.size):
                counts[match.a + offset] += 1
                remaining[match.b + offset] = object()
                used.add(match.b + offset)
    return used


def evaluate_chunk_preservation(document, chunks: list[dict], tables: list[dict], *,
                                parsed_sha256: str, source_pdf_sha256: str,
                                exclusions: dict[str, str] | None = None,
                                minimum_match_tokens: int = 3,
                                allowed_duplicate_tokens_per_block: int = 0,
                                include_all_blocks: bool = False) -> dict:
    """Compare canonical parsed blocks to final retrieval chunk text, by ID.

    Legacy refs are accepted with an explicit provenance warning. No corpus
    text search is used to rescue absent IDs. Repeated heading context is an
    intentional exception to duplication counting, not to recovery checking.
    """
    started = perf_counter()
    if minimum_match_tokens < 1 or allowed_duplicate_tokens_per_block < 0:
        raise ValueError("Invalid matching or duplication allowance")
    for table in tables:
        if table.get("source_pdf_sha256") != source_pdf_sha256 or table.get("parsed_sha256") != parsed_sha256:
            raise ValueError("Stale corrected table supplied to preservation evaluation")
    blocks, excluded = source_inventory(document, tables, exclusions)
    known = {block["block_id"] for block in [*blocks, *excluded]}
    by_ref, tokenized = defaultdict(list), {}
    unknown, legacy = set(), []
    chunk_ids = set()
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in chunk_ids:
            raise ValueError(f"Duplicate chunk ID: {cid}")
        chunk_ids.add(cid)
        if chunk.get("source_pdf_sha256") != source_pdf_sha256:
            raise ValueError(f"Chunk source PDF hash mismatch: {cid}")
        if chunk.get("parsed_sha256") not in (None, parsed_sha256):
            raise ValueError(f"Chunk parsed hash mismatch: {cid}")
        if "source_block_ids" not in chunk or chunk.get("parsed_sha256") is None:
            legacy.append(cid)
        refs = set(chunk.get("source_block_ids", chunk.get("refs", [])))
        if chunk.get("content_kind") == "corrected_table_row":
            refs.add(f"{chunk['table_ref']}/normalized_rows/{chunk['table_row_index']}")
        unknown.update(refs - known)
        for ref in refs:
            by_ref[ref].append(chunk)
        tokenized[cid] = (normalized_tokens(chunk["text"]),
                          normalized_tokens("\n".join(chunk.get("headings", []))))

    counts_by_ref = {block["block_id"]: [0] * len(block["tokens"]) for block in blocks}
    claimed_body_positions, claimed_heading_positions = defaultdict(set), defaultdict(set)
    # Allocate recovery to all source blocks before calling leftover occurrences
    # duplicates. One chunk occurrence cannot recover two separate body blocks.
    # Longest blocks first reduce accidental allocation to short generic phrases.
    for uncovered_only in (True, False):
        for block in sorted(blocks, key=lambda item: -len(item["tokens"])):
            ref = block["block_id"]
            heading = block["label"] in {"section_header", "title"}
            for chunk in by_ref[ref]:
                cid = chunk["chunk_id"]
                body, headings = tokenized[cid]
                target = [*body, object(), *headings] if heading else body
                used = claimed_heading_positions[ref, cid] if heading else claimed_body_positions[cid]
                target = [object() if index in used else token for index, token in enumerate(target)]
                used.update(_align(block["tokens"], target, counts_by_ref[ref], minimum_match_tokens,
                                   uncovered_only=uncovered_only))

    results = []
    for block in blocks:
        counts = counts_by_ref[block["block_id"]]
        candidates = by_ref[block["block_id"]]
        heading = block["label"] in {"section_header", "title"}
        recovered = sum(count > 0 for count in counts)
        repeated = sum(max(0, count - 1) for count in counts)
        # Heading repetition is supplied intentionally as retrieval context.
        excessive = max(0, repeated - allowed_duplicate_tokens_per_block) if not heading else 0
        results.append(dict(
            **{key: value for key, value in block.items() if key != "tokens"},
            referenced_chunk_ids=[chunk["chunk_id"] for chunk in candidates],
            eligible_tokens=len(counts), recovered_tokens=recovered,
            recovery_ratio=recovered / len(counts),
            omitted_spans=_ranges(counts, lambda count: count == 0, block["tokens"]),
            duplicated_spans=_ranges(counts, lambda count: count > 1, block["tokens"]),
            duplicate_token_occurrences=repeated,
            excessive_duplicate_token_occurrences=excessive,
            duplication_exclusion="intentional repeated heading context" if heading else None,
        ))
    total = sum(row["eligible_tokens"] for row in results)
    recovered = sum(row["recovered_tokens"] for row in results)
    duplicate = sum(row["excessive_duplicate_token_occurrences"] for row in results)
    missing = [row["block_id"] for row in results if not row["recovered_tokens"]]
    partial = [row["block_id"] for row in results if 0 < row["recovered_tokens"] < row["eligible_tokens"]]
    represented = sum(bool(row["referenced_chunk_ids"]) for row in results)
    return dict(
        metric_version=METRIC_VERSION, parsed_sha256=parsed_sha256, source_pdf_sha256=source_pdf_sha256,
        evaluation_wall_seconds=round(perf_counter() - started, 6),
        status=("FAIL" if total and recovered / total < CONTENT_RECOVERY_FAIL_BELOW else
                "WARN" if recovered < total or duplicate or unknown or legacy else
                "PASS" if total else "NOT_APPLICABLE"),
        settings=dict(normalization="nfkc_casefold_lexical_v1", minimum_match_tokens=minimum_match_tokens,
                      content_recovery_fail_below=CONTENT_RECOVERY_FAIL_BELOW,
                      content_recovery_warn_below=1.0,
                      allowed_duplicate_tokens_per_block=allowed_duplicate_tokens_per_block),
        eligible_blocks=len(results), represented_blocks=represented,
        block_reference_recall=represented / len(results) if results else None,
        fully_recovered_blocks=sum(row["recovered_tokens"] == row["eligible_tokens"] for row in results),
        eligible_tokens=total, recovered_tokens=recovered,
        chunk_content_recovery=recovered / total if total else None,
        excessive_duplicate_token_occurrences=duplicate,
        excessive_duplication_ratio=duplicate / total if total else None,
        missing_block_ids=missing, partially_recovered_block_ids=partial,
        unreferenced_block_ids=[row["block_id"] for row in results if not row["referenced_chunk_ids"]],
        unknown_source_block_ids=sorted(unknown), legacy_provenance_chunk_ids=legacy,
        excluded_blocks=excluded,
        block_details_policy="all" if include_all_blocks else "omitted_or_excessively_duplicated",
        blocks=results if include_all_blocks else [row for row in results
            if row["omitted_spans"] or row["excessive_duplicate_token_occurrences"]],
    )
