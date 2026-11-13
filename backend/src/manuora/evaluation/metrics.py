"""Model-independent retrieval metrics for a frozen gold set and ranked chunks."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean


DEFAULT_K = (1, 3, 5, 10)


def _same_scope(gold: dict, chunk: dict) -> bool:
    return (
        chunk.get("product_id") == gold.get("product")
        and (gold.get("version") is None or chunk.get("product_version") == gold["version"])
        and (gold.get("document_id") is None or chunk.get("document_id") == gold["document_id"])
    )


def _section_match(section: str, headings: list[str]) -> bool:
    target = section.strip().casefold()
    if not target:
        return False
    numbered = bool(re.fullmatch(r"\d+(?:\.\d+)*", target))
    for heading in headings:
        value = heading.strip().casefold()
        if value == target:
            return True
        if numbered and re.match(rf"^{re.escape(target)}(?:\s|$)", value):
            return True
    return False


def _dcg(gains: list[int]) -> float:
    return sum((2 ** gain - 1) / math.log2(rank + 2)
               for rank, gain in enumerate(gains))


def _parent_id(chunk: dict) -> str:
    return str(chunk.get("parent_chunk_id") or chunk["chunk_id"])


def score_query(gold: dict, ranked: list[dict], ks: tuple[int, ...] = DEFAULT_K,
                *, split_part_counts: dict[str, int] | None = None) -> dict:
    """Score distinct source chunks, preserving raw vector-rank diagnostics."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive ranks")
    vector_ids = [str(chunk.get("vector_chunk_id") or chunk["chunk_id"]) for chunk in ranked]
    if len(vector_ids) != len(set(vector_ids)):
        raise ValueError(f"duplicate retrieved vector ID for {gold['id']}")
    parents = []
    seen = set()
    for chunk in ranked:
        parent = _parent_id(chunk)
        if parent not in seen:
            parents.append(chunk)
            seen.add(parent)
    relevance = {str(key): int(value) for key, value in gold.get("relevance", {}).items()}
    if any(value < 0 for value in relevance.values()):
        raise ValueError("relevance grades must be nonnegative")
    for chunk_id in gold.get("relevant_chunk_ids", []):
        relevance.setdefault(chunk_id, 1)
    positive = {chunk_id for chunk_id, value in relevance.items() if value > 0}
    sections = gold.get("relevant_sections", [])
    pages = set(gold.get("relevant_pages", []))
    gains = [relevance.get(_parent_id(chunk), 0) if _same_scope(gold, chunk) else 0
             for chunk in parents]
    first = next((rank for rank, gain in enumerate(gains, 1) if gain > 0), None)
    raw_first = next((rank for rank, chunk in enumerate(ranked, 1)
                      if _same_scope(gold, chunk) and relevance.get(_parent_id(chunk), 0) > 0), None)
    relevant_parts = set(gold.get("relevant_vector_chunk_ids", []))
    part_first = next((rank for rank, chunk in enumerate(ranked, 1)
                       if _same_scope(gold, chunk) and
                       chunk.get("vector_chunk_id") in relevant_parts), None)
    split_positive = unsplit_positive = None
    evidence_split = None
    if split_part_counts is not None and positive:
        missing = positive - split_part_counts.keys()
        if missing:
            raise ValueError(f"gold evidence absent from split map for {gold['id']}: {sorted(missing)}")
        split_positive = {chunk_id for chunk_id in positive if split_part_counts[chunk_id] > 1}
        unsplit_positive = positive - split_positive
        evidence_split = ("mixed" if split_positive and unsplit_positive else
                          "split" if split_positive else "unsplit")
    result = {
        "id": gold["id"],
        "product": gold.get("product"),
        "version": gold.get("version"),
        "document_id": gold.get("document_id"),
        "category": gold.get("category"),
        "answerable": bool(positive or relevant_parts or sections or pages),
        "evidence_split": evidence_split,
        "raw_vector_results": len(ranked),
        "unique_parent_results": len(parents),
        "mrr": (1 / first if first else 0.0) if positive else None,
        "raw_mrr": (1 / raw_first if raw_first else 0.0) if positive else None,
        "part_mrr": (1 / part_first if part_first else 0.0) if relevant_parts else None,
    }
    for k in sorted(set(ks)):
        prefix = parents[:k]
        raw_prefix = ranked[:k]
        found = {_parent_id(chunk) for chunk, gain in zip(prefix, gains[:k]) if gain > 0}
        raw_found = {_parent_id(chunk) for chunk in raw_prefix
                     if _same_scope(gold, chunk) and relevance.get(_parent_id(chunk), 0) > 0}
        result[f"recall@{k}"] = float(bool(found)) if positive else None
        result[f"fractional_recall@{k}"] = len(found) / len(positive) if positive else None
        result[f"raw_recall@{k}"] = float(bool(raw_found)) if positive else None
        result[f"raw_fractional_recall@{k}"] = len(raw_found) / len(positive) if positive else None
        result[f"part_hit@{k}"] = (
            float(any(_same_scope(gold, chunk) and
                      chunk.get("vector_chunk_id") in relevant_parts for chunk in raw_prefix))
            if relevant_parts else None
        )
        result[f"duplicate_parts@{k}"] = len(raw_prefix) - len({_parent_id(chunk) for chunk in raw_prefix})
        result[f"split_recall@{k}"] = (
            len(found & split_positive) / len(split_positive) if split_positive else None
        )
        result[f"unsplit_recall@{k}"] = (
            len(found & unsplit_positive) / len(unsplit_positive) if unsplit_positive else None
        )
        ideal = _dcg(sorted(relevance.values(), reverse=True)[:k]) if positive else 0.0
        result[f"ndcg@{k}"] = _dcg(gains[:k]) / ideal if ideal else None
        result[f"section_hit@{k}"] = (
            float(any(_same_scope(gold, chunk) and
                      any(_section_match(section, chunk.get("headings", [])) for section in sections)
                      for chunk in prefix)) if sections else None
        )
        result[f"page_hit@{k}"] = (
            float(any(_same_scope(gold, chunk) and pages.intersection(chunk.get("pages", []))
                      for chunk in prefix)) if pages else None
        )
        result[f"wrong_version@{k}"] = (
            float(any(chunk.get("product_id") == gold.get("product")
                      and chunk.get("product_version") != gold["version"]
                      for chunk in prefix)) if gold.get("version") is not None else None
        )
    return result


def aggregate(scores: list[dict]) -> dict:
    """Macro-average available metrics overall and by product/category."""
    groups: dict[str, list[dict]] = defaultdict(list)
    groups["all"] = scores
    for score in scores:
        for field in ("product", "category", "evidence_split"):
            if score.get(field):
                groups[f"{field}:{score[field]}"].append(score)
        if score.get("document_id"):
            manual = f"{score.get('product')}/{score.get('version')}/{score['document_id']}"
            groups[f"manual:{manual}"].append(score)
    output = {}
    for name, rows in groups.items():
        metrics = sorted({key for row in rows for key in row
                          if key in {"mrr", "raw_mrr", "part_mrr"} or "@" in key})
        output[name] = {"queries": len(rows),
                        "answerable_queries": sum(row["answerable"] for row in rows)}
        for metric in metrics:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            output[name][metric] = mean(values) if values else None
    return output
