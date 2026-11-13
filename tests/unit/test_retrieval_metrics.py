import pytest

from manuora.evaluation.metrics import aggregate, score_query


def test_ranked_evidence_and_version_metrics() -> None:
    gold = {
        "id": "q1", "product": "nextcloud", "version": "34", "category": "procedure",
        "relevance": {"right_a": 3, "right_b": 1},
        "relevant_sections": ["5.2.2"], "relevant_pages": [26],
    }
    ranked = [
        {"chunk_id": "old", "product_id": "nextcloud", "product_version": "35",
         "headings": ["5.2.2 Files"], "pages": [26]},
        {"chunk_id": "right_b", "product_id": "nextcloud", "product_version": "34",
         "headings": ["5.2.2 Files"], "pages": [26]},
        {"chunk_id": "right_a", "product_id": "nextcloud", "product_version": "34",
         "headings": ["5.2.2 Files"], "pages": [27]},
    ]
    score = score_query(gold, ranked, (1, 2, 3))
    assert score["recall@1"] == 0
    assert score["recall@2"] == 1
    assert score["fractional_recall@2"] == 0.5
    assert score["mrr"] == 0.5
    assert score["raw_mrr"] == 0.5
    assert score["section_hit@1"] == 0
    assert score["page_hit@2"] == 1
    assert score["wrong_version@1"] == 1
    assert 0 < score["ndcg@3"] < 1
    assert aggregate([score])["category:procedure"]["mrr"] == 0.5


def test_unanswerable_query_is_excluded_from_positive_evidence_metrics() -> None:
    score = score_query({"id": "q2", "product": "qgis", "version": "3.28"}, [], (5,))
    assert score["answerable"] is False
    assert score["recall@5"] is None
    assert score["mrr"] is None
    assert aggregate([score])["all"]["answerable_queries"] == 0


def test_related_vector_parts_count_once_and_show_split_diagnostics() -> None:
    gold = {
        "id": "q3", "product": "qgis", "version": "3.44", "document_id": "guide",
        "relevance": {"target": 3, "other": 1},
        "relevant_vector_chunk_ids": ["target:2"],
    }
    ranked = [
        {"chunk_id": "wrong", "vector_chunk_id": "wrong:1", "parent_chunk_id": "wrong",
         "product_id": "qgis", "product_version": "3.44", "document_id": "guide"},
        {"chunk_id": "wrong", "vector_chunk_id": "wrong:2", "parent_chunk_id": "wrong",
         "product_id": "qgis", "product_version": "3.44", "document_id": "guide"},
        {"chunk_id": "target", "vector_chunk_id": "target:1", "parent_chunk_id": "target",
         "product_id": "qgis", "product_version": "3.44", "document_id": "guide"},
        {"chunk_id": "target", "vector_chunk_id": "target:2", "parent_chunk_id": "target",
         "product_id": "qgis", "product_version": "3.44", "document_id": "guide"},
        {"chunk_id": "other", "vector_chunk_id": "other:1", "parent_chunk_id": "other",
         "product_id": "qgis", "product_version": "3.44", "document_id": "guide"},
    ]
    score = score_query(gold, ranked, (1, 2, 3, 4),
                        split_part_counts={"target": 2, "other": 1})
    assert score["raw_vector_results"] == 5
    assert score["unique_parent_results"] == 3
    assert score["duplicate_parts@2"] == 1
    assert score["mrr"] == 0.5
    assert score["raw_mrr"] == pytest.approx(1 / 3)
    assert score["recall@2"] == 1
    assert score["raw_recall@2"] == 0
    assert score["fractional_recall@3"] == 1
    assert score["raw_fractional_recall@3"] == 0.5
    assert score["split_recall@2"] == 1
    assert score["unsplit_recall@2"] == 0
    assert score["part_hit@3"] == 0
    assert score["part_hit@4"] == 1
    assert score["part_mrr"] == 0.25
    assert score["evidence_split"] == "mixed"
    assert aggregate([score])["manual:qgis/3.44/guide"]["recall@2"] == 1
    with pytest.raises(ValueError, match="duplicate retrieved vector ID"):
        score_query(gold, ranked + [ranked[0]])
