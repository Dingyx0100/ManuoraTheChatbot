"""Final evidence step: gate modes and packing limits."""
import pytest

from manuora.generation.local import format_evidence
from manuora.retrieval.evidence import select_evidence


def hits(*scores):
    return [{"chunk_id": str(i), "rerank_score": s} for i, s in enumerate(scores)]


def ids(selected):
    return [h["chunk_id"] for h in selected]


def test_absolute_gate_matches_original_app_behaviour():
    ranked = hits(6.0, 4.0, 3.0, 2.9, 1.0)
    assert ids(select_evidence(ranked)) == ["0", "1", "2"]
    assert ids(select_evidence(ranked, max_passages=2)) == ["0", "1"]


def test_relative_gate_follows_the_top_score_with_optional_floor():
    low = hits(2.0, 1.5, 0.2, -3.0)
    assert select_evidence(low) == []  # absolute 3.0 rejects everything
    assert ids(select_evidence(low, gate="relative", relative_margin=2.0, minimum_score=None)) == ["0", "1", "2"]
    assert ids(select_evidence(low, gate="relative", relative_margin=2.0, minimum_score=1.0)) == ["0", "1"]
    assert ids(select_evidence(hits(9.0, 8.5, 3.0), gate="relative", relative_margin=1.0,
                               max_passages=8)) == ["0", "1"]


def test_invalid_gate_settings_fail_loudly():
    with pytest.raises(ValueError):
        select_evidence(hits(1.0), gate="fuzzy")
    with pytest.raises(ValueError):
        select_evidence(hits(1.0), gate="relative", relative_margin=None)
    with pytest.raises(ValueError):
        select_evidence(hits(1.0), max_passages=0)


def test_config_section_can_be_passed_whole():
    config = {"max_passages": 1, "gate": "absolute", "minimum_score": 3.0,
              "relative_margin": None, "passage_char_limit": 850}
    assert ids(select_evidence(hits(5.0, 4.0), **config)) == ["0"]


def test_passage_char_limit_none_keeps_whole_text():
    hit = {"document_id": "d", "pages": [1], "headings": ["H"], "text": "x" * 1000}
    assert format_evidence([hit]).count("x") == 850
    assert format_evidence([hit], passage_char_limit=None).count("x") == 1000
