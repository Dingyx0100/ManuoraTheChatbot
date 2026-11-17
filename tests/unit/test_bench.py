"""The benchmark file stays valid, the holdout stays separate from dev, and scoring works."""
import re

from manuora.evaluation import bench


def _key(text):
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def test_public_bench_is_valid_and_public_only():
    cases = bench.read_file(bench.PUBLIC)
    assert cases, "eval/bench/cases.jsonl is missing or empty"
    assert bench.validate(cases, public=True) == []


def test_private_bench_is_valid_when_present():
    private = bench.read_file(bench.PRIVATE)
    assert bench.validate(private, public=False) == []
    ids = [c["id"] for c in bench.read_file(bench.PUBLIC) + private]
    assert len(ids) == len(set(ids)), "an id is used in both the public and the private file"


def test_holdout_shares_no_question_with_dev():
    cases = bench.read_file(bench.PUBLIC) + bench.read_file(bench.PRIVATE)
    dev = {_key(c["question"]) for c in cases if c["split"] == "dev"}
    shared = [c["id"] for c in cases if c["split"] == "holdout" and _key(c["question"]) in dev]
    assert not shared, f"holdout questions also in dev: {shared}"


def test_evidence_score_handles_both_gold_forms():
    spans = {"gold": {"span_groups": [[{"id": "a", "quote": "Open the menu"}], [{"id": "b", "quote": ""}, {"id": "c", "quote": ""}]]}}
    passages = [{"chunk_id": "a", "text": "First, open the menu.", "product_version": "9"}]
    assert bench.evidence_score(spans, passages) == {"complete": False, "recall": 0.5}
    assert bench.evidence_score(spans, passages + [{"chunk_id": "c", "text": "x"}])["complete"]
    facts = {"gold": {"required_facts": [{"fact_id": "f", "evidence": [{"version": "35", "quote": "Initial Setup"}]}],
                      "accepted_alternatives": []}}
    assert bench.evidence_score(facts, [{"chunk_id": "z", "text": "see Initial Setup", "product_version": "35"}])["complete"]
    assert not bench.evidence_score(facts, [{"chunk_id": "z", "text": "see Initial Setup", "product_version": "36"}])["complete"]


def test_validate_catches_common_mistakes():
    good = dict(id="x-s001", category="single", split="dev", status="checked", product_id="qgis", versions=["3.44"],
                question="q", expect="answer", gold={"span_groups": [[{"id": "a", "quote": ""}]]})
    assert bench.validate([good], public=True) == []
    assert bench.validate([good, dict(good)], public=True)  # duplicate id
    assert bench.validate([dict(good, product_id="secretproduct")], public=True)  # not a public product
    assert bench.validate([dict(good, gold=None)], public=True)  # answerable without gold
    assert bench.validate([dict(good, split="test")], public=True)
