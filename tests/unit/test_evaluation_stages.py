"""Shared experiment scoring: quote coverage on any passages and the stage report."""
from manuora.evaluation.coverage import quote_coverage, span_coverage
from manuora.evaluation.stages import lost_at, markdown, summarize

QUESTION = {"required_facts": [
    {"fact_id": "F35", "evidence": [{"version": "35", "quote": "Open the Settings menu."}]},
    {"fact_id": "F36", "evidence": [{"version": "36", "quote": "Use the new Admin panel"}]}],
    "accepted_alternatives": [
    {"fact_id": "F36", "evidence": [{"version": "36", "quote": "Admin panel moved"}]}]}


def chunk(version, text, cid="c"):
    return {"chunk_id": cid, "product_version": version, "text": text}


def test_quote_coverage_needs_matching_version_and_accepts_alternatives():
    both = [chunk("35", "First, open the settings menu!"), chunk("36", "The admin panel moved to the top bar.")]
    assert quote_coverage(QUESTION, both)["complete"]
    wrong_version = [chunk("36", "Open the Settings menu."), chunk("36", "Use the new Admin panel.")]
    result = quote_coverage(QUESTION, wrong_version)
    assert not result["complete"] and result["found"] == ["F36"]


def test_span_coverage_requires_every_group():
    gold = {"span_groups": [[{"id": "a", "quote": "step one"}], [{"id": "b", "quote": "step two"}, {"id": "c", "quote": "two"}]]}
    assert span_coverage(gold, [chunk(None, "Step one.", "a"), chunk(None, "Then two.", "c")])["complete"]
    assert span_coverage(gold, [chunk(None, "Step one.", "a")])["recall"] == 0.5


def test_stage_summary_and_loss_point():
    full, empty = {"complete": True, "recall": 1.0}, {"complete": False, "recall": 0.5}
    results = {"q1": {"A": {"candidates": full, "reranked": full, "packed": empty}},
               "q2": {"A": {"candidates": full, "reranked": full, "packed": full}}}
    summary = summarize(results)
    assert summary["A"]["packed"] == {"complete": 1, "n": 2, "recall": 0.75}
    assert lost_at(results["q1"]["A"]) == "packed" and lost_at(results["q2"]["A"]) is None
    assert "| A | 2/2 | 2/2 | 1/2 |" in markdown(summary)
