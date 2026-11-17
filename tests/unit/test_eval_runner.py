"""Runner scoring, aggregation and the release gate, on synthetic replies (no models)."""
import json

import pytest

from manuora.evaluation import compare, run


@pytest.fixture(autouse=True)
def no_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "LABELS", tmp_path / "labels.jsonl")


WORKFLOW = {"id": "kc-w001", "category": "workflow", "expect": "answer", "product_id": "kicad", "versions": ["9"],
            "question": "q", "gold": {"span_groups": [[{"id": "a", "quote": "Add a symbol"}], [{"id": "b", "quote": ""}]]}}
NEGATIVE = {"id": "nc-n001", "category": "negative", "expect": "abstain", "product_id": "nextcloud", "versions": ["36"],
            "question": "How do I back up to Backblaze?", "gold": None}


def body(kind, answer="", packed=(), selected=None, outcome=None):
    p = [{"chunk_id": c, "text": t, "product_version": "9"} for c, t in packed]
    stages = {"candidates": p, "reranked": p, "selected": p if selected is None else selected, "packed": p}
    return {"kind": kind, "answer": answer, "trace": {"stages": stages, "outcome": outcome or kind}}


def test_parse_sets_builds_nested_overrides():
    assert run.parse_sets(["evidence.max_passages=12", "evidence.gate=relative", "grounding.term_check=null"]) == \
        {"evidence": {"max_passages": 12, "gate": "relative"}, "grounding": {"term_check": None}}
    with pytest.raises(SystemExit):
        run.parse_sets(["evidence.max_passages"])


def test_request_uses_a_single_named_version_only():
    assert run.request_for(WORKFLOW) == {"question": "q", "product_id": "kicad", "product_version": "9"}
    assert run.request_for(dict(WORKFLOW, versions=["35", "36"]))["product_version"] is None
    assert run.request_for(dict(WORKFLOW, versions=["None"]))["product_version"] is None


def test_score_case_retrieval_and_answer_tiers():
    full = [("a", "Then add a symbol."), ("b", "x")]
    r = run.score_case(WORKFLOW, body("retrieval_only", packed=full), 1.0, tier="retrieval")
    assert r["complete"]["packed"] and "answered" not in r
    a = run.score_case(WORKFLOW, body("answer", "Do it [1].", packed=full), 2.0)
    assert a["answered"] and a["cited_complete"] is False  # [1] alone misses group b
    assert run.score_case(WORKFLOW, body("answer", "Do it [1][2].", packed=full), 2.0)["cited_complete"]
    n = run.score_case(NEGATIVE, body("answer", "Use the Backblaze app [1]."), 1.0)
    assert n["declined"] is False and n["label_key"]
    assert run.score_case(NEGATIVE, body("abstain"), 1.0)["declined"] is True


def test_aggregate_counts_and_labels():
    full = [("a", "Then add a symbol."), ("b", "x")]
    records = [run.score_case(WORKFLOW, body("answer", "[1][2]", packed=full), 1.0),
               run.score_case(dict(WORKFLOW, id="kc-w002"), body("abstain", packed=full[:1]), 3.0),
               run.score_case(NEGATIVE, body("answer", "Use Backblaze [1]."), 2.0)]
    shown = records[2]["label_key"]
    rows = run.aggregate(records, {shown: "made_up"})
    assert rows["workflow"]["packed_complete"] == 1 and rows["workflow"]["cited_complete"] == 1
    assert rows["workflow"]["abstained"] == 1
    assert rows["negative"]["shown"] == 1 and rows["negative"]["made_up"] == 1 and rows["negative"]["unlabelled"] == 0
    assert rows["all"]["n"] == 3 and rows["all"]["p50_seconds"] == 2.0


def _side(tmp_path, name, packed_flags, tier="retrieval"):
    records = [{"id": f"kc-w{i:03d}", "category": "workflow", "expect": "answer", "private": False, "kind": "retrieval_only",
                "seconds": 1.0, "complete": {s: flag for s in run.STAGES}} for i, flag in enumerate(packed_flags)]
    folder = tmp_path / name
    folder.mkdir()
    meta = {"name": name, "split": "dev", "tier": tier}
    (folder / "scorecard.json").write_text(json.dumps({"meta": meta}), encoding="utf-8")
    (folder / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return folder


def test_gate_fails_on_a_two_case_drop_and_passes_within_tolerance(tmp_path):
    base = _side(tmp_path, "base", [True] * 5)
    report, ok = compare.compare(base, _side(tmp_path, "minus1", [True] * 4 + [False]))
    assert ok and "PASS workflow packed_complete: 5 -> 4" in report
    report, ok = compare.compare(base, _side(tmp_path, "minus2", [True] * 3 + [False] * 2))
    assert not ok and "**FAIL** workflow packed_complete" in report


def test_gate_refuses_different_tiers_and_compares_only_shared_cases(tmp_path):
    base = _side(tmp_path, "base", [True] * 3)
    _, ok = compare.compare(base, _side(tmp_path, "answer", [True] * 3, tier="answer"))
    assert not ok
    more = _side(tmp_path, "more", [True] * 3 + [False] * 4)  # four new bench cases, all failing
    report, ok = compare.compare(base, more)
    assert ok and "3 cases in both" in report
