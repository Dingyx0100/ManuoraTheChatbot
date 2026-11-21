"""Synthetic routing examples (scripts/router/generate_data.py) as routing-tier cases (experiment R).

Each example becomes a case the routing tier scores exactly like a benchmark case: the same history replay,
the same post-router checks, the same metrics, grouped by scenario. Only `split == "train"` rows may be used
to fit or prompt a router; `val` picks checkpoints; `test_unseen` and `test_style` are scored once.
"""
from __future__ import annotations

import random
from pathlib import Path

from manuora.evaluation import bench
from manuora.evaluation.common import ROOT

DATA = ROOT / "data/router"
SPLITS = ("train", "val", "test_unseen", "test_style", "test_phrasing")


def load(name: str, splits=SPLITS) -> list[dict]:
    rows = bench.read_file(DATA / name / "examples.jsonl")
    return [r for r in rows if r["split"] in splits]


def as_case(example: dict) -> dict:
    lab = example["label"]
    expect = "answer" if lab["action"] in ("direct", "rewrite") else lab["action"]
    return {"id": example["id"], "question": example["question"], "history": example.get("history") or [],
            "ui": example.get("ui") or {"product_id": None, "product_version": None},
            "category": example["scenario"], "expect": expect, "product_id": lab.get("product") or "",
            "versions": lab.get("versions") or [], "tags": [], "_private": False,
            "route": {"action": lab["action"], "product": lab.get("product"), "versions": lab.get("versions") or [],
                      "needs": "many" if len(lab.get("needs") or []) >= 2 else "one"}}


def docs_for(example: dict, documents: list[dict]) -> list[dict]:
    """The products the example's chat page offers: its own `catalog` ({product: [versions]}, used for invented
    products) or the library's documents."""
    catalog = example.get("catalog")
    if not catalog:
        return documents
    return [{"product_id": p, "product_version": str(v), "document_id": f"{p}-{v}"} for p, vs in catalog.items() for v in vs]


def final_context(example: dict, documents: list[dict], rules: str = "v2") -> dict | None:
    """The context the router sees for the example's message (earlier turns routed by the v2 rules).
    None when an early reply (greeting, thanks) answers it before any router runs."""
    from manuora.evaluation.run import replay
    from manuora.generation.context import ContextRouter

    context, _ = replay(as_case(example), docs_for(example, documents), ContextRouter("rules"), rules)(example["question"])
    return context


def few_shot(examples: list[dict], documents: list[dict], n: int = 12, seed: int = 5) -> list[list[str]]:
    """n (input, output) pairs from the train split, spread over scenarios, for the untuned and 8B arms."""
    from manuora.generation.router import known_versions, label_json, router_input

    assert all(e["split"] == "train" for e in examples), "few-shot examples must come from the train split"
    rng = random.Random(seed)
    by_scenario: dict[str, list[dict]] = {}
    for e in examples:
        by_scenario.setdefault(e["scenario"], []).append(e)
    catalog = known_versions(documents)
    shots = []
    scenarios = sorted(by_scenario)
    rng.shuffle(scenarios)
    for scenario in (scenarios * 2)[:n]:
        for e in rng.sample(by_scenario[scenario], len(by_scenario[scenario])):
            context = final_context(e, documents)
            if context is not None and not any(s[0].endswith(e["question"]) for s in shots):
                shots.append([router_input(context, catalog), label_json(e["label"])])
                break
    return shots


def evaluate(examples: list[dict], conversation: dict, documents: list[dict], router=None) -> list[dict]:
    """Routing-tier records for synthetic examples under these conversation settings."""
    from manuora.evaluation.run import make_router, route_case

    router = router or make_router(conversation, documents)
    rules = conversation.get("context_rules", "v1")
    records = []
    for e in examples:
        record = route_case(as_case(e), docs_for(e, documents), router, rules)
        record.update(split=e["split"], style=e["style"])
        records.append(record)
    return records


def save_shots(shots: list[list[str]], path: Path) -> None:
    bench.write_file(path, shots)
