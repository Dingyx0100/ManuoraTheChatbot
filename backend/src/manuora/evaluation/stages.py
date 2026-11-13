"""Stage-wise evidence report: at which pipeline stage does a question lose its evidence?

An experiment records, per case and arm, the passages that survive each stage, e.g.

    {"candidates": [...], "reranked": [...], "selected": [...], "packed": [...]}

and passes a `score(case, passages) -> {"complete": bool, "recall": float}` function
(see manuora.evaluation.coverage). The report counts complete cases per stage, so a
change shows exactly which stage it moved instead of only the final number.
"""
from __future__ import annotations

from statistics import mean

STAGES = ("candidates", "reranked", "selected", "packed")


def score_stages(case, stages: dict[str, list], score) -> dict:
    return {name: score(case, passages) for name, passages in stages.items()}


def summarize(results: dict[str, dict[str, dict]]) -> dict:
    """results[case_id][arm][stage] = score dict  ->  summary[arm][stage]."""
    summary: dict[str, dict] = {}
    for per_arm in results.values():
        for arm, per_stage in per_arm.items():
            for stage, s in per_stage.items():
                summary.setdefault(arm, {}).setdefault(stage, []).append(s)
    return {arm: {stage: dict(complete=sum(s["complete"] for s in scores), n=len(scores),
                              recall=round(mean(s["recall"] for s in scores), 3))
                  for stage, scores in per_stage.items()}
            for arm, per_stage in summary.items()}


def lost_at(per_stage: dict[str, dict], order=STAGES) -> str | None:
    """First stage at which a case that was complete earlier stops being complete."""
    seen_complete = False
    for stage in [s for s in order if s in per_stage]:
        if per_stage[stage]["complete"]:
            seen_complete = True
        elif seen_complete:
            return stage
    return None


def markdown(summary: dict, order=STAGES) -> str:
    stages = [s for s in order if any(s in v for v in summary.values())]
    lines = ["| arm | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    for arm, per_stage in summary.items():
        cells = [f"{per_stage[s]['complete']}/{per_stage[s]['n']}" if s in per_stage else "" for s in stages]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
