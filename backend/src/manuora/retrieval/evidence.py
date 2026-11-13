"""Final evidence step: which reranked passages may reach the generator.

The generator then packs a ranked prefix of these into its token budget
(`ManuoraGraphGenerator._prepare_evidence`). Both halves are configured in the
`evidence:` section of config/assistant.yaml so experiments vary configuration
instead of re-implementing selection.
"""

from __future__ import annotations

GATES = ("absolute", "relative")


def select_evidence(ranked: list[dict], *, max_passages: int = 4, gate: str = "absolute",
                    minimum_score: float | None = 3.0, relative_margin: float | None = None,
                    **_: object) -> list[dict]:
    """Keep up to `max_passages` hits from a reranked list (highest score first).

    absolute: keep hits scoring >= minimum_score (the original app behaviour).
    relative: keep hits scoring >= top score - relative_margin, and >= minimum_score
              when a floor is given. Adapts to questions whose scores are all low.
    Extra keys (packing settings from the same config section) are ignored.
    """
    if gate not in GATES:
        raise ValueError(f"unknown evidence gate {gate!r}; expected one of {GATES}")
    if max_passages < 1:
        raise ValueError("max_passages must be positive")
    if not ranked:
        return []
    if gate == "relative":
        if relative_margin is None or relative_margin < 0:
            raise ValueError("relative gate needs a non-negative relative_margin")
        threshold = ranked[0]["rerank_score"] - relative_margin
        if minimum_score is not None:
            threshold = max(threshold, minimum_score)
    else:
        if minimum_score is None:
            raise ValueError("absolute gate needs minimum_score")
        threshold = minimum_score
    return [hit for hit in ranked if hit["rerank_score"] >= threshold][:max_passages]
