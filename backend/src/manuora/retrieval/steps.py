"""Step searches for how-to questions (candidate, not wired into the app yet; see experiments/workflow_pool).

1. `decompose_prompt` asks the chat model for one search query per distinct step.
2. Each query is retrieved, reranked and gated on its own.
3. `merge_steps` combines the lists: the question's best passage, then each step's best unused passages
   round-robin (in step order), then the rest of the question's own selection.
"""
from __future__ import annotations

import json
import re

MAX_STEPS = 5

DECOMPOSE_PROMPT = (
    "You plan searches in the {product} {version} manual for a user's question.\n"
    "Split the task into the separate actions or features that the manual would describe in different sections, "
    "and write one short search query for each, in the order the user would do them (2 to 4 queries). "
    "Every query must name a different action, setting or command. Never rephrase the same step twice.\n"
    "If the question asks for a single fact or a single setting, write exactly one query.\n"
    "Use the manual's likely wording (menu, dialog, setting or command names) and leave out the user's story.\n"
    'Example for a photo app: "How do I put my holiday pictures in an album and send it to my family?" -> '
    '{{"queries": ["create a new album", "add photos to an album", "share an album by link"]}}\n'
    'Reply with JSON only: {{"queries": [...]}}\n\nQuestion: {question}')

# Appended to the generator's system prompt.
STEPS_ANSWER_RULE = (" For a task with several steps, answer as a numbered list of steps in the order the user should"
                     " do them; end every step with the citation of the passage that supports it.")


def decompose_prompt(question: str, product: str, versions: list) -> str:
    named = [str(v) for v in versions if v not in (None, "None")]
    return DECOMPOSE_PROMPT.format(product=product, version=" / ".join(named), question=question)


def parse_queries(text: str) -> list[str]:
    """Queries from the model's JSON reply; [] when it is unusable (the caller then searches the question only)."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        queries = json.loads(m.group(0))["queries"] if m else []
    except (ValueError, KeyError, TypeError):
        return []
    if not isinstance(queries, list):
        return []
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:MAX_STEPS]


def merge_steps(question_hits: list[dict], step_hits: list[list[dict]], per_step: int = 1,
                max_passages: int = 12) -> list[dict]:
    """question_hits / step_hits: already reranked and gated lists of passages with `chunk_id`."""
    out, seen = [], set()

    def take(hit):
        if hit["chunk_id"] in seen:
            return False
        seen.add(hit["chunk_id"]); out.append(hit)
        return True

    for hit in question_hits[:1]:
        take(hit)
    offered = [iter(hits) for hits in step_hits]
    for _ in range(per_step):
        for it in offered:
            next((hit for hit in it if take(hit)), None)
    for hit in question_hits[1:]:
        take(hit)
    return out[:max_passages]
