# Benchmark changelog

One entry per batch of added, reviewed or retired cases. The cases are in `cases.jsonl` (public manuals). Cases for local-only manuals are kept outside the repository, and their counts are not listed here.

## 2026-11-17: initial migration
- 361 public cases taken from the legacy question sets by `scripts/eval/migrate_legacy.py`.
- **Holdout (162):**
  - Source: `manuora_test_v1`–`v4`.
  - 45 single, 39 workflow, 25 version, 53 negative.
  - Used only for release checks.
- **Dev (199):**
  - Sources: `workflow_dev_v1`, `workflow_packing_v0_1`, `workflow_pilot_v0_1`, `grounding_dev_v1`, `version_dev_v1`, `manuora_gold_v0_2`, `version_comparison_v0_1`, `conversation_retrieval_v0_3` (contextual cases).
  - 52 single, 48 workflow, 40 version, 43 negative, 16 conversation.
- **Status:** every case is `checked`, meaning its gold anchors resolved against the current index. None is `reviewed` yet.

## 2026-11-17: routing cases (experiment R)
- **Schema:**
  - optional `ui` (what the chat page had selected)
  - optional `route` (expected routing decision)
  - `product_id: ""` for cross-product conversation cases
  - tag `routing_only` for cases only the routing tier scores
- **8 clarify/edit cases moved into the public file** (`x-c001`–`x-c008`, from `conversation_retrieval_v0_3` c17–c24). They contain no local-only manuals, but had been filed as private because they have no single product. Fixes:
  - `x-c003`/`x-c004` regain the page selection (Nextcloud 36) their conflict depends on.
  - `x-c006` drops a turn that belonged to another conversation.
- **41 new dev cases** `rt-c001`–`rt-c041`, `draft`, `routing_only`. Hand-written before rules v2 existed:
  - 10 follow-ups
  - 8 self-contained questions containing it/that/them/instead
  - 4 references with no history
  - 3 answer edits
  - 2 version corrections
  - 3 switches and returns
  - 2 two-product questions
  - 4 two-version comparisons
  - 3 unusual version forms
  - 2 small-talk lines
- **Pending:** they are scored only after review (`python -m manuora.evaluation.bench review --category conversation`).
- **Relabelled `x-c005`** (user runs Nextcloud 12 and 36, asks "Which side is the restore icon on?") from clarify to answer for both versions: rewrite, versions 12 and 36. Decision by the user: answering for both is more helpful than asking which server.

## 2026-11-20: two products are answered, not clarified (experiment R, round 3)
- User decision: a question naming two products is searched once per product and answered for each.
  Clarify only when the question itself is unclear.
- **Relabelled (2, both dev drafts, routing only):** `rt-c031`, `rt-c032`: clarify → direct, with `route.product`
  a list of both products and no versions.
- `route.product` may now be a list (every product, searched separately).
