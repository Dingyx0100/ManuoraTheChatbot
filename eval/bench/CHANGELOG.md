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
