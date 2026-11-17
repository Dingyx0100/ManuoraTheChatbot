# Manuora

Manuora is a self-hosted support chatbot that answers questions from versioned product manuals. It has been used with Nextcloud v12–36, QGIS 2.18–3.44, KiCad 5.1–10 and FreeCAD 0.16. Everything runs locally: Docling parsing, Qwen3 embeddings and reranker, and Qwen3-8B on Docker Model Runner.

When the manuals don't support an answer, Manuora says so instead of guessing.

## How a question is answered

```
question → rules router (direct / rewrite follow-up / clarify / edit previous answer)
         → hybrid retrieval: Qwen3-Embedding-0.6B in Qdrant + BM25 (SQLite FTS5), RRF k=60, 30 candidates each
         → Qwen3-Reranker-0.6B on the top 12, then evidence selection: keep 4, drop scores below 3.0
         → Qwen3-8B (2048-token context, 4 × 850-character excerpts) with LangGraph memory
         → citation and name checks, then answer or abstain
```

All settings are in [config/assistant.yaml](config/assistant.yaml).

## Folders

| Path | What's there |
|---|---|
| `backend/src/manuora/api/` | FastAPI app (`app.py`), routing and scope rules (`routing.py`), answer grounding checks (`grounding.py`) |
| `backend/src/manuora/generation/` | LangGraph conversation generator (`local.py`), follow-up and context router (`context.py`), Docker model load/unload (`model_control.py`) |
| `backend/src/manuora/retrieval/` | Qdrant indexing, BM25, hybrid fusion, reranker, evidence selection (`evidence.py`), optional features that are off by default (`steps.py`, `outlines.py`) |
| `backend/src/manuora/ingestion/` | PDF → profile → Docling parse → selective OCR → sections → tables → quality report → retrieval chunks |
| `backend/src/manuora/knowledge/` | Product manifest schemas |
| `backend/src/manuora/evaluation/` | Benchmark tools (`bench`), runner and scorecard (`run`), baseline comparison and release gate (`compare`), gold coverage scoring |
| `eval/` | The benchmark (`bench/cases.jsonl`), release gate rules (`gates.yaml`), baselines per tier and split (`baselines/`), holdout usage log |
| `frontend/index.html` | Chat page served by the app |
| `scripts/` | Pipeline entry points (below), `maintenance/` (parsing-quality tools), `review/` (HTML review pages) |
| `config/` | Assistant and embedding configs |
| `tests/unit/` | Unit tests; no models or manuals needed |

The manuals, parsed output, indexes and models live under `data/`, which is not in the repository.

## Setup

You need:
- Python 3.10 or later
- Docker Desktop with Model Runner enabled, serving `huggingface.co/qwen/qwen3-8b-gguf:q4_k_m`
- A Qdrant container named `manuora-qdrant` on port 6333

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[generation,ingestion,geometry,embeddings,test]"
docker run -d --name manuora-qdrant -p 6333:6333 qdrant/qdrant
```

The embedding and reranker models load with `local_files_only: true`, so download them once (the pinned revisions are in `config/`).

## Commands

```powershell
python scripts/ingest.py               # preview what would be (re)processed
python scripts/ingest.py --run         # manifest → parse → chunks → visual regions (skips up-to-date stages)
python scripts/index_embeddings.py     # Qdrant dense index
python scripts/index_sparse.py         # BM25 SQLite index
.\start_manuora.ps1                    # app on http://127.0.0.1:8000
python -m pytest tests/unit            # unit tests
```

Put source PDFs under `data/sources/<product>/<version>/`. `ingest.py` writes a `manifest.yaml` next to them and the processed output under `processed/<doc_id>/`.

### Configuration

`config/assistant.yaml` holds the defaults. These override it, in order:

1. `config/local.yaml` (optional, not in git), for machine-specific settings and local-only products.
2. Environment variables:
   - `MANUORA_CONFIG`: another base file
   - `MANUORA_DATA_DIR`: where `data/...` paths point
   - `MANUORA_LLM_URL`: the chat-completions endpoint
   - `MANUORA_QDRANT_URL`

`/health` reports that the process is up. `/ready` checks the settings, the BM25 index and Qdrant, and reports whether the chat model is loaded.

Manuora serves one user at a time: chat requests are serialized by a single lock.

## Testing and releases

There are three layers:

| What | Command | Needs |
|---|---|---|
| Unit tests (also run in CI on every push) | `python -m pytest tests/unit` | nothing |
| Retrieval benchmark: does the evidence reach the model? | `python -m manuora.evaluation.run --tier retrieval` | indexes, embedding and reranker models |
| Answer benchmark: answers, citations, abstentions | `python -m manuora.evaluation.run --tier answer` | the above, plus the chat model |

**The benchmark** is a single file, [eval/bench/cases.jsonl](eval/bench/cases.jsonl).
- It covers single facts, multi-step workflows, version questions, questions the manuals don't answer, and follow-ups.
- Each case has its gold evidence: manual passages, each with a quote.
- `dev` cases are for experiments. `holdout` cases are scored only for a release (`--release`), and every holdout run is logged in `eval/holdout_log.jsonl`.
- Maintenance commands:
  - `python -m manuora.evaluation.bench stats`: case counts
  - `python -m manuora.evaluation.bench review`: one page of cases nobody has reviewed yet
  - `python -m manuora.evaluation.bench mark reviewed <ids>`: record a review
- Local-only manuals keep their cases in `data/bench_private/cases.jsonl`, which uses the same schema and is not in git.

**A run** sends every case through the real chat path and writes `data/eval_runs/<run>/scorecard.md`.
- The scorecard shows, per category, how many cases still hold their full gold evidence after each stage (candidates → reranked → selected → packed).
- In the answer tier it also shows how many answers cite it, and how many questions the manuals don't answer were declined.
- **Labels:** an answer shown for such a question needs a person to label it once (`python -m manuora.evaluation.run label`). Labels are stored by answer, so each one is asked only once.

**Trying a change** (an experiment) never needs a new dataset:

```powershell
python -m manuora.evaluation.run --tier retrieval --name wide `
    --set evidence.gate=relative --set evidence.relative_margin=8 --set evidence.max_passages=12
python -m manuora.evaluation.compare data/eval_runs/<run>                # vs the baseline, with the gate
python -m manuora.evaluation.compare data/eval_runs/<a> data/eval_runs/<b>   # two arms
```

If a change exposes something the benchmark doesn't cover, add cases to `dev` and note them in [eval/bench/CHANGELOG.md](eval/bench/CHANGELOG.md).

**Releasing:**
1. Run the answer tier on `dev` and compare it with the baseline.
2. Run the answer tier on `holdout` with `--release` and compare.
3. If [the gate](eval/gates.yaml) passes, promote both runs (`python -m manuora.evaluation.compare promote data/eval_runs/<run>`). Each run becomes the baseline for its tier and split in `eval/baselines/`.
4. Commit and tag the release.
