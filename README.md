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
| `backend/src/manuora/evaluation/` | Retrieval metrics, gold-set coverage scoring, stage-wise reports |
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
python -m pytest                       # unit tests
```

Put source PDFs under `data/sources/<product>/<version>/`. `ingest.py` writes a `manifest.yaml` next to them and the processed output under `processed/<doc_id>/`.
