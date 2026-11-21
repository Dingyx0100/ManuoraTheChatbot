# Manuora

Manuora is a self-hosted support chatbot that answers questions from versioned product manuals. It has been used with Nextcloud v12–36, QGIS 2.18–3.44, KiCad 5.1–10 and FreeCAD 0.16. Everything runs locally: Docling parsing, Qwen3 embeddings and reranker, and Qwen3-8B on Docker Model Runner.

When the manuals don't support an answer, Manuora says so instead of guessing.

## How a question is answered

```
question → router (direct / rewrite follow-up / clarify / edit previous answer / chat), plus product and versions
         → hybrid retrieval: Qwen3-Embedding-0.6B in Qdrant + BM25 sparse vectors, fused by Qdrant (RRF k=60), 30 candidates each
         → Qwen3-Reranker-0.6B on the top 12, then evidence selection: keep 4, drop scores below 3.0
         → Qwen3-8B (2048-token context, 4 × 850-character excerpts) with LangGraph memory
         → citation and name checks, then answer or abstain
```

All settings are in [config/assistant.yaml](config/assistant.yaml).

## Folders

| Path | What's there |
|---|---|
| `backend/src/manuora/api/` | FastAPI app (`app.py`), routing and scope rules (`routing.py`), answer grounding checks (`grounding.py`) |
| `backend/src/manuora/generation/` | LangGraph conversation generator (`local.py`), context rules (`context.py`), the router and its shared checks (`router.py`), Docker model load/unload (`model_control.py`) |
| `backend/src/manuora/retrieval/` | Qdrant indexing, BM25, hybrid fusion, reranker, evidence selection (`evidence.py`), optional features that are off by default (`steps.py`, `outlines.py`) |
| `backend/src/manuora/ingestion/` | PDF → profile → Docling parse → selective OCR → sections → tables → quality report → retrieval chunks |
| `backend/src/manuora/knowledge/` | Product manifest schemas |
| `backend/src/manuora/evaluation/` | Benchmark tools (`bench`), runner and scorecard (`run`), baseline comparison and release gate (`compare`), gold coverage scoring |
| `eval/` | The benchmark (`bench/cases.jsonl`), release gate rules (`gates.yaml`), baselines per tier and split (`baselines/`), holdout usage log |
| `frontend/` | The web app: `index.html` shell, `assets/app.js` (chat, model controls), `assets/library.js` (Library page), `assets/app.css` (light and dark themes) |
| `backend/src/manuora/library/` | Library: manifests catalog (`catalog.py`), upload jobs (`jobs.py`), incremental add/remove in the indexes (`indexer.py`) |
| `scripts/` | Pipeline entry points (below), `maintenance/` (parsing-quality tools), `review/` (HTML review pages), `router/` (routing data, training, router server) |
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
python scripts/index_sparse.py         # SQLite chunk store (also the fallback FTS5 BM25 index)
python scripts/index_qdrant_lexical.py # BM25 sparse vectors: copies the dense collection into <name>__bm25
.\start_manuora.ps1                    # app on http://127.0.0.1:8000
python -m pytest tests/unit            # unit tests
```

Put source PDFs under `data/sources/<product>/<version>/`. `ingest.py` writes a `manifest.yaml` next to them and the processed output under `processed/<doc_id>/`.

### Adding manuals from the app

Open **Library** in the sidebar and drop PDFs. Each file gets a suggested product, version, title and type to confirm. A job then runs upload → parse → chunk → embed → index, one at a time, and the manual becomes searchable in chat when it reaches **Ready**.

- **Placement:** the PDF goes to `data/sources/<Product>/<version>/`, next to existing versions, with its own `manifest.yaml` entry. A manual with the same product, version and title replaces the old one.
- **Indexing:** incremental. Only the new manual's passages are embedded and added to the SQLite store and both Qdrant collections. Afterwards the three index manifests get the corpus hash of the chunk files on disk, so they stay consistent. The collection and index names keep the hash they were built with; a full rebuild (`index_embeddings.py`, `index_sparse.py`, `index_qdrant_lexical.py`) gives them a new name.
- **Remove** takes a manual out of all indexes and moves its PDF and processed files to `data/library/trash/`.
- **Parsing device:** when the chat model is loaded, parsing runs on the CPU (slower) so the two don't compete for GPU memory.
- **State:** jobs and their logs are in `data/library/`. A job interrupted by a restart can be retried from the step it reached.
- **Eval comparability:** adding or removing manuals changes retrieval results. Compare eval runs only on the same corpus.

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

### Router

The router decides what a message needs: search it as written, rewrite a follow-up into a standalone question,
ask for clarification, reformat the answer on screen, or reply as small talk. It also picks the product and
versions. Messages naming several products are searched once per product, and "latest" means the newest version
in the library (the answer says which).

`conversation.context_router_method` in `config/assistant.yaml`:

- `llm` (default): a fine-tuned Qwen3-0.6B served by llama-server. Start it with
  `python scripts/router/serve_router.py` (`--stop` to stop). It needs the router model in `data/models/`.
- If that server is unreachable or answers with unusable JSON, the TF-IDF classifier in
  `context_router_llm.fallback_classifier` routes instead. Without that file, the rules route.
- `rules` with `context_rules: v1`: the original rules router, no model files needed.

Every router's decision passes the same checks (`generation/router.py`, `checked`):

- Products and versions must exist in the library, and a product named in the question can't contradict the
  page selection.
- An edit needs an answer on screen; otherwise the router asks.
- For the learned router: a version the message names is kept even if the model dropped it, a product with one
  version gets that version, and a product id the library doesn't have, or one the message never names against
  the page selection, gives way to the product the rules found or the page selected.

The trained models are not in the repository yet.

#### How the router was trained

The router writes one JSON decision (`action`, `product`, `versions`, `needs`, `query`) for a prompt that lists
the library's products and versions, the page selection, whether an answer is on screen, the user's last two
messages, and the new message.

**Training data** (`scripts/router/build_mix.py`, about 2,800 training and 350 validation examples). It's built so
the weights can be shared:

| Source | What it gives | Licence |
|---|---|---|
| Synthetic messages written by Qwen3-8B from Nextcloud manual passages (`generate_data.py`) | all actions with products and versions: single and multi-step questions, version comparisons and corrections, follow-ups, edits, topic switches, several products, version formats and replies | Nextcloud documentation, CC BY 3.0 |
| Synthetic messages written from no manual | greetings, complaints about the assistant, references to something never named | — |
| Messages about 8 invented products (`fictional_products.py`, then `generate_data.py --fictional`) | the same scenarios as the Nextcloud set, on products with one or many versions and varied version formats; each example carries its own product list | invented by Qwen3-8B |
| [CANARD](https://sites.google.com/view/qanta/projects/canard) | follow-up rewrites, standalone follow-ups, first questions, and follow-ups without their history (clarify) | CC BY-SA 4.0 |
| [QReCC](https://github.com/apple/ml-qrecc), Natural Questions conversations only | the same roles | CC BY-SA 3.0 |
| [CLINC150](https://github.com/clinc/oos-eval), small-talk intents only | chat | CC BY 3.0 |

Public data is converted by `external_data.py`: one turn per conversation, a cap per source and action, each
dataset's dev split for validation and its test split never trained on, and at most 40% of training. QGIS-derived
messages are left out of training (the QGIS manuals are GFDL) and used as a test instead. Qwen3-8B and
Qwen3-0.6B-Base are Apache 2.0.

**Training** (`train_lora.py`): LoRA (r 16, alpha 32, attention and MLP) on Qwen3-0.6B-Base, loss on the JSON only,
effective batch 16, learning rate 2e-4, best epoch on validation. `merge_lora.py` merges the adapter, and
llama.cpp's `convert_hf_to_gguf.py` writes the Q8_0 GGUF that llama-server runs. The fallback classifier
(`train_classifier.py --method tfidf`) is trained on the same data.

**Evaluation** (`python -m manuora.evaluation.run --tier routing`, and the routing-only arms of the experiment):
tests no training row comes from: KiCad and FreeCAD messages, typo-styled messages, a separately generated set
with new wording, QGIS messages, 60 hand-written messages, and the public datasets' test splits. On the
benchmark's dev routing cases it routes 204 of 207 correctly, at about 125 ms per message on an RTX 5060 Ti with
the 8B chat model loaded.

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
