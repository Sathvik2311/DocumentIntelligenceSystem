# RAG Document Intelligence System

A production-style Retrieval-Augmented Generation web app: upload PDFs / DOCX / TXT, ask questions in natural language, get grounded answers with chunk-level citations. Pluggable LLM provider (local Ollama by default; Google Gemini and Anthropic Claude one env-var away). Hybrid retrieval (BM25 + cosine via Reciprocal Rank Fusion) followed by a cross-encoder reranker. Built-in eval harness with Hit@k, MRR, and LLM-judge faithfulness scoring.

## Stack

- **Backend** — FastAPI · Pydantic v2 · structured JSON errors
- **Retrieval** — ChromaDB (persistent, cosine HNSW) · sentence-transformers (`all-MiniLM-L6-v2`) · `rank-bm25` · `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generation** — pluggable: Ollama (`llama3.2`) · Google Gemini (`gemini-2.0-flash`) · Anthropic Claude (`claude-haiku-4-5`)
- **Frontend** — Streamlit (chat UI with per-doc scoping, inline upload, citation previews, A/B toggles for hybrid + rerank)
- **Tests** — 62 pytest tests + an eval harness with golden Q&A pairs
- **Container** — Docker + docker-compose

## Quickstart (Docker)

```bash
cp .env.example .env
# Edit .env — at minimum, pick LLM_PROVIDER. Ollama is the free default.

docker compose up --build
```

- Backend API → http://localhost:8000  (Swagger UI at `/docs`)
- Streamlit UI → http://localhost:8501

If you're using the default Ollama provider, run `ollama serve` and `ollama pull llama3.2` on the host first; the containers reach it via `host.docker.internal`.

To wipe the corpus and start fresh:
```bash
docker compose down -v
```

## Quickstart (local / no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Terminal 1
uvicorn backend.main:app --reload

# Terminal 2
streamlit run frontend/app.py
```

## CLI

```bash
python ingest.py path/to/file.pdf
python query.py "what are the key findings?" --top-k 3
python query.py "..." --retrieval-only          # skip the LLM, dump chunks only
```

## Architecture

```
upload  →  parse (PyMuPDF / python-docx / TXT)
        →  chunk (tiktoken cl100k_base, 1000/200 sliding window, page-aware)
        →  embed (sentence-transformers)
        →  ChromaDB (cosine HNSW, persistent volume)

query   →  embed                 ─┐
        →  BM25 keyword search   ─┴→  Reciprocal Rank Fusion (k=60)
                                              │
                                              ▼
                                  cross-encoder rerank (top-20 → top-5)
                                              │
                                              ▼
                                  prompt(system + retrieved + history + question)
                                              │
                                              ▼
                                  LLM provider (Ollama / Gemini / Anthropic)
                                              │
                                              ▼
                                  answer + [N] citations + token usage
```

Each retrieval stage is independently toggleable via `.env` (`ENABLE_HYBRID_SEARCH`, `ENABLE_RERANKER`) or per-request fields (`use_hybrid`, `use_reranker`).

## Tests & evals

```bash
pytest tests/ -q                                              # 62 tests, ~9 s
python -m tests.eval.run_eval --ablate                        # cosine vs hybrid vs hybrid+rerank
python -m tests.eval.run_eval --ablate --with-llm             # adds LLM-judge faithfulness
```

Sample ablation output:

```
| mode                | hit@5  | mrr   | faithfulness | n  |
|---------------------|--------|-------|--------------|----|
| cosine              | 0.700  | 0.650 |   —          | 10 |
| hybrid              | 0.700  | 0.650 |   —          | 10 |
| hybrid+rerank       | 0.700  | 0.700 |   —          | 10 |
```

## Project layout

```
backend/
  main.py                    FastAPI app + structured error handlers
  config.py                  pydantic-settings (env-driven config)
  routers/
    documents.py             POST /upload, GET /, DELETE /{id}
    query.py                 POST /query
  services/
    ingestion.py             parse → chunk → embed → persist
    retrieval.py             dense + sparse + RRF + rerank pipeline
    bm25.py                  in-process BM25 index over the Chroma collection
    reranker.py              cross-encoder
    generation.py            provider abstraction (Ollama / Gemini / Anthropic)
  models/schemas.py          Pydantic v2 request/response models
frontend/
  app.py                     Streamlit chat UI
tests/
  test_*.py                  unit + integration coverage (62 cases)
  eval/                      golden Q&A pairs + run_eval.py + LLM judge
ingest.py                    CLI entry — `python ingest.py file.pdf`
query.py                     CLI entry — `python query.py "..."`
Dockerfile                   single image, used by both services
docker-compose.yml           backend + frontend + chroma volume
```

See [CLAUDE.md](CLAUDE.md) for developer commands and conventions.
