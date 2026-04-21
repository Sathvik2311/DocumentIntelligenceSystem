## Commands
- `uvicorn backend.main:app --reload` — start FastAPI dev server (port 8000)
- `streamlit run frontend/app.py` — start Streamlit UI (port 8501)
- `pytest tests/` — run all tests
- `docker-compose up --build` — run full stack in Docker

## Stack
- Python 3.11, FastAPI, LangChain, ChromaDB, sentence-transformers
- Streamlit frontend
- Anthropic API (claude-haiku-4-5) for LLM

## Conventions
- All secrets via .env (never commit .env)
- Type hints and docstrings on all functions
- Pydantic schemas for all API models
- Structured JSON error responses
- Python logging, no print statements

## Architecture
- backend/services/ingestion.py — parse → chunk → embed → store
- backend/services/retrieval.py — embed query → similarity search → return chunks
- backend/services/generation.py — build prompt → call LLM → return answer + citations
- ChromaDB persists to ./chroma_db/ volume
