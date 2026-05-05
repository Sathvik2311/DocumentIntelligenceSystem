# Single image used by both the FastAPI backend and the Streamlit frontend.
# Each service in docker-compose.yml overrides the `command:` so the API container
# launches uvicorn while the UI container launches streamlit.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers

# build-essential is required by some scientific deps (chromadb's hnswlib, etc.).
# Trim it after install to keep the image small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches when only source files change.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy the full source tree.
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY ingest.py query.py ./

# Pre-create the on-disk Chroma directory so the volume mount has a clean target.
RUN mkdir -p /app/chroma_db /cache/huggingface /cache/sentence-transformers

EXPOSE 8000 8501

# Default command runs the API; the frontend service overrides this in compose.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
