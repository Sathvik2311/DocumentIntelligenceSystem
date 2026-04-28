"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings populated from a local .env file or the process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Active LLM provider for generation. Switching is a one-line .env change.
    llm_provider: Literal["ollama", "gemini", "anthropic"] = Field(default="ollama")

    # --- Ollama (local) ---
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")

    # --- Google Gemini ---
    gemini_api_key: str = Field(default="", description="Google Gemini API key.")
    gemini_model: str = Field(default="gemini-2.0-flash")

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")
    anthropic_model: str = Field(default="claude-haiku-4-5")

    # --- Frontend ---
    backend_url: str = Field(default="http://localhost:8000", description="Base URL the Streamlit app uses to reach FastAPI.")

    # --- Embedding / store / retrieval ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chroma_persist_dir: str = Field(default="./chroma_db")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k: int = Field(default=5)
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
