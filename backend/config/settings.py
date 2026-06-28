"""
backend/config/settings.py
──────────────────────────
Single source of truth for every environment variable in the application.

Uses Pydantic's BaseSettings so that:
  - All values are type-validated at startup (fail fast on misconfiguration).
  - Variables are read from the OS environment *and* from a .env file automatically.
  - Defaults are documented right here — no more hunting through os.getenv() calls.

Usage
-----
    from backend.config.settings import settings

    api_key = settings.groq_api_key
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.

    All fields map 1-to-1 to environment variables (case-insensitive).
    Required fields (no default) raise a ValidationError at startup if missing.
    """

    # ── Pinecone ────────────────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "rag-chatbot-free"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"

    # ── Groq LLM ────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.0
    groq_max_tokens: int = 512

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Ingestion ───────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ───────────────────────────────────────────────────────────
    retriever_top_k: int = 3
    retriever_context_max_chars: int = 500

    # ── Chat history ────────────────────────────────────────────────────────
    chat_history_window: int = 2        # Number of past exchanges to include
    chat_history_max_chars: int = 200   # Truncation limit per message

    # ── Document Registry ────────────────────────────────────────────────────────
    registry_path: str = ".document_registry.json"   # Flat-file metadata store
    max_upload_files: int = 10                         # Max simultaneous uploads

    # ── Auth API ──────────────────────────────────────────────────────────────────
    api_base_url: str = "http://localhost:8000/api/v1"

    # ── App metadata ────────────────────────────────────────────────────────────
    app_title: str = "DocChat AI"
    app_icon: str = "🧠"
    app_version: str = "1.0.0"

    # ── Pydantic config ──────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",          # Load from .env if present
        env_file_encoding="utf-8",
        case_sensitive=False,     # PINECONE_API_KEY == pinecone_api_key
        extra="ignore",           # Silently ignore unknown env vars
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton Settings instance.

    Using lru_cache(maxsize=1) means Settings() is only constructed once
    per process — subsequent calls return the same object instantly.
    This is the recommended pattern for Pydantic settings in production.
    """
    return Settings()


# Module-level convenience alias so callers can do:
#   from backend.config.settings import settings
settings: Settings = get_settings()
