"""
backend/services/embedding_service.py
───────────────────────────────────────
Single Responsibility: manage the lifecycle of the text embedding model.

Design decisions:
  - Returns a HuggingFaceEmbeddings instance (LangChain-compatible).
  - The factory function is designed to be wrapped with @st.cache_resource
    in the UI layer, so the model is loaded once per Streamlit process.
  - The service itself has NO Streamlit dependency — it can be called from
    CLI scripts, FastAPI endpoints, or test suites without modification.
  - The embedding model name is injected via Settings, not hardcoded.

Liskov Substitution: the returned object satisfies LangChain's Embeddings
interface, so any other provider (OpenAI, Cohere, etc.) can replace it.

Usage
-----
    from backend.services.embedding_service import create_embedding_model

    embeddings = create_embedding_model()
"""

from langchain_huggingface import HuggingFaceEmbeddings

from backend.config.settings import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Instantiate and return a HuggingFace sentence-transformer embedding model.

    The model is loaded from HuggingFace Hub on first call.
    Subsequent calls (when wrapped with @st.cache_resource) return the
    cached instance immediately.

    Returns:
        HuggingFaceEmbeddings — LangChain-compatible embeddings object.
    """
    log.info("Loading embedding model: %s", settings.embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    log.info("Embedding model loaded successfully.")
    return embeddings
