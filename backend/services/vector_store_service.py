"""
backend/services/vector_store_service.py
─────────────────────────────────────────
Single Responsibility: manage Pinecone index lifecycle and vector store access.

Responsibilities:
  1. Ensure the Pinecone index exists (create if absent).
  2. Return a LangChain PineconeVectorStore bound to the index.
  3. Expose a retriever configured from Settings.

Design decisions:
  - All Pinecone configuration (cloud, region, metric, dimension) comes from
    Settings — zero hardcoded strings in this file.
  - `ensure_index()` is idempotent: safe to call on every app start.
  - `get_vector_store()` is designed to be cached by the caller (@st.cache_resource).
  - No Streamlit imports — fully portable.

Open/Closed Principle: to swap Pinecone for Weaviate or Chroma, only this
file changes; services that call `get_retriever()` are untouched.

Usage
-----
    from backend.services.vector_store_service import (
        ensure_index,
        get_vector_store,
        get_retriever,
    )
"""

from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from backend.config.settings import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)


def ensure_index() -> str:
    """
    Guarantee the configured Pinecone index exists.

    Creates the index with the correct dimension and metric if it doesn't
    already exist. Safe to call on every startup (idempotent).

    Returns:
        The Pinecone index name from settings.

    Raises:
        pinecone.exceptions.PineconeException: if the Pinecone API is
        unreachable or the API key is invalid.
    """
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name
    existing_names = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_names:
        log.info("Pinecone index '%s' not found — creating...", index_name)
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dimension,
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        log.info("Pinecone index '%s' created successfully.", index_name)
    else:
        log.info("Pinecone index '%s' already exists.", index_name)

    return index_name


def get_vector_store(embeddings: HuggingFaceEmbeddings) -> PineconeVectorStore:
    """
    Return a LangChain PineconeVectorStore bound to our index.

    Calls `ensure_index()` first so the index always exists before access.

    Args:
        embeddings: An instantiated LangChain-compatible embeddings object.

    Returns:
        A PineconeVectorStore ready for similarity search or upsert.
    """
    index_name = ensure_index()
    log.debug("Connecting to PineconeVectorStore (index='%s').", index_name)
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )


def get_retriever(embeddings: HuggingFaceEmbeddings) -> VectorStoreRetriever:
    """
    Return a configured VectorStoreRetriever for similarity search.

    The number of documents to retrieve (top-k) is controlled by
    ``settings.retriever_top_k``.

    Args:
        embeddings: An instantiated LangChain-compatible embeddings object.

    Returns:
        A VectorStoreRetriever instance.
    """
    vector_store = get_vector_store(embeddings)
    return vector_store.as_retriever(
        search_kwargs={"k": settings.retriever_top_k}
    )


def upsert_documents(chunks: list, embeddings: HuggingFaceEmbeddings) -> None:
    """
    Embed and upsert a list of document chunks into Pinecone.

    Args:
        chunks:     List of LangChain Document objects (split chunks).
        embeddings: An instantiated LangChain-compatible embeddings object.
    """
    index_name = ensure_index()
    log.info("Upserting %d chunks to Pinecone index '%s'...", len(chunks), index_name)
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name,
    )
    log.info("Upsert complete.")
