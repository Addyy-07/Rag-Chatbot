"""
backend/services/vector_store_service.py
─────────────────────────────────────────
Single Responsibility: manage Pinecone index lifecycle and vector store access.

Responsibilities:
  1. Ensure the Pinecone index exists (create if absent).
  2. Return a LangChain PineconeVectorStore bound to the index.
  3. Expose a namespace-scoped retriever configured from Settings.
  4. Upsert document chunks into a specified namespace.

Namespace support (v2):
  All public functions now accept an optional ``namespace`` parameter.
  Callers that omit it get the old behaviour (default namespace "").
  This preserves full backward compatibility with existing tests.

Open/Closed Principle: to swap Pinecone for Weaviate or Chroma, only this
file changes; services that call get_retriever() are untouched.

Usage
-----
    from backend.services.vector_store_service import (
        ensure_index,
        get_vector_store,
        get_retriever,
        upsert_documents,
    )

    # Single namespace (legacy / default)
    retriever = get_retriever(embeddings)

    # Namespace-scoped (multi-doc)
    retriever = get_retriever(embeddings, namespace="abc-uuid-123")
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
        log.debug("Pinecone index '%s' already exists.", index_name)

    return index_name


def get_vector_store(
    embeddings: HuggingFaceEmbeddings,
    namespace: str = "",
) -> PineconeVectorStore:
    """
    Return a LangChain PineconeVectorStore bound to our index and namespace.

    Args:
        embeddings: An instantiated LangChain-compatible embeddings object.
        namespace:  Pinecone namespace string. Empty string = default namespace.

    Returns:
        A PineconeVectorStore ready for similarity search or upsert.
    """
    index_name = ensure_index()
    log.debug(
        "Connecting to PineconeVectorStore (index='%s', namespace='%s').",
        index_name,
        namespace or "<default>",
    )
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )


def get_retriever(
    embeddings: HuggingFaceEmbeddings,
    namespace: str = "",
) -> VectorStoreRetriever:
    """
    Return a configured VectorStoreRetriever for similarity search.

    Args:
        embeddings: An instantiated LangChain-compatible embeddings object.
        namespace:  Pinecone namespace string. Empty = default namespace.

    Returns:
        A VectorStoreRetriever instance scoped to the namespace.
    """
    vector_store = get_vector_store(embeddings, namespace=namespace)
    return vector_store.as_retriever(
        search_kwargs={"k": settings.retriever_top_k}
    )


def upsert_documents(
    chunks: list,
    embeddings: HuggingFaceEmbeddings,
    namespace: str = "",
) -> None:
    """
    Embed and upsert a list of document chunks into a Pinecone namespace.

    Args:
        chunks:     List of LangChain Document objects (split chunks).
        embeddings: An instantiated LangChain-compatible embeddings object.
        namespace:  Target Pinecone namespace. Empty = default namespace.
    """
    index_name = ensure_index()
    log.info(
        "Upserting %d chunks to Pinecone index='%s', namespace='%s'...",
        len(chunks),
        index_name,
        namespace or "<default>",
    )
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name,
        namespace=namespace,
    )
    log.info("Upsert complete.")
