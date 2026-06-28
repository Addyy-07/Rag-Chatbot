"""
backend/services/multi_retriever_service.py
────────────────────────────────────────────
Cross-namespace retrieval: retrieve and merge results from N Pinecone namespaces.

Responsibility:
  - Given a question and a list of Pinecone namespaces, retrieve top-k chunks
    from each namespace independently.
  - Merge all retrieved chunks into a single deduplicated list.
  - Preserve relevance ordering within each namespace (Pinecone returns by score).

Design decisions:
  - Sequential retrieval (one namespace at a time) — avoids Pinecone rate limits
    and keeps the code simple. For SaaS scale, switch to asyncio.gather().
  - Deduplication by content hash — prevents the same chunk appearing twice if
    a document was accidentally uploaded to multiple namespaces.
  - Returns `list[Document]` — same type as a single retriever, so `_build_context()`
    in chat_service.py needs no changes.
  - Empty namespace list returns an empty list immediately (guard clause).

Chat modes → namespace lists
  📄 Single Doc  → ["<doc_uuid>"]
  📚 Selected    → ["<uuid_1>", "<uuid_2>", ...]
  🌐 All Docs    → registry.get_all_namespaces()

Usage
-----
    from backend.services.multi_retriever_service import retrieve_from_namespaces

    docs = retrieve_from_namespaces("What is RAG?", ["ns1", "ns2"], embeddings)
"""

import hashlib
from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.config.settings import settings
from backend.services.vector_store_service import ensure_index
from backend.utils.logger import get_logger

log = get_logger(__name__)


def _content_hash(doc: Document) -> str:
    """Return a stable SHA-256 hash of a document's page content."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _get_namespace_retriever(
    namespace: str,
    embeddings: HuggingFaceEmbeddings,
    top_k: Optional[int] = None,
):
    """
    Build a Pinecone retriever scoped to a single namespace.

    Args:
        namespace:  Pinecone namespace string (== document_id).
        embeddings: Shared embeddings singleton.
        top_k:      Number of chunks to retrieve. Defaults to settings.retriever_top_k.

    Returns:
        A LangChain VectorStoreRetriever scoped to the namespace.
    """
    index_name = ensure_index()
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    k = top_k or settings.retriever_top_k
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_from_namespaces(
    question: str,
    namespaces: list[str],
    embeddings: HuggingFaceEmbeddings,
    top_k_per_namespace: Optional[int] = None,
) -> list[Document]:
    """
    Retrieve and merge relevant document chunks across multiple namespaces.

    For each namespace, runs a similarity search and collects the top-k
    results. All results are merged, deduplicated by content hash, and
    returned as a flat list (preserving intra-namespace relevance order).

    Args:
        question:             The user's query string.
        namespaces:           List of Pinecone namespace strings to search.
                              Pass an empty list to get an empty result.
        embeddings:           Shared HuggingFaceEmbeddings singleton.
        top_k_per_namespace:  Chunks to retrieve per namespace.
                              Defaults to settings.retriever_top_k.

    Returns:
        Deduplicated list of LangChain Document objects.
        Length ≤ len(namespaces) × top_k_per_namespace.

    Raises:
        Any exception from Pinecone if a namespace is unreachable.
    """
    if not namespaces:
        log.warning("retrieve_from_namespaces called with empty namespace list.")
        return []

    log.info(
        "Multi-retrieval: querying %d namespace(s) for '%s...'",
        len(namespaces),
        question[:50],
    )

    all_docs: list[Document] = []
    seen_hashes: set[str] = set()

    for namespace in namespaces:
        try:
            retriever = _get_namespace_retriever(
                namespace, embeddings, top_k_per_namespace
            )
            docs = retriever.invoke(question)
            log.debug(
                "Namespace '%s': retrieved %d chunk(s).", namespace, len(docs)
            )

            # Deduplicate across namespaces
            for doc in docs:
                h = _content_hash(doc)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    # Annotate with source namespace for traceability
                    doc.metadata["source_namespace"] = namespace
                    all_docs.append(doc)

        except Exception as exc:
            log.error(
                "Failed to retrieve from namespace '%s': %s. Skipping.",
                namespace,
                exc,
            )
            # Gracefully degrade — continue with other namespaces

    log.info(
        "Multi-retrieval complete: %d unique chunk(s) from %d namespace(s).",
        len(all_docs),
        len(namespaces),
    )
    return all_docs
