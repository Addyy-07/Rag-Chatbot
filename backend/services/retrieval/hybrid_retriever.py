"""
backend/services/retrieval/hybrid_retriever.py
───────────────────────────────────────────────
Hybrid Search Retriever combining Sparse (BM25) and Dense (Pinecone MMR) search.

Uses LangChain's EnsembleRetriever to merge the results of a BM25 keyword search
and a Pinecone Maximum Marginal Relevance (MMR) vector search using
Reciprocal Rank Fusion (RRF).

For the sparse component, raw text chunks are dynamically fetched from MongoDB
to build an ephemeral BM25 index in memory.
"""

import hashlib
from typing import Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever

from backend.config.settings import settings
from backend.db.mongo_client import db_manager
from backend.services.vector_store_service import ensure_index
from backend.utils.logger import get_logger

log = get_logger(__name__)


def _content_hash(doc: Document) -> str:
    """Return a stable SHA-256 hash of a document's page content."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _get_bm25_retriever_for_namespace(namespace: str, top_k: int) -> Optional[BM25Retriever]:
    """
    Fetch raw chunks from MongoDB and build a BM25Retriever for the namespace.
    Returns None if no chunks are found in MongoDB.
    """
    try:
        db = db_manager.get_db()
        cursor = db.document_chunks.find({"namespace": namespace})
        docs = []
        for row in cursor:
            docs.append(Document(page_content=row["text"], metadata=row.get("metadata", {})))
        
        if not docs:
            log.warning("No chunks found in MongoDB for namespace '%s'. BM25 will be skipped.", namespace)
            return None
            
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = top_k
        return retriever
    except Exception as exc:
        log.error("Failed to build BM25 retriever for namespace '%s': %s", namespace, exc)
        return None


def _get_pinecone_retriever_for_namespace(
    namespace: str,
    embeddings: HuggingFaceEmbeddings,
    top_k: int,
):
    """
    Build a Pinecone retriever scoped to a single namespace with MMR enabled.
    """
    index_name = ensure_index()
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    # Enable Maximum Marginal Relevance (MMR) for diversity
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 2}
    )


def retrieve_hybrid(
    question: str,
    namespaces: list[str],
    embeddings: HuggingFaceEmbeddings,
    top_k_per_namespace: Optional[int] = None,
) -> list[Document]:
    """
    Perform hybrid retrieval across multiple namespaces.
    
    For each namespace:
      1. Initializes a Dense MMR Retriever (Pinecone).
      2. Initializes a Sparse Retriever (BM25 via MongoDB).
      3. Uses EnsembleRetriever to merge and rank them (weights: 50/50).
      
    Merges and deduplicates results across all namespaces.
    """
    if not namespaces:
        log.warning("retrieve_hybrid called with empty namespace list.")
        return []

    log.info(
        "Hybrid retrieval: querying %d namespace(s) for '%s...'",
        len(namespaces),
        question[:50],
    )

    all_docs: list[Document] = []
    seen_hashes: set[str] = set()
    k = top_k_per_namespace or settings.retriever_top_k

    for namespace in namespaces:
        try:
            dense_retriever = _get_pinecone_retriever_for_namespace(namespace, embeddings, k)
            sparse_retriever = _get_bm25_retriever_for_namespace(namespace, k)

            if sparse_retriever:
                # Hybrid search using EnsembleRetriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[dense_retriever, sparse_retriever],
                    weights=[0.5, 0.5]
                )
                docs = ensemble_retriever.invoke(question)
                log.debug("Namespace '%s': retrieved %d chunks via HYBRID search.", namespace, len(docs))
            else:
                # Fallback to dense only if MongoDB chunks are missing
                docs = dense_retriever.invoke(question)
                log.debug("Namespace '%s': retrieved %d chunks via DENSE-ONLY search.", namespace, len(docs))

            # Deduplicate across namespaces
            for doc in docs:
                h = _content_hash(doc)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    doc.metadata["source_namespace"] = namespace
                    all_docs.append(doc)

        except Exception as exc:
            log.error(
                "Failed to retrieve from namespace '%s': %s. Skipping.",
                namespace,
                exc,
            )

    log.info(
        "Hybrid retrieval complete: %d unique chunk(s) from %d namespace(s).",
        len(all_docs),
        len(namespaces),
    )
    return all_docs
