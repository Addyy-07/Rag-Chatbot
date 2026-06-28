"""
backend/services/retrieval/reranker.py
────────────────────────────────────────
Cross-Encoder Reranker for advanced context compression.

Takes a list of retrieved documents and scores them against the query
using a HuggingFace Cross-Encoder model. Documents that score below
a defined threshold are discarded, and the rest are sorted by relevance.
"""

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from backend.utils.logger import get_logger

log = get_logger(__name__)

# Singleton Cross-Encoder model to avoid reloading on every query
_cross_encoder_model = None

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder_model
    if _cross_encoder_model is None:
        log.info("Loading Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
        _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder_model


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
    relevance_threshold: float = -5.0, # Cross-encoders can output negative logits
) -> List[Document]:
    """
    Rerank a list of documents using a Cross-Encoder and filter out low-relevance chunks.
    
    Args:
        query: The rewritten standalone query.
        documents: The candidate documents retrieved from hybrid search.
        top_k: Maximum number of documents to return after reranking.
        relevance_threshold: The minimum score required to keep a document.
        
    Returns:
        A list of Document objects, sorted by descending relevance.
    """
    if not documents:
        return []
        
    log.info("Reranking %d documents...", len(documents))
    
    model = get_cross_encoder()
    
    # CrossEncoder expects pairs of (query, document_text)
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)
    
    # Zip docs with their scores
    scored_docs = list(zip(documents, scores))
    
    # Sort by descending score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Filter by threshold and take top_k
    final_docs = []
    for doc, score in scored_docs:
        if score >= relevance_threshold:
            # Annotate with reranker score for debugging/UI if needed
            doc.metadata["reranker_score"] = float(score)
            final_docs.append(doc)
            if len(final_docs) >= top_k:
                break
                
    log.info(
        "Reranking complete: kept %d/%d documents (threshold=%.2f).", 
        len(final_docs), 
        len(documents), 
        relevance_threshold
    )
    return final_docs
