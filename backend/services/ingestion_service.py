"""
backend/services/ingestion_service.py
───────────────────────────────────────
Single Responsibility: orchestrate the PDF → chunks → vector store pipeline.

Pipeline steps:
  1. Load the PDF from a file path using PyPDFLoader.
  2. Split documents into overlapping chunks via RecursiveCharacterTextSplitter.
  3. Upsert chunks into a specific Pinecone namespace.
  4. Return a DocumentRecord with full metadata.

Changes in v2 (multi-doc):
  - Accepts ``namespace``, ``filename``, and ``size_bytes`` parameters.
  - Returns ``DocumentRecord`` instead of ``IngestionResult``.
  - ``IngestionResult`` is kept as an alias for backward compatibility.

Usage
-----
    from backend.services.ingestion_service import ingest_pdf
    from backend.models.document import DocumentRecord

    record: DocumentRecord = ingest_pdf(
        pdf_path="/tmp/report.pdf",
        embeddings=embeddings,
        document_id="abc-uuid-123",
        filename="report.pdf",
        size_bytes=204800,
    )
"""

import uuid
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config.settings import settings
from backend.models.document import DocumentRecord, make_upload_date
from backend.services.vector_store_service import upsert_documents
from backend.utils.logger import get_logger

log = get_logger(__name__)


# ── Backward-compat alias ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class IngestionResult:
    """
    Legacy result type — kept for backward compatibility with existing tests.
    New code should use DocumentRecord directly.
    """
    page_count: int
    chunk_count: int


def ingest_pdf(
    pdf_path: str,
    embeddings: HuggingFaceEmbeddings,
    document_id: str | None = None,
    filename: str = "document.pdf",
    size_bytes: int = 0,
) -> DocumentRecord:
    """
    Run the full PDF ingestion pipeline for a single document.

    Steps:
      1. Load all pages from the PDF at ``pdf_path``.
      2. Split pages into overlapping text chunks.
      3. Embed chunks and upsert to the document's Pinecone namespace.
      4. Return a populated DocumentRecord.

    Args:
        pdf_path:     Absolute path to the PDF file on disk.
        embeddings:   An instantiated LangChain-compatible embeddings model.
        document_id:  UUID string for this document. Generated if not provided.
        filename:     Original filename (used for display and metadata).
        size_bytes:   File size in bytes (used for display metadata).

    Returns:
        DocumentRecord with all metadata populated.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        Any exception propagated from PyPDFLoader or PineconeVectorStore.
    """
    doc_id = document_id or str(uuid.uuid4())
    namespace = doc_id  # Namespace == document_id

    # Step 1 — Load PDF
    log.info("Loading PDF: %s (doc_id=%s)", pdf_path, doc_id)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    log.info("Loaded %d page(s).", len(documents))

    # Step 2 — Split into chunks
    log.info(
        "Splitting into chunks (size=%d, overlap=%d)...",
        settings.chunk_size,
        settings.chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    log.info("Created %d chunk(s).", len(chunks))

    # Step 2b — Stamp source metadata onto every chunk for citation extraction.
    # PyPDFLoader already adds metadata["source"] (temp path) and metadata["page"]
    # (0-indexed). We overwrite "source" with the clean filename and add document_id
    # so retrieved chunks carry full citation info without a registry lookup.
    for chunk in chunks:
        chunk.metadata["filename"] = filename
        chunk.metadata["document_id"] = doc_id

    # Step 3 — Upsert to namespace-scoped vector store
    upsert_documents(chunks, embeddings, namespace=namespace)

    # Step 4 — Build and return DocumentRecord
    record = DocumentRecord(
        document_id=doc_id,
        filename=filename,
        upload_date=make_upload_date(),
        page_count=len(documents),
        chunk_count=len(chunks),
        namespace=namespace,
        size_bytes=size_bytes,
    )
    log.info(
        "Ingestion complete: doc_id=%s, %d pages, %d chunks, namespace='%s'.",
        doc_id,
        record.page_count,
        record.chunk_count,
        namespace,
    )
    return record
