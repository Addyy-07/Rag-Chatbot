"""
backend/services/ingestion_service.py
───────────────────────────────────────
Single Responsibility: orchestrate the PDF → chunks → vector store pipeline.

Pipeline steps:
  1. Load the PDF from a file path using PyPDFLoader.
  2. Split documents into overlapping chunks via RecursiveCharacterTextSplitter.
  3. Upsert chunks into Pinecone via VectorStoreService.

Design decisions:
  - Accepts a file path (str), NOT an UploadedFile object — keeps this service
    completely decoupled from Streamlit. The router converts the uploaded file
    to a temp path before calling here.
  - Chunk size and overlap come from Settings (not hardcoded).
  - Returns a typed dataclass so callers don't need to unpack bare tuples.
  - Logging replaces all print() statements.

Usage
-----
    from backend.services.ingestion_service import ingest_pdf, IngestionResult

    result: IngestionResult = ingest_pdf(pdf_path, embeddings)
    print(f"Ingested {result.chunk_count} chunks from {result.page_count} pages.")
"""

from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config.settings import settings
from backend.services.vector_store_service import upsert_documents
from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """
    Immutable result returned by ``ingest_pdf()``.

    Attributes:
        page_count:  Number of pages in the original PDF.
        chunk_count: Number of chunks stored in the vector database.
    """

    page_count: int
    chunk_count: int


def ingest_pdf(pdf_path: str, embeddings: HuggingFaceEmbeddings) -> IngestionResult:
    """
    Run the full PDF ingestion pipeline.

    Steps:
      1. Load all pages from the PDF at ``pdf_path``.
      2. Split pages into overlapping text chunks.
      3. Embed chunks and upsert to Pinecone.

    Args:
        pdf_path:   Absolute path to the PDF file on disk.
        embeddings: An instantiated LangChain-compatible embeddings model.

    Returns:
        IngestionResult containing page and chunk counts.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        Any exception propagated from PyPDFLoader or PineconeVectorStore.
    """
    # Step 1 — Load PDF
    log.info("Loading PDF: %s", pdf_path)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    log.info("Loaded %d page(s) from PDF.", len(documents))

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

    # Step 3 — Upsert to vector store
    upsert_documents(chunks, embeddings)

    result = IngestionResult(page_count=len(documents), chunk_count=len(chunks))
    log.info(
        "Ingestion complete: %d pages, %d chunks.",
        result.page_count,
        result.chunk_count,
    )
    return result
