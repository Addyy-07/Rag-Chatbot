"""
backend/routes/ingest_router.py
────────────────────────────────
Orchestration layer for the PDF ingestion flow.

Responsibility:
  - Bridge between the Streamlit UI event (file upload) and the service layer.
  - Handle the Streamlit-specific UploadedFile → bytes → temp path conversion.
  - Coordinate EmbeddingService + IngestionService without owning any logic.

Why a separate router?
  - The router is the ONLY layer that knows about both Streamlit types and
    service types. Services remain UI-agnostic.
  - When we add a FastAPI backend, we replace/add routers — services don't change.
  - Thin: no business logic lives here.

Usage (from main.py)
-----
    from backend.routes.ingest_router import handle_pdf_upload

    result = handle_pdf_upload(uploaded_file, embeddings)
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from backend.services.ingestion_service import IngestionResult, ingest_pdf
from backend.utils.file_utils import temp_pdf_path
from backend.utils.logger import get_logger

log = get_logger(__name__)


def handle_pdf_upload(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    embeddings: HuggingFaceEmbeddings,
) -> IngestionResult:
    """
    Orchestrate the full ingestion flow for a Streamlit UploadedFile.

    Steps:
      1. Read raw bytes from the uploaded file object.
      2. Write bytes to a secure temporary file (auto-cleaned on exit).
      3. Delegate the actual ingestion to ``ingest_pdf()``.

    Args:
        uploaded_file: The Streamlit UploadedFile object from st.file_uploader.
        embeddings:    Cached HuggingFaceEmbeddings singleton from the UI layer.

    Returns:
        IngestionResult with page_count and chunk_count.

    Raises:
        Any exception from ingestion_service or vector_store_service.
    """
    log.info("Ingest router: handling upload of '%s'.", uploaded_file.name)
    file_bytes = uploaded_file.read()

    with temp_pdf_path(file_bytes) as pdf_path:
        result = ingest_pdf(pdf_path, embeddings)

    log.info(
        "Ingest router: '%s' ingested — %d pages, %d chunks.",
        uploaded_file.name,
        result.page_count,
        result.chunk_count,
    )
    return result
