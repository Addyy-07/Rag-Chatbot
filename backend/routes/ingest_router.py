"""
backend/routes/ingest_router.py
────────────────────────────────
Orchestration layer for the PDF ingestion flow.

Responsibility:
  - Bridge between the Streamlit UI (file upload events) and the service layer.
  - Handle the Streamlit-specific UploadedFile → bytes → temp path conversion.
  - Generate a UUID per document and coordinate with DocumentRegistry.
  - Support single or multi-file uploads.

Changes in v2 (multi-doc):
  - ``handle_pdf_upload()`` now accepts a list of UploadedFile objects.
  - Returns a list of DocumentRecord objects (one per file).
  - Each file is processed independently with its own UUID and namespace.
  - Partial success: if one file fails, the rest are still processed.

Usage (from main.py)
-----
    from backend.routes.ingest_router import handle_pdf_upload

    records = handle_pdf_upload(uploaded_files, embeddings, registry)
"""

import uuid
from typing import Optional
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_huggingface import HuggingFaceEmbeddings

from backend.models.document import DocumentRecord
from backend.services.document_registry import DocumentRegistry
from backend.services.ingestion_service import ingest_pdf
from backend.utils.file_utils import temp_pdf_path
from backend.utils.logger import get_logger

log = get_logger(__name__)


def handle_pdf_upload(
    uploaded_files: list[UploadedFile],
    embeddings: HuggingFaceEmbeddings,
    registry: DocumentRegistry,
    token: str,
    owner_id: str | None = None,
) -> list[DocumentRecord]:
    """
    Orchestrate the full ingestion flow for one or more Streamlit UploadedFiles.

    For each file:
      1. Generate a unique document_id (UUID4).
      2. Read raw bytes and write to a secure temp file.
      3. Run the ingestion pipeline (load → split → embed → upsert).
      4. Register the returned DocumentRecord in the registry.

    Partial success: if one file fails, processing continues with remaining
    files. Failures are logged; the caller receives only successful records.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        embeddings:     Cached HuggingFaceEmbeddings singleton.
        registry:       DocumentRegistry instance for persistence.
        owner_id:       Optional identifier for the user or entity owning these files.

    Returns:
        List of successfully ingested DocumentRecord objects.
        May be shorter than ``uploaded_files`` if some files failed.
    """
    if not uploaded_files:
        log.warning("handle_pdf_upload called with empty file list.")
        return []

    log.info("Ingest router: processing %d file(s).", len(uploaded_files))
    successful: list[DocumentRecord] = []

    from backend.services.usage_api_client import check_upload_limit

    for uploaded_file in uploaded_files:
        # Check SaaS Upload Limit before processing each file
        success, error_msg = check_upload_limit(token)
        if not success:
            st.error(f"Cannot upload '{uploaded_file.name}': {error_msg}")
            log.warning("Upload limit exceeded. Skipped remaining files.")
            break

        doc_id = str(uuid.uuid4())
        log.info(
            "Processing '%s' → doc_id=%s", uploaded_file.name, doc_id
        )
        try:
            file_bytes = uploaded_file.read()
            with temp_pdf_path(file_bytes) as pdf_path:
                record = ingest_pdf(
                    pdf_path=pdf_path,
                    embeddings=embeddings,
                    document_id=doc_id,
                    filename=uploaded_file.name,
                    size_bytes=len(file_bytes),
                    owner_id=owner_id,
                )
            registry.add(record)
            successful.append(record)
            log.info(
                "Ingest router: '%s' done — %d pages, %d chunks, namespace='%s'.",
                record.filename,
                record.page_count,
                record.chunk_count,
                record.namespace,
            )
        except Exception as exc:
            log.error(
                "Ingest router: failed to process '%s': %s",
                uploaded_file.name,
                exc,
                exc_info=True,
            )
            # Continue processing remaining files

    log.info(
        "Ingest router: %d/%d file(s) ingested successfully.",
        len(successful),
        len(uploaded_files),
    )
    return successful
