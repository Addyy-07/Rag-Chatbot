"""
tests/test_ingestion_service.py
────────────────────────────────
Unit tests for backend.services.ingestion_service.

Strategy:
  - All external dependencies (PyPDFLoader, text splitter, upsert) are mocked
    so tests are fast, offline, and deterministic.
  - Tests cover: happy path, empty document list, and correct settings usage.

Run:
    pytest tests/test_ingestion_service.py -v
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from backend.services.ingestion_service import IngestionResult, ingest_pdf


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings():
    """Return a MagicMock standing in for HuggingFaceEmbeddings."""
    return MagicMock()


@pytest.fixture
def fake_documents():
    """Return a list of mock LangChain Document objects."""
    doc = MagicMock()
    doc.page_content = "This is page content for testing."
    return [doc, doc]  # 2 pages


@pytest.fixture
def fake_chunks(fake_documents):
    """Return a list of mock chunk Document objects (3 chunks from 2 pages)."""
    chunk = MagicMock()
    chunk.page_content = "This is a chunk."
    return [chunk, chunk, chunk]  # 3 chunks


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestIngestPdf:

    @patch("backend.services.ingestion_service.upsert_documents")
    @patch("backend.services.ingestion_service.RecursiveCharacterTextSplitter")
    @patch("backend.services.ingestion_service.PyPDFLoader")
    def test_happy_path_returns_correct_result(
        self,
        mock_loader_cls,
        mock_splitter_cls,
        mock_upsert,
        fake_documents,
        fake_chunks,
        mock_embeddings,
    ):
        """ingest_pdf() should return an IngestionResult with correct counts."""
        mock_loader_cls.return_value.load.return_value = fake_documents
        mock_splitter_cls.return_value.split_documents.return_value = fake_chunks

        result = ingest_pdf("/fake/path.pdf", mock_embeddings)

        assert isinstance(result, IngestionResult)
        assert result.page_count == 2
        assert result.chunk_count == 3

    @patch("backend.services.ingestion_service.upsert_documents")
    @patch("backend.services.ingestion_service.RecursiveCharacterTextSplitter")
    @patch("backend.services.ingestion_service.PyPDFLoader")
    def test_upsert_is_called_once(
        self,
        mock_loader_cls,
        mock_splitter_cls,
        mock_upsert,
        fake_documents,
        fake_chunks,
        mock_embeddings,
    ):
        """upsert_documents() must be called exactly once per ingest_pdf() call."""
        mock_loader_cls.return_value.load.return_value = fake_documents
        mock_splitter_cls.return_value.split_documents.return_value = fake_chunks

        ingest_pdf("/fake/path.pdf", mock_embeddings)

        mock_upsert.assert_called_once_with(fake_chunks, mock_embeddings)

    @patch("backend.services.ingestion_service.upsert_documents")
    @patch("backend.services.ingestion_service.RecursiveCharacterTextSplitter")
    @patch("backend.services.ingestion_service.PyPDFLoader")
    def test_empty_pdf_returns_zero_counts(
        self,
        mock_loader_cls,
        mock_splitter_cls,
        mock_upsert,
        mock_embeddings,
    ):
        """An empty PDF (no pages, no chunks) returns counts of 0."""
        mock_loader_cls.return_value.load.return_value = []
        mock_splitter_cls.return_value.split_documents.return_value = []

        result = ingest_pdf("/fake/empty.pdf", mock_embeddings)

        assert result.page_count == 0
        assert result.chunk_count == 0

    @patch("backend.services.ingestion_service.upsert_documents")
    @patch("backend.services.ingestion_service.RecursiveCharacterTextSplitter")
    @patch("backend.services.ingestion_service.PyPDFLoader")
    def test_result_is_immutable(
        self,
        mock_loader_cls,
        mock_splitter_cls,
        mock_upsert,
        fake_documents,
        fake_chunks,
        mock_embeddings,
    ):
        """IngestionResult is a frozen dataclass — mutation must raise TypeError."""
        mock_loader_cls.return_value.load.return_value = fake_documents
        mock_splitter_cls.return_value.split_documents.return_value = fake_chunks

        result = ingest_pdf("/fake/path.pdf", mock_embeddings)

        with pytest.raises((AttributeError, TypeError)):
            result.page_count = 999  # type: ignore[misc]
