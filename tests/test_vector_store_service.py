"""
tests/test_vector_store_service.py
────────────────────────────────────
Unit tests for backend.services.vector_store_service.

Strategy:
  - Pinecone client is fully mocked — no real API calls.
  - Tests verify index creation logic, idempotency, and upsert delegation.

Run:
    pytest tests/test_vector_store_service.py -v
"""

from unittest.mock import MagicMock, call, patch

import pytest

from backend.services.vector_store_service import ensure_index, upsert_documents


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings():
    return MagicMock()


@pytest.fixture
def mock_pinecone(monkeypatch):
    """Patch the Pinecone class so no real HTTP calls are made."""
    with patch("backend.services.vector_store_service.Pinecone") as mock_pc_cls:
        mock_pc = MagicMock()
        mock_pc_cls.return_value = mock_pc
        yield mock_pc


# ── ensure_index tests ─────────────────────────────────────────────────────────

class TestEnsureIndex:

    def test_creates_index_when_absent(self, mock_pinecone):
        """If the index doesn't exist, create_index must be called."""
        mock_pinecone.list_indexes.return_value = []  # Empty — index absent

        result = ensure_index()

        mock_pinecone.create_index.assert_called_once()
        assert isinstance(result, str)

    def test_skips_creation_when_index_exists(self, mock_pinecone):
        """If the index already exists, create_index must NOT be called."""
        existing_idx = MagicMock()
        existing_idx.name = "rag-chatbot-free"   # matches settings default
        mock_pinecone.list_indexes.return_value = [existing_idx]

        ensure_index()

        mock_pinecone.create_index.assert_not_called()

    def test_returns_index_name_string(self, mock_pinecone):
        """ensure_index() always returns a non-empty string."""
        existing_idx = MagicMock()
        existing_idx.name = "rag-chatbot-free"
        mock_pinecone.list_indexes.return_value = [existing_idx]

        result = ensure_index()

        assert isinstance(result, str)
        assert len(result) > 0


# ── upsert_documents tests ─────────────────────────────────────────────────────

class TestUpsertDocuments:

    @patch("backend.services.vector_store_service.PineconeVectorStore")
    def test_from_documents_called_with_chunks(
        self, mock_vs_cls, mock_pinecone, mock_embeddings
    ):
        """upsert_documents() must call PineconeVectorStore.from_documents."""
        existing_idx = MagicMock()
        existing_idx.name = "rag-chatbot-free"
        mock_pinecone.list_indexes.return_value = [existing_idx]

        fake_chunks = [MagicMock(), MagicMock()]
        upsert_documents(fake_chunks, mock_embeddings)

        mock_vs_cls.from_documents.assert_called_once_with(
            fake_chunks,
            mock_embeddings,
            index_name="rag-chatbot-free",
        )

    @patch("backend.services.vector_store_service.PineconeVectorStore")
    def test_empty_chunks_still_calls_upsert(
        self, mock_vs_cls, mock_pinecone, mock_embeddings
    ):
        """Even an empty chunks list should still invoke from_documents."""
        existing_idx = MagicMock()
        existing_idx.name = "rag-chatbot-free"
        mock_pinecone.list_indexes.return_value = [existing_idx]

        upsert_documents([], mock_embeddings)

        mock_vs_cls.from_documents.assert_called_once()
