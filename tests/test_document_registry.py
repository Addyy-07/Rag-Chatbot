"""
tests/test_document_registry.py
────────────────────────────────
Unit tests for backend.services.document_registry.DocumentRegistry.

Strategy:
  - Each test uses a fresh temp file via tmp_path fixture (pytest built-in).
  - Pinecone is fully mocked — no real API calls.
  - Tests cover: add, get, get_all, delete, idempotent add, ordering.

Run:
    pytest tests/test_document_registry.py -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.models.document import DocumentRecord
from backend.services.document_registry import DocumentRegistry


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def registry(tmp_path) -> DocumentRegistry:
    """Return a DocumentRegistry backed by a temp file."""
    return DocumentRegistry(registry_path=str(tmp_path / "test_registry.json"))


@pytest.fixture
def doc_a() -> DocumentRecord:
    return DocumentRecord(
        document_id="aaaa-0001",
        filename="report_a.pdf",
        upload_date="2026-06-01T10:00:00+00:00",
        page_count=5,
        chunk_count=20,
        namespace="aaaa-0001",
        size_bytes=102400,
    )


@pytest.fixture
def doc_b() -> DocumentRecord:
    return DocumentRecord(
        document_id="bbbb-0002",
        filename="report_b.pdf",
        upload_date="2026-06-02T10:00:00+00:00",
        page_count=10,
        chunk_count=40,
        namespace="bbbb-0002",
        size_bytes=204800,
    )


# ── add() tests ────────────────────────────────────────────────────────────────

class TestAdd:

    def test_adds_document_to_empty_registry(self, registry, doc_a):
        registry.add(doc_a)
        assert registry.count() == 1

    def test_persists_to_json_file(self, registry, doc_a, tmp_path):
        registry.add(doc_a)
        raw = json.loads((tmp_path / "test_registry.json").read_text())
        assert len(raw) == 1
        assert raw[0]["document_id"] == "aaaa-0001"

    def test_idempotent_add_replaces_existing(self, registry, doc_a):
        """Adding the same document_id twice should result in only one record."""
        registry.add(doc_a)
        registry.add(doc_a)
        assert registry.count() == 1

    def test_adds_multiple_documents(self, registry, doc_a, doc_b):
        registry.add(doc_a)
        registry.add(doc_b)
        assert registry.count() == 2


# ── get() tests ────────────────────────────────────────────────────────────────

class TestGet:

    def test_returns_record_by_id(self, registry, doc_a):
        registry.add(doc_a)
        result = registry.get("aaaa-0001")
        assert result is not None
        assert result.filename == "report_a.pdf"

    def test_returns_none_for_missing_id(self, registry):
        result = registry.get("nonexistent-id")
        assert result is None

    def test_returns_correct_record_when_multiple_exist(self, registry, doc_a, doc_b):
        registry.add(doc_a)
        registry.add(doc_b)
        result = registry.get("bbbb-0002")
        assert result.filename == "report_b.pdf"


# ── get_all() tests ────────────────────────────────────────────────────────────

class TestGetAll:

    def test_returns_empty_list_for_empty_registry(self, registry):
        assert registry.get_all() == []

    def test_returns_all_documents(self, registry, doc_a, doc_b):
        registry.add(doc_a)
        registry.add(doc_b)
        results = registry.get_all()
        assert len(results) == 2

    def test_sorted_newest_first(self, registry, doc_a, doc_b):
        """doc_b has a later upload_date — should appear first."""
        registry.add(doc_a)
        registry.add(doc_b)
        results = registry.get_all()
        assert results[0].document_id == "bbbb-0002"

    def test_returns_document_record_instances(self, registry, doc_a):
        registry.add(doc_a)
        results = registry.get_all()
        assert all(isinstance(r, DocumentRecord) for r in results)


# ── delete() tests ─────────────────────────────────────────────────────────────

class TestDelete:

    def test_deletes_existing_document(self, registry, doc_a):
        registry.add(doc_a)
        result = registry.delete("aaaa-0001")
        assert result is True
        assert registry.count() == 0

    def test_returns_false_for_missing_document(self, registry):
        result = registry.delete("nonexistent-id")
        assert result is False

    def test_leaves_other_documents_intact(self, registry, doc_a, doc_b):
        registry.add(doc_a)
        registry.add(doc_b)
        registry.delete("aaaa-0001")
        remaining = registry.get_all()
        assert len(remaining) == 1
        assert remaining[0].document_id == "bbbb-0002"

    def test_no_pinecone_call_when_embeddings_is_none(self, registry, doc_a):
        """Passing embeddings=None should skip Pinecone purge."""
        registry.add(doc_a)
        with patch("backend.services.document_registry.Pinecone") as mock_pc:
            registry.delete("aaaa-0001", embeddings=None)
            mock_pc.assert_not_called()

    @patch("backend.services.document_registry.Pinecone")
    def test_purges_pinecone_namespace_when_embeddings_provided(
        self, mock_pc_cls, registry, doc_a
    ):
        """Passing embeddings should trigger Pinecone namespace purge."""
        mock_pc = MagicMock()
        mock_pc_cls.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index

        registry.add(doc_a)
        registry.delete("aaaa-0001", embeddings=MagicMock())

        mock_index.delete.assert_called_once_with(
            delete_all=True, namespace="aaaa-0001"
        )


# ── get_all_namespaces() tests ─────────────────────────────────────────────────

class TestGetAllNamespaces:

    def test_returns_namespace_strings(self, registry, doc_a, doc_b):
        registry.add(doc_a)
        registry.add(doc_b)
        namespaces = registry.get_all_namespaces()
        assert set(namespaces) == {"aaaa-0001", "bbbb-0002"}

    def test_empty_registry_returns_empty_list(self, registry):
        assert registry.get_all_namespaces() == []
