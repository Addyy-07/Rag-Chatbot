"""
tests/test_multi_retriever_service.py
──────────────────────────────────────
Unit tests for backend.services.multi_retriever_service.

Strategy:
  - Pinecone and VectorStore are fully mocked.
  - Tests cover: single namespace, multiple namespaces, deduplication,
    graceful degradation on namespace failure, empty namespace list.

Run:
    pytest tests/test_multi_retriever_service.py -v
"""

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document

from backend.services.multi_retriever_service import (
    _content_hash,
    retrieve_from_namespaces,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings():
    return MagicMock()


def _make_doc(content: str, metadata: dict | None = None) -> Document:
    """Helper: build a LangChain Document with given content."""
    return Document(page_content=content, metadata=metadata or {})


# ── _content_hash tests ────────────────────────────────────────────────────────

class TestContentHash:

    def test_same_content_produces_same_hash(self):
        doc1 = _make_doc("Hello world")
        doc2 = _make_doc("Hello world")
        assert _content_hash(doc1) == _content_hash(doc2)

    def test_different_content_produces_different_hash(self):
        doc1 = _make_doc("Hello world")
        doc2 = _make_doc("Goodbye world")
        assert _content_hash(doc1) != _content_hash(doc2)

    def test_returns_string(self):
        doc = _make_doc("test")
        assert isinstance(_content_hash(doc), str)

    def test_hash_length_is_64_chars(self):
        """SHA-256 hex digest is always 64 characters."""
        doc = _make_doc("content")
        assert len(_content_hash(doc)) == 64


# ── retrieve_from_namespaces tests ─────────────────────────────────────────────

class TestRetrieveFromNamespaces:

    @patch("backend.services.multi_retriever_service.PineconeVectorStore")
    @patch("backend.services.multi_retriever_service.ensure_index")
    def test_empty_namespace_list_returns_empty(
        self, mock_ensure, mock_vs, mock_embeddings
    ):
        """Empty namespace list → empty result with no Pinecone calls."""
        result = retrieve_from_namespaces("question", [], mock_embeddings)
        assert result == []
        mock_vs.assert_not_called()

    @patch("backend.services.multi_retriever_service.PineconeVectorStore")
    @patch("backend.services.multi_retriever_service.ensure_index")
    def test_single_namespace_returns_docs(
        self, mock_ensure, mock_vs_cls, mock_embeddings
    ):
        """Single namespace returns the docs from that namespace."""
        mock_ensure.return_value = "my-index"
        doc = _make_doc("Chunk from namespace A")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_vs_instance = MagicMock()
        mock_vs_instance.as_retriever.return_value = mock_retriever
        mock_vs_cls.return_value = mock_vs_instance

        result = retrieve_from_namespaces("test question", ["ns-a"], mock_embeddings)

        assert len(result) == 1
        assert result[0].page_content == "Chunk from namespace A"

    @patch("backend.services.multi_retriever_service.PineconeVectorStore")
    @patch("backend.services.multi_retriever_service.ensure_index")
    def test_multiple_namespaces_merges_results(
        self, mock_ensure, mock_vs_cls, mock_embeddings
    ):
        """Multiple namespaces: results from all are merged."""
        mock_ensure.return_value = "my-index"

        doc_a = _make_doc("Chunk from namespace A")
        doc_b = _make_doc("Chunk from namespace B")

        call_count = {"n": 0}

        def vs_side_effect(**kwargs):
            instance = MagicMock()
            retriever = MagicMock()
            # Alternate between returning doc_a and doc_b
            retriever.invoke.return_value = (
                [doc_a] if call_count["n"] == 0 else [doc_b]
            )
            call_count["n"] += 1
            instance.as_retriever.return_value = retriever
            return instance

        mock_vs_cls.side_effect = vs_side_effect

        result = retrieve_from_namespaces(
            "test question", ["ns-a", "ns-b"], mock_embeddings
        )

        assert len(result) == 2

    @patch("backend.services.multi_retriever_service.PineconeVectorStore")
    @patch("backend.services.multi_retriever_service.ensure_index")
    def test_duplicate_content_is_deduplicated(
        self, mock_ensure, mock_vs_cls, mock_embeddings
    ):
        """Same chunk returned from two namespaces should appear only once."""
        mock_ensure.return_value = "my-index"
        same_doc = _make_doc("Identical content in both namespaces")

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [same_doc]
        mock_vs_instance = MagicMock()
        mock_vs_instance.as_retriever.return_value = mock_retriever
        mock_vs_cls.return_value = mock_vs_instance

        result = retrieve_from_namespaces(
            "question", ["ns-a", "ns-b"], mock_embeddings
        )

        assert len(result) == 1

    @patch("backend.services.multi_retriever_service.PineconeVectorStore")
    @patch("backend.services.multi_retriever_service.ensure_index")
    def test_graceful_degradation_on_namespace_failure(
        self, mock_ensure, mock_vs_cls, mock_embeddings
    ):
        """If one namespace fails, results from healthy namespaces are returned."""
        mock_ensure.return_value = "my-index"
        doc_b = _make_doc("Chunk from healthy namespace B")

        call_count = {"n": 0}

        def vs_side_effect(**kwargs):
            instance = MagicMock()
            retriever = MagicMock()
            if call_count["n"] == 0:
                # First namespace: raise an error
                retriever.invoke.side_effect = RuntimeError("Pinecone timeout")
            else:
                retriever.invoke.return_value = [doc_b]
            call_count["n"] += 1
            instance.as_retriever.return_value = retriever
            return instance

        mock_vs_cls.side_effect = vs_side_effect

        # Should NOT raise, should return the doc from the healthy namespace
        result = retrieve_from_namespaces(
            "question", ["ns-fail", "ns-ok"], mock_embeddings
        )

        assert len(result) == 1
        assert result[0].page_content == "Chunk from healthy namespace B"

    @patch("backend.services.multi_retriever_service.PineconeVectorStore")
    @patch("backend.services.multi_retriever_service.ensure_index")
    def test_source_namespace_annotated_in_metadata(
        self, mock_ensure, mock_vs_cls, mock_embeddings
    ):
        """Each returned doc should have 'source_namespace' in its metadata."""
        mock_ensure.return_value = "my-index"
        doc = _make_doc("Some content")

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]
        mock_vs_instance = MagicMock()
        mock_vs_instance.as_retriever.return_value = mock_retriever
        mock_vs_cls.return_value = mock_vs_instance

        result = retrieve_from_namespaces("q", ["namespace-x"], mock_embeddings)

        assert result[0].metadata.get("source_namespace") == "namespace-x"
