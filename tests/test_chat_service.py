"""
tests/test_chat_service.py
───────────────────────────
Unit tests for backend.services.chat_service.

Strategy:
  - The LLM (ChatGroq), retriever, and chain are fully mocked.
  - Tests verify: correct chain invocation, history formatting, and
    context truncation behaviour.

Run:
    pytest tests/test_chat_service.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.services.chat_service import (
    ChatMessage,
    _build_context,
    _format_history,
    get_answer,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embeddings():
    return MagicMock()


@pytest.fixture
def sample_history() -> list[ChatMessage]:
    return [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
        {"role": "user", "content": "Can you give an example?"},
    ]


@pytest.fixture
def sample_docs():
    """Return mock Document objects with controllable page_content."""
    doc1 = MagicMock()
    doc1.page_content = "A" * 600   # Longer than retriever_context_max_chars (500)
    doc2 = MagicMock()
    doc2.page_content = "B" * 300   # Shorter than limit
    return [doc1, doc2]


# ── _format_history tests ──────────────────────────────────────────────────────

class TestFormatHistory:

    def test_returns_string(self, sample_history):
        result = _format_history(sample_history)
        assert isinstance(result, str)

    def test_labels_user_as_human(self, sample_history):
        result = _format_history(sample_history)
        assert "Human:" in result

    def test_labels_assistant_correctly(self):
        history = [{"role": "assistant", "content": "Hello there!"}]
        result = _format_history(history)
        assert "Assistant:" in result

    def test_empty_history_returns_empty_string(self):
        result = _format_history([])
        assert result == ""

    def test_content_is_truncated_to_max_chars(self):
        long_content = "X" * 1000
        history = [{"role": "user", "content": long_content}]
        result = _format_history(history)
        # The truncated content should appear, not the full 1000 chars
        assert "X" * 200 in result
        assert "X" * 201 not in result or len(result) < 1000


# ── _build_context tests ───────────────────────────────────────────────────────

class TestBuildContext:

    def test_returns_string(self, sample_docs):
        result = _build_context(sample_docs)
        assert isinstance(result, str)

    def test_long_chunk_is_truncated(self, sample_docs):
        result = _build_context(sample_docs)
        # First doc has 600 chars but limit is 500; check truncation occurred
        lines = result.split("\n\n")
        assert len(lines[0]) <= 500

    def test_empty_docs_returns_empty_string(self):
        result = _build_context([])
        assert result == ""

    def test_chunks_separated_by_double_newline(self, sample_docs):
        result = _build_context(sample_docs)
        assert "\n\n" in result


# ── get_answer integration test ────────────────────────────────────────────────

class TestGetAnswer:

    @patch("backend.services.chat_service.StrOutputParser")
    @patch("backend.services.chat_service.ChatGroq")
    @patch("backend.services.chat_service.get_retriever")
    def test_returns_string_answer(
        self,
        mock_get_retriever,
        mock_groq_cls,
        mock_parser_cls,
        sample_history,
        mock_embeddings,
    ):
        """get_answer() must return the plain string produced by the chain."""
        # Set up mock retriever returning one doc
        mock_doc = MagicMock()
        mock_doc.page_content = "The capital of France is Paris."
        mock_get_retriever.return_value.invoke.return_value = [mock_doc]

        # Mock the chain pipeline to return a fixed answer
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Paris is the capital of France."

        # Patch the | operator chain: RAG_PROMPT | llm | parser → mock_chain
        with patch("backend.services.chat_service.RAG_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            answer = get_answer("What is the capital?", sample_history, mock_embeddings)

        assert isinstance(answer, str)

    @patch("backend.services.chat_service.get_retriever")
    def test_retriever_called_with_question(
        self,
        mock_get_retriever,
        sample_history,
        mock_embeddings,
    ):
        """The retriever must be invoked with the user's question."""
        mock_doc = MagicMock()
        mock_doc.page_content = "Some content."
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_get_retriever.return_value = mock_retriever

        with patch("backend.services.chat_service.ChatGroq"), \
             patch("backend.services.chat_service.RAG_PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "answer"
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_chain.__or__ = MagicMock(return_value=mock_chain)

            get_answer("Capital of France?", sample_history, mock_embeddings)

        mock_retriever.invoke.assert_called_once_with("Capital of France?")
