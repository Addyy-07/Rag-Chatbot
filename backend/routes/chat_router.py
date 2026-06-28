"""
backend/routes/chat_router.py
──────────────────────────────
Orchestration layer for the chat Q&A flow.

Responsibility:
  - Bridge between the Streamlit UI (chat input + mode selection) and ChatService.
  - Validate inputs: non-empty question, at least one namespace selected.
  - Pass namespaces through to the service layer.
  - Return a plain string answer to the UI — no LangChain types leak up.

Changes in v2 (multi-doc):
  - ``handle_chat_query()`` now accepts ``namespaces: list[str]``.
  - Validates that at least one namespace is provided.
  - Raises ValueError with user-friendly messages on invalid input.

Chat modes (determined by the caller):
  📄 Single Doc  → namespaces=["<doc_uuid>"]
  📚 Selected    → namespaces=["<uuid_1>", "<uuid_2>", ...]
  🌐 All Docs    → namespaces=registry.get_all_namespaces()

Usage (from main.py)
-----
    from backend.routes.chat_router import handle_chat_query

    answer = handle_chat_query(
        question="What is RAG?",
        history=st.session_state.messages,
        embeddings=embeddings,
        namespaces=["abc-uuid-123"],
    )
"""

from langchain_huggingface import HuggingFaceEmbeddings

from backend.services.chat_service import ChatMessage, get_answer
from backend.utils.logger import get_logger

log = get_logger(__name__)


def handle_chat_query(
    question: str,
    history: list[ChatMessage],
    embeddings: HuggingFaceEmbeddings,
    namespaces: list[str] | None = None,
) -> str:
    """
    Orchestrate a single RAG Q&A turn across one or more document namespaces.

    Args:
        question:    The user's current question (raw string from chat input).
        history:     Full list of ChatMessage dicts from session state.
        embeddings:  Cached embeddings singleton from the UI layer.
        namespaces:  List of Pinecone namespace strings to search.
                     None = legacy single-namespace mode.

    Returns:
        Plain-text answer from the LLM.

    Raises:
        ValueError: If the question is empty or no namespaces are provided.
        Any exception from chat_service.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    target_namespaces = namespaces or [""]

    if not any(ns.strip() for ns in target_namespaces):
        raise ValueError(
            "No documents selected. Please select at least one document to chat with."
        )

    log.info(
        "Chat router: question=%d chars, namespaces=%s",
        len(question),
        target_namespaces,
    )
    answer = get_answer(question, history, embeddings, namespaces=target_namespaces)
    log.info("Chat router: answer returned (%d chars).", len(answer))
    return answer
