"""
backend/routes/chat_router.py
──────────────────────────────
Orchestration layer for the chat Q&A flow.

Responsibility:
  - Bridge between the Streamlit UI (chat input) and the ChatService.
  - Validate inputs and pass them to the service layer.
  - Return a plain string answer to the UI — no LangChain types leak up.

Why a separate router?
  - Services are tested independently; routers handle the wiring.
  - Swapping to FastAPI: add a new router, keep the same ChatService.
  - If you later add rate limiting, caching, or auth middleware — it goes here.

Usage (from main.py)
-----
    from backend.routes.chat_router import handle_chat_query

    answer = handle_chat_query(question, st.session_state.messages, embeddings)
"""

from langchain_huggingface import HuggingFaceEmbeddings

from backend.services.chat_service import ChatMessage, get_answer
from backend.utils.logger import get_logger

log = get_logger(__name__)


def handle_chat_query(
    question: str,
    history: list[ChatMessage],
    embeddings: HuggingFaceEmbeddings,
) -> str:
    """
    Orchestrate a single RAG Q&A turn.

    Validates the question is non-empty, then delegates to the ChatService.

    Args:
        question:   The user's current question (raw string from chat input).
        history:    Full list of ChatMessage dicts from session state.
        embeddings: Cached embeddings singleton from the UI layer.

    Returns:
        Plain-text answer from the LLM.

    Raises:
        ValueError: If the question is empty or whitespace-only.
        Any exception from chat_service.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    log.info("Chat router: processing question (%d chars).", len(question))
    answer = get_answer(question, history, embeddings)
    log.info("Chat router: answer returned (%d chars).", len(answer))
    return answer
