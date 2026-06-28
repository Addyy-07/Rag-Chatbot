"""
backend/services/chat_service.py
──────────────────────────────────
Single Responsibility: execute the RAG question-answering pipeline.

Pipeline steps:
  1. Retrieve the top-k most relevant document chunks from Pinecone.
  2. Truncate each chunk to avoid exceeding context limits.
  3. Format recent chat history into a conversation string.
  4. Invoke the LangChain chain: prompt → LLM → string output.

Design decisions:
  - Chat history and context truncation limits come from Settings.
  - The prompt template is imported from the prompts module — not defined here.
  - The LLM (ChatGroq) is instantiated inside the function so it is
    always fresh (stateless), which is safe and idiomatic for serverless/Streamlit.
  - Returns a plain str — no LangChain types leak into the router or UI.
  - `ChatMessage` is a simple TypedDict so routers can type-annotate history.

Usage
-----
    from backend.services.chat_service import get_answer, ChatMessage

    history: list[ChatMessage] = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for..."},
    ]
    answer = get_answer("Can you give an example?", history, embeddings)
"""

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config.settings import settings
from backend.prompts.rag_prompt import RAG_PROMPT
from backend.services.vector_store_service import get_retriever
from backend.utils.logger import get_logger

log = get_logger(__name__)


class ChatMessage(TypedDict):
    """A single message in the conversation history."""

    role: str     # "user" | "assistant"
    content: str


def _format_history(history: list[ChatMessage]) -> str:
    """
    Format the last N conversation turns into a plain-text string.

    Only the most recent ``settings.chat_history_window`` messages are
    included to keep token usage bounded. Each message is truncated to
    ``settings.chat_history_max_chars`` characters.

    Args:
        history: Full list of chat messages (oldest first).

    Returns:
        Formatted string like "Human: ...\nAssistant: ..."
    """
    recent = history[-settings.chat_history_window :]
    lines = []
    for msg in recent:
        role_label = "Human" if msg["role"] == "user" else "Assistant"
        truncated = msg["content"][: settings.chat_history_max_chars]
        lines.append(f"{role_label}: {truncated}")
    return "\n".join(lines)


def _build_context(docs: list) -> str:
    """
    Concatenate retrieved document chunks into a single context string.

    Each chunk is truncated to ``settings.retriever_context_max_chars`` to
    prevent the LLM context window from being exhausted by a single chunk.

    Args:
        docs: List of LangChain Document objects from the retriever.

    Returns:
        Newline-separated string of truncated chunk contents.
    """
    return "\n\n".join(
        doc.page_content[: settings.retriever_context_max_chars] for doc in docs
    )


def get_answer(
    question: str,
    history: list[ChatMessage],
    embeddings: HuggingFaceEmbeddings,
) -> str:
    """
    Run the full RAG Q&A pipeline and return the LLM's answer.

    Steps:
      1. Build a retriever from the vector store.
      2. Retrieve relevant document chunks for the question.
      3. Format history and context.
      4. Invoke the RAG chain (prompt → LLM → string output).

    Args:
        question:   The user's current question.
        history:    Full list of prior ChatMessage dicts.
        embeddings: Instantiated embeddings model (shared singleton).

    Returns:
        Plain-text answer string from the LLM.

    Raises:
        Any exception from Pinecone retrieval or Groq API calls.
    """
    log.info("RAG pipeline started. Question: '%s...'", question[:60])

    # Step 1 — Retrieve relevant chunks
    retriever = get_retriever(embeddings)
    docs = retriever.invoke(question)
    log.debug("Retrieved %d document chunk(s).", len(docs))

    # Step 2 — Build context and history strings
    context = _build_context(docs)
    history_text = _format_history(history)

    # Step 3 — Instantiate LLM (stateless, fresh per call)
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
    )

    # Step 4 — Invoke chain: prompt | llm | output parser
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer: str = chain.invoke(
        {
            "history": history_text,
            "context": context,
            "question": question,
        }
    )

    log.info("RAG pipeline complete. Answer length: %d chars.", len(answer))
    return answer
