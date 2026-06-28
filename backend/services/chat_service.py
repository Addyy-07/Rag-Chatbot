"""
backend/services/chat_service.py
──────────────────────────────────
Single Responsibility: execute the RAG question-answering pipeline.

Pipeline steps:
  1. Retrieve relevant document chunks from one or more Pinecone namespaces.
  2. Truncate each chunk to avoid exceeding context limits.
  3. Format recent chat history into a conversation string.
  4. Invoke the LangChain chain: prompt → LLM → string output.
  5. Extract source citations from retrieved chunk metadata.
  6. Return a ChatResult containing both the answer and deduplicated citations.

Changes in v3 (source citations):
  - ``get_answer()`` now returns ``ChatResult`` instead of ``str``.
  - Citations are built from retrieved Document metadata (filename, page, doc_id).
  - Deduplication: one citation per (filename, page_number) pair, by relevance order.
  - Backward compatible: ``result.answer`` gives the plain string.

Usage
-----
    from backend.services.chat_service import get_answer, ChatMessage
    from backend.models.chat import ChatResult

    result: ChatResult = get_answer("What is RAG?", history, embeddings,
                                    namespaces=["abc-uuid"])
    print(result.answer)
    for c in result.citations:
        print(c.display_label)   # "report.pdf — Page 4"
        print(c.preview)         # truncated chunk text
"""

from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config.settings import settings
from backend.models.chat import ChatResult, SourceCitation
from backend.prompts.rag_prompt import RAG_PROMPT
from backend.services.multi_retriever_service import retrieve_from_namespaces
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
        Formatted string like "Human: ...\\nAssistant: ..."
    """
    recent = history[-settings.chat_history_window:]
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
        docs: List of LangChain Document objects from the retriever(s).

    Returns:
        Newline-separated string of truncated chunk contents.
    """
    return "\n\n".join(
        doc.page_content[: settings.retriever_context_max_chars] for doc in docs
    )


def _extract_citations(docs: list) -> list[SourceCitation]:
    """
    Build a deduplicated list of SourceCitation from retrieved document chunks.

    Each doc is expected to carry metadata stamped during ingestion:
      - ``filename``    : original PDF filename (e.g. "report.pdf")
      - ``document_id`` : UUID of the parent document (= Pinecone namespace)
      - ``page``        : 0-indexed page number (from PyPDFLoader)

    Deduplication: only the first (highest-relevance) chunk per
    (filename, page_number) pair is kept. If metadata is missing, graceful
    fallbacks are used so citations never hard-fail.

    Args:
        docs: Ordered list of retrieved LangChain Document objects.

    Returns:
        Deduplicated list of SourceCitation, ordered by retrieval rank.
    """
    seen: set[tuple[str, int]] = set()
    citations: list[SourceCitation] = []

    for doc in docs:
        meta = doc.metadata or {}
        filename = meta.get("filename") or meta.get("source", "Unknown source")
        # Strip any temp-file path prefix if filename wasn't stamped
        if "/" in filename or "\\" in filename:
            filename = filename.split("/")[-1].split("\\")[-1]

        page_0indexed = meta.get("page", 0)
        page_number = int(page_0indexed) + 1  # Convert to 1-indexed

        key = (filename, page_number)
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            SourceCitation(
                filename=filename,
                page_number=page_number,
                chunk_text=doc.page_content,
                document_id=meta.get("document_id", meta.get("source_namespace", "")),
            )
        )

    log.debug("Extracted %d unique citation(s).", len(citations))
    return citations


def get_answer(
    question: str,
    history: list[ChatMessage],
    embeddings: HuggingFaceEmbeddings,
    namespaces: list[str] | None = None,
) -> ChatResult:
    """
    Run the full multi-namespace RAG Q&A pipeline and return a ChatResult.

    Steps:
      1. Retrieve relevant chunks from all specified namespaces.
      2. Build merged context and format history.
      3. Invoke the RAG chain (prompt → LLM → string output).
      4. Extract source citations from retrieved chunk metadata.
      5. Return ChatResult(answer, citations).

    Args:
        question:    The user's current question.
        history:     Full list of prior ChatMessage dicts.
        embeddings:  Instantiated embeddings model (shared singleton).
        namespaces:  List of Pinecone namespace strings to search.
                     Pass None or [""] to use the default namespace (legacy mode).

    Returns:
        ChatResult with `answer` (str) and `citations` (list[SourceCitation]).

    Raises:
        Any exception from Pinecone retrieval or Groq API calls.
    """
    # Normalise: None → default namespace for backward compat
    target_namespaces: list[str] = namespaces if namespaces else [""]

    log.info(
        "RAG pipeline: question='%s...', namespaces=%s",
        question[:60],
        target_namespaces,
    )

    # Step 1 — Retrieve relevant chunks across namespaces
    docs = retrieve_from_namespaces(question, target_namespaces, embeddings)
    log.debug("Total retrieved: %d chunk(s).", len(docs))

    # Step 2 — Build context and history
    context = _build_context(docs)
    history_text = _format_history(history)

    if not context.strip():
        log.warning("No relevant context found in any namespace.")
        context = "No relevant document content was found for this question."

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

    # Step 5 — Extract source citations from retrieved chunk metadata
    citations = _extract_citations(docs)

    log.info(
        "RAG pipeline complete. Answer: %d chars, %d citation(s).",
        len(answer),
        len(citations),
    )
    return ChatResult(answer=answer, citations=citations)
