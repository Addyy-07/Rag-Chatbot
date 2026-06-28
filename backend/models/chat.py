"""
backend/models/chat.py
───────────────────────
Data models for the chat Q&A pipeline output.

SourceCitation  — metadata for a single retrieved document chunk.
ChatResult      — the full response from get_answer(): answer + citation list.

Why separate from document.py?
  - document.py owns ingestion-time metadata (DocumentRecord).
  - chat.py owns query-time output (ChatResult, SourceCitation).
  - Keeps each model focused and the imports easy to follow.

Usage
-----
    from backend.models.chat import ChatResult, SourceCitation

    result: ChatResult = get_answer(...)
    print(result.answer)
    for cite in result.citations:
        print(f"{cite.filename} p.{cite.page_number}: {cite.preview}")
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceCitation:
    """
    Metadata for a single retrieved document chunk used as evidence.

    Attributes:
        filename:     Original PDF filename (e.g. "Operating_System.pdf").
        page_number:  1-indexed page number within the PDF.
        chunk_text:   Full text of the retrieved chunk (for expandable view).
        document_id:  Pinecone namespace / document UUID.
    """

    filename: str
    page_number: int     # 1-indexed for display
    chunk_text: str      # Full chunk text
    document_id: str     # source namespace

    @property
    def preview(self) -> str:
        """Return a short preview (first 200 chars) of the chunk text."""
        text = self.chunk_text.strip()
        if len(text) <= 200:
            return text
        return text[:197] + "..."

    @property
    def display_label(self) -> str:
        """Return a display label like 'Operating_System.pdf — Page 42'."""
        return f"{self.filename} — Page {self.page_number}"


@dataclass
class ChatResult:
    """
    Full output from the RAG Q&A pipeline.

    Attributes:
        answer:     Plain-text LLM answer.
        citations:  Deduplicated list of source citations ordered by relevance.
                    Empty if no relevant chunks were retrieved.
    """

    answer: str
    citations: list[SourceCitation] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        """True if at least one source citation is available."""
        return len(self.citations) > 0
