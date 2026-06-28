"""
backend/models/document.py
───────────────────────────
Pydantic data model representing a single ingested PDF document.

Why Pydantic (not dataclass)?
  - Built-in JSON serialisation/deserialisation → used by DocumentRegistry.
  - Field validation at construction time (e.g. non-empty filename).
  - model_dump() / model_validate() give clean dict ↔ model conversion.
  - Easy to extend with Optional fields for future SaaS metadata.

Usage
-----
    from backend.models.document import DocumentRecord

    record = DocumentRecord(
        document_id="abc-123",
        filename="report.pdf",
        upload_date="2026-06-28T08:00:00",
        page_count=10,
        chunk_count=42,
        namespace="abc-123",
        size_bytes=204800,
    )
    json_str = record.model_dump_json()
    restored = DocumentRecord.model_validate_json(json_str)
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class DocumentRecord(BaseModel):
    """
    Immutable metadata record for a single ingested PDF.

    Attributes:
        document_id:  UUID4 string — uniquely identifies this document.
        filename:     Original filename as uploaded (e.g. "annual_report.pdf").
        upload_date:  ISO-8601 UTC timestamp of when ingestion completed.
        page_count:   Number of pages in the original PDF.
        chunk_count:  Number of text chunks stored in the vector database.
        namespace:    Pinecone namespace name (equals document_id).
        size_bytes:   Raw file size in bytes.
    """

    document_id: str = Field(..., description="UUID4 document identifier")
    filename: str = Field(..., min_length=1, description="Original PDF filename")
    upload_date: str = Field(..., description="ISO-8601 UTC upload timestamp")
    page_count: int = Field(..., ge=0, description="Number of PDF pages")
    chunk_count: int = Field(..., ge=0, description="Number of vector chunks stored")
    namespace: str = Field(..., description="Pinecone namespace (== document_id)")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")

    @field_validator("filename")
    @classmethod
    def strip_filename(cls, v: str) -> str:
        """Strip accidental whitespace from filenames."""
        return v.strip()

    @property
    def display_name(self) -> str:
        """
        Truncated display name for UI labels (max 30 chars).

        Example: "very_long_annual_report_2026.pdf" → "very_long_annual_report_2026..."
        """
        if len(self.filename) <= 30:
            return self.filename
        stem = self.filename[: self.filename.rfind(".")]
        ext = self.filename[self.filename.rfind("."):]
        return stem[:27] + "..." + ext

    @property
    def upload_date_display(self) -> str:
        """
        Human-friendly upload date string.

        Example: "2026-06-28T08:00:00" → "Jun 28, 2026 08:00"
        """
        try:
            dt = datetime.fromisoformat(self.upload_date)
            return dt.strftime("%b %d, %Y %H:%M")
        except ValueError:
            return self.upload_date

    model_config = {"frozen": True}  # Immutable after construction


def make_upload_date() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()
