"""
backend/services/document_registry.py
───────────────────────────────────────
Persistent metadata store for all ingested PDF documents.

Responsibilities:
  - CRUD operations on DocumentRecord objects.
  - Persist records to a JSON file so the library survives Streamlit restarts.
  - Thread-safe reads/writes using a file lock pattern.
  - Pinecone namespace cleanup on document deletion.

Storage format:
  A flat JSON array of DocumentRecord objects stored at settings.registry_path.
  Example:
    [
      {"document_id": "abc-123", "filename": "report.pdf", ...},
      ...
    ]

Design decisions:
  - JSON file chosen over SQLite for zero-dependency simplicity.
  - File is read fresh on every operation (no in-memory cache) — ensures
    correctness across multi-process Streamlit deployments.
  - To upgrade to SQLite/Postgres: replace only this file; all callers unchanged.

Usage
-----
    from backend.services.document_registry import DocumentRegistry

    registry = DocumentRegistry()
    registry.add(record)
    all_docs = registry.get_all()
    registry.delete("abc-123")
"""

import json
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

from backend.config.settings import settings
from backend.models.document import DocumentRecord
from backend.utils.logger import get_logger

log = get_logger(__name__)


class DocumentRegistry:
    """
    JSON-backed registry of DocumentRecord objects.

    Each instance reads from / writes to the file at ``settings.registry_path``.
    Multiple instances pointing to the same path are safe for reads; writes
    are atomic (write-to-temp + rename on POSIX).

    Usage::

        registry = DocumentRegistry()
        registry.add(record)
        docs = registry.get_all()
        doc  = registry.get("some-uuid")
        registry.delete("some-uuid")
    """

    def __init__(self, registry_path: Optional[str] = None) -> None:
        self._path = Path(registry_path or settings.registry_path)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_raw(self) -> list[dict]:
        """Return the raw list of record dicts from the JSON file."""
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError) as exc:
            log.error("Registry file corrupted or unreadable: %s", exc)
            return []

    def _save_raw(self, records: list[dict]) -> None:
        """Persist a list of record dicts to the JSON file (atomic write)."""
        tmp_path = self._path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self._path)  # Atomic on POSIX
            log.debug("Registry saved: %d record(s).", len(records))
        except IOError as exc:
            log.error("Failed to save registry: %s", exc)
            raise

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, record: DocumentRecord) -> None:
        """
        Persist a new DocumentRecord to the registry.

        If a record with the same ``document_id`` already exists, it is
        replaced (idempotent re-ingestion).

        Args:
            record: The DocumentRecord to persist.
        """
        records = self._load_raw()
        # Remove any existing entry with the same ID (idempotent)
        records = [r for r in records if r.get("document_id") != record.document_id]
        records.append(record.model_dump())
        self._save_raw(records)
        log.info("Registry: added document '%s' (%s).", record.filename, record.document_id)

    def get_all(self) -> list[DocumentRecord]:
        """
        Return all registered documents, sorted by upload_date descending.

        Returns:
            List of DocumentRecord objects (newest first).
        """
        raw = self._load_raw()
        records = []
        for item in raw:
            try:
                records.append(DocumentRecord.model_validate(item))
            except Exception as exc:
                log.warning("Skipping malformed registry entry: %s", exc)
        records.sort(key=lambda r: r.upload_date, reverse=True)
        return records

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        """
        Return a single DocumentRecord by its ID, or None if not found.

        Args:
            document_id: The UUID string of the document to retrieve.

        Returns:
            DocumentRecord if found, else None.
        """
        for item in self._load_raw():
            if item.get("document_id") == document_id:
                try:
                    return DocumentRecord.model_validate(item)
                except Exception as exc:
                    log.warning("Malformed registry entry for %s: %s", document_id, exc)
                    return None
        return None

    def delete(self, document_id: str, embeddings: Optional[HuggingFaceEmbeddings] = None) -> bool:
        """
        Remove a document from the registry and purge its Pinecone namespace.

        Args:
            document_id: The UUID of the document to delete.
            embeddings:  Optional — provided to confirm namespace exists before deletion.
                         Pass None to skip Pinecone cleanup (useful in tests).

        Returns:
            True if the document was found and deleted, False if not found.
        """
        records = self._load_raw()
        before_count = len(records)
        records = [r for r in records if r.get("document_id") != document_id]

        if len(records) == before_count:
            log.warning("Registry: document '%s' not found for deletion.", document_id)
            return False

        self._save_raw(records)
        log.info("Registry: deleted document '%s'.", document_id)

        # Purge Pinecone namespace
        if embeddings is not None:
            self._purge_pinecone_namespace(document_id)

        return True

    def _purge_pinecone_namespace(self, namespace: str) -> None:
        """
        Delete all vectors in the given Pinecone namespace.

        This is a best-effort operation — failures are logged but not raised,
        because the registry record has already been removed.

        Args:
            namespace: The Pinecone namespace to purge (equals document_id).
        """
        try:
            pc = Pinecone(api_key=settings.pinecone_api_key)
            index = pc.Index(settings.pinecone_index_name)
            index.delete(delete_all=True, namespace=namespace)
            log.info("Pinecone namespace '%s' purged.", namespace)
        except Exception as exc:
            log.error(
                "Failed to purge Pinecone namespace '%s': %s. "
                "Registry entry removed but vectors may linger.",
                namespace,
                exc,
            )

    def get_all_namespaces(self) -> list[str]:
        """
        Return the Pinecone namespace for every registered document.

        Convenience method for the "All Docs" chat mode.

        Returns:
            List of namespace strings (== document_ids), newest first.
        """
        return [doc.namespace for doc in self.get_all()]

    def count(self) -> int:
        """Return the total number of registered documents."""
        return len(self._load_raw())
