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
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

from backend.config.settings import settings
from backend.models.document import DocumentRecord
from backend.db.mongo_client import db_manager
from backend.utils.logger import get_logger

log = get_logger(__name__)


class DocumentRegistry:
    """
    MongoDB-backed registry of DocumentRecord objects.
    """

    def __init__(self, registry_path: Optional[str] = None) -> None:
        # registry_path is kept for backward compatibility signature, but ignored.
        self.db = db_manager.get_db()
        # Ensure index on document_id for fast lookups
        self.db.document_registry.create_index([("document_id", 1)], unique=True)
        self.db.document_registry.create_index([("owner_id", 1)])

    # ── Public API ─────────────────────────────────────────────────────────────

    def add(self, record: DocumentRecord) -> None:
        """
        Persist a new DocumentRecord to the MongoDB registry.
        """
        data = record.model_dump()
        self.db.document_registry.replace_one(
            {"document_id": record.document_id},
            data,
            upsert=True
        )
        log.info("Registry: added document '%s' (%s) to MongoDB.", record.filename, record.document_id)

    def get_all(self, owner_id: str | None = None) -> list[DocumentRecord]:
        """
        Return all documents in the registry (newest first).
        If owner_id is provided, only return documents owned by that user.
        """
        query = {"owner_id": owner_id} if owner_id else {}
        cursor = self.db.document_registry.find(query).sort("upload_date", -1)
        records = []
        for item in cursor:
            try:
                # Remove MongoDB _id before validating
                item.pop("_id", None)
                records.append(DocumentRecord.model_validate(item))
            except Exception as exc:
                log.warning("Skipping malformed registry entry: %s", exc)
        return records

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        """
        Return a single DocumentRecord by its ID, or None if not found.
        """
        item = self.db.document_registry.find_one({"document_id": document_id})
        if item:
            item.pop("_id", None)
            try:
                return DocumentRecord.model_validate(item)
            except Exception as exc:
                log.warning("Malformed registry entry for %s: %s", document_id, exc)
        return None

    def delete(self, document_id: str, embeddings: Optional[HuggingFaceEmbeddings] = None, owner_id: str | None = None) -> bool:
        """
        Remove a document from the registry and purge its Pinecone namespace.
        """
        query = {"document_id": document_id}
        if owner_id:
            query["owner_id"] = owner_id
            
        result = self.db.document_registry.delete_one(query)
        
        if result.deleted_count == 0:
            log.warning("Registry: document '%s' not found for deletion or unauthorized.", document_id)
            return False

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

    def get_all_namespaces(self, owner_id: str | None = None) -> list[str]:
        """
        Return the Pinecone namespace for every registered document.

        Convenience method for the "All Docs" chat mode.

        Returns:
            List of namespace strings (== document_ids), newest first.
        """
        return [doc.namespace for doc in self.get_all(owner_id=owner_id)]

    def count(self, owner_id: str | None = None) -> int:
        """Return the total number of registered documents (for a given user if specified)."""
        query = {"owner_id": owner_id} if owner_id else {}
        return self.db.document_registry.count_documents(query)
