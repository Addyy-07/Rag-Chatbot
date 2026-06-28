"""
backend/utils/file_utils.py
────────────────────────────
Reusable helpers for temporary file handling.

Responsibility:
  - Create safe temporary files from in-memory bytes.
  - Guarantee cleanup even when exceptions occur (context manager pattern).
  - Keep all file-system side-effects isolated from business logic.

Usage
-----
    from backend.utils.file_utils import temp_pdf_path

    with temp_pdf_path(uploaded_bytes) as path:
        loader = PyPDFLoader(path)
        docs = loader.load()
    # File is automatically deleted here, even on error
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from backend.utils.logger import get_logger

log = get_logger(__name__)


@contextmanager
def temp_pdf_path(file_bytes: bytes) -> Generator[str, None, None]:
    """
    Context manager that writes bytes to a temporary .pdf file.

    Guarantees the temp file is deleted after the ``with`` block exits,
    whether it exits normally or via an exception.

    Args:
        file_bytes: Raw PDF bytes (e.g. from an uploaded Streamlit file).

    Yields:
        Absolute path (str) to the temporary file on disk.

    Example::

        with temp_pdf_path(uploaded_file.read()) as pdf_path:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
    """
    tmp_path: str = ""
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", mode="wb"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        log.debug("Temporary PDF written to %s (%d bytes)", tmp_path, len(file_bytes))
        yield tmp_path

    finally:
        if tmp_path and Path(tmp_path).exists():
            os.unlink(tmp_path)
            log.debug("Temporary file %s deleted", tmp_path)


def human_readable_size(size_bytes: int) -> str:
    """
    Convert a byte count into a human-friendly string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted string like "1.4 MB" or "312.0 KB".

    Example::

        label = human_readable_size(1_450_000)  # "1.4 MB"
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"
