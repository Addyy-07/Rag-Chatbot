"""
backend/utils/logger.py
───────────────────────
Centralised logging configuration for the entire application.

Why not print()?
  - print() is not thread-safe for concurrent requests.
  - print() offers no log levels, no timestamps, no filtering.
  - Structured logging integrates with monitoring systems (Datadog, CloudWatch, etc.).

Usage
-----
    from backend.utils.logger import get_logger

    log = get_logger(__name__)
    log.info("PDF ingested", extra={"chunks": 42, "pages": 10})
    log.error("Pinecone unreachable", exc_info=True)
"""

import logging
import sys
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO


def _build_handler() -> logging.StreamHandler:
    """Return a stdout StreamHandler with our standard formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def configure_root_logger(level: int = DEFAULT_LEVEL) -> None:
    """
    Configure the root logger once at application startup.

    Call this once from main.py or the application entrypoint.
    Subsequent calls are no-ops (guarded by handler check).
    """
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured — idempotent

    root.setLevel(level)
    root.addHandler(_build_handler())

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None, level: int = DEFAULT_LEVEL) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name:  Module name — pass __name__ for automatic namespacing.
        level: Override log level for this specific logger (optional).

    Returns:
        A configured Logger instance.
    """
    configure_root_logger()
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    return logger
