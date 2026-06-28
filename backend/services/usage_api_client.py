"""
backend/services/usage_api_client.py
────────────────────────────────────
Client for the SaaS Usage Limits FastAPI endpoints.
"""

import httpx
from typing import Dict, Any, Tuple
from backend.config.settings import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)

def get_usage_limits(token: str) -> Dict[str, Any]:
    """Get the current user's usage and limits."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = httpx.get(
            f"{settings.api_base_url}/usage/limits",
            headers=headers,
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        log.error("Failed to fetch usage limits: %s", exc)
        return {}


def track_query(token: str) -> Tuple[bool, str]:
    """
    Increment query count.
    Returns (success: bool, error_message: str).
    """
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = httpx.post(
            f"{settings.api_base_url}/usage/track-query",
            headers=headers,
            timeout=5.0,
        )
        if response.status_code == 429:
            data = response.json()
            return False, data.get("detail", "Query limit exceeded.")
        response.raise_for_status()
        return True, ""
    except Exception as exc:
        log.error("Failed to track query: %s", exc)
        return True, "" # Fail open if tracking is down


def check_upload_limit(token: str) -> Tuple[bool, str]:
    """
    Check if the user can upload a PDF.
    Returns (success: bool, error_message: str).
    """
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = httpx.post(
            f"{settings.api_base_url}/usage/check-upload",
            headers=headers,
            timeout=5.0,
        )
        if response.status_code == 429:
            data = response.json()
            return False, data.get("detail", "PDF upload limit exceeded.")
        response.raise_for_status()
        return True, ""
    except Exception as exc:
        log.error("Failed to check upload limit: %s", exc)
        return True, "" # Fail open
