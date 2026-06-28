"""
backend/services/chat_api_client.py
───────────────────────────────────
Client functions to interact with the FastAPI Chat History endpoints.
"""

import httpx
from typing import List, Dict, Any
from backend.config.settings import settings

def _get_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def get_chat_sessions(token: str) -> List[Dict[str, Any]]:
    try:
        resp = httpx.get(
            f"{settings.api_base_url}/chats",
            headers=_get_headers(token),
            timeout=5.0
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
    return []

def get_chat_session(session_id: str, token: str) -> Dict[str, Any] | None:
    try:
        resp = httpx.get(
            f"{settings.api_base_url}/chats/{session_id}",
            headers=_get_headers(token),
            timeout=5.0
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
    return None

def create_chat_session(title: str, target_namespaces: List[str], token: str) -> str | None:
    try:
        resp = httpx.post(
            f"{settings.api_base_url}/chats",
            json={"title": title, "target_namespaces": target_namespaces},
            headers=_get_headers(token),
            timeout=5.0
        )
        if resp.status_code == 201:
            return resp.json().get("id")
    except Exception as e:
        pass
    return None

def add_message(session_id: str, role: str, content: str, citations: List[Dict[str, Any]], token: str) -> bool:
    try:
        resp = httpx.post(
            f"{settings.api_base_url}/chats/{session_id}/messages",
            json={
                "role": role,
                "content": content,
                "citations": citations
            },
            headers=_get_headers(token),
            timeout=5.0
        )
        return resp.status_code == 201
    except Exception as e:
        pass
    return False

def rename_chat_session(session_id: str, title: str, token: str) -> bool:
    try:
        resp = httpx.patch(
            f"{settings.api_base_url}/chats/{session_id}",
            json={"title": title},
            headers=_get_headers(token),
            timeout=5.0
        )
        return resp.status_code == 200
    except Exception as e:
        pass
    return False

def delete_chat_session(session_id: str, token: str) -> bool:
    try:
        resp = httpx.delete(
            f"{settings.api_base_url}/chats/{session_id}",
            headers=_get_headers(token),
            timeout=5.0
        )
        return resp.status_code == 204
    except Exception as e:
        pass
    return False
