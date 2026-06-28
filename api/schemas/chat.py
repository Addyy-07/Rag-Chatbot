"""
api/schemas/chat.py
───────────────────
Pydantic schemas for Chat History API requests and responses.
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from api.models.chat import MessageRecord

class ChatSessionCreate(BaseModel):
    title: str = "New Chat"
    target_namespaces: list[str] = []

class ChatSessionUpdate(BaseModel):
    title: Optional[str] = None

class MessageCreate(BaseModel):
    role: str
    content: str
    citations: Optional[list[dict]] = None

class ChatSessionSummaryResponse(BaseModel):
    id: str
    user_id: str
    title: str
    target_namespaces: list[str]
    created_at: datetime
    updated_at: datetime
    message_count: int

class ChatSessionDetailResponse(BaseModel):
    id: str
    user_id: str
    title: str
    target_namespaces: list[str]
    messages: list[MessageRecord]
    created_at: datetime
    updated_at: datetime
