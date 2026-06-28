"""
api/routes/chat_history_router.py
─────────────────────────────────
FastAPI router for chat history operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from api.middleware.auth_middleware import get_current_user
from api.models.user import UserDocument
from api.schemas.chat import (
    ChatSessionCreate, 
    ChatSessionUpdate, 
    MessageCreate, 
    ChatSessionSummaryResponse, 
    ChatSessionDetailResponse
)
from api.models.chat import MessageRecord, CitationRecord
from api.services import chat_history_service

router = APIRouter(prefix="/chats", tags=["Chat History"])


@router.get("", response_model=List[ChatSessionSummaryResponse])
async def list_chats(current_user: UserDocument = Depends(get_current_user)):
    """Get all chat sessions for the authenticated user."""
    return await chat_history_service.get_user_chat_sessions(str(current_user.id))


@router.post("", response_model=ChatSessionDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(
    chat_req: ChatSessionCreate, 
    current_user: UserDocument = Depends(get_current_user)
):
    """Create a new chat session."""
    return await chat_history_service.create_chat_session(
        user_id=str(current_user.id),
        title=chat_req.title,
        target_namespaces=chat_req.target_namespaces
    )


@router.get("/{session_id}", response_model=ChatSessionDetailResponse)
async def get_chat(
    session_id: str, 
    current_user: UserDocument = Depends(get_current_user)
):
    """Get a specific chat session with its messages."""
    return await chat_history_service.get_chat_session(session_id, str(current_user.id))


@router.patch("/{session_id}", response_model=ChatSessionDetailResponse)
async def update_chat(
    session_id: str, 
    update_req: ChatSessionUpdate,
    current_user: UserDocument = Depends(get_current_user)
):
    """Rename a chat session."""
    if not update_req.title:
        raise HTTPException(status_code=400, detail="Title is required")
        
    return await chat_history_service.update_chat_session(
        session_id=session_id, 
        user_id=str(current_user.id), 
        title=update_req.title
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    session_id: str, 
    current_user: UserDocument = Depends(get_current_user)
):
    """Delete a chat session."""
    await chat_history_service.delete_chat_session(session_id, str(current_user.id))


@router.post("/{session_id}/messages", status_code=status.HTTP_201_CREATED)
async def add_message(
    session_id: str, 
    msg_req: MessageCreate,
    current_user: UserDocument = Depends(get_current_user)
):
    """Append a message to an existing chat session."""
    citations = []
    if msg_req.citations:
        citations = [CitationRecord(**c) for c in msg_req.citations]
        
    record = MessageRecord(
        role=msg_req.role,
        content=msg_req.content,
        citations=citations
    )
    
    await chat_history_service.add_message_to_session(
        session_id=session_id, 
        user_id=str(current_user.id), 
        message=record
    )
    return {"status": "success"}
