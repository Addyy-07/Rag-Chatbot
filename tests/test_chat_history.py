import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from bson import ObjectId
from datetime import datetime

from api.models.chat import MessageRecord
from api.services.chat_history_service import (
    get_user_chat_sessions,
    get_chat_session,
    create_chat_session,
    update_chat_session,
    delete_chat_session,
    add_message_to_session
)
from fastapi import HTTPException

# Dummy data
USER_ID = "507f1f77bcf86cd799439011"
SESSION_ID = "607f1f77bcf86cd799439011"

@pytest.fixture
def mock_db():
    with patch("api.services.chat_history_service.db_manager") as mock_manager:
        yield mock_manager.db

@pytest.mark.asyncio
async def test_get_user_chat_sessions(mock_db):
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = [
        {"_id": ObjectId(SESSION_ID), "user_id": USER_ID, "title": "Test Chat", "target_namespaces": [], "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(), "message_count": 0}
    ]
    mock_db.chat_sessions.aggregate.return_value = mock_cursor
    
    sessions = await get_user_chat_sessions(USER_ID)
    
    assert len(sessions) == 1
    assert sessions[0]["id"] == SESSION_ID
    assert sessions[0]["title"] == "Test Chat"

@pytest.mark.asyncio
async def test_get_chat_session(mock_db):
    mock_db.chat_sessions.find_one = AsyncMock(return_value={
        "_id": ObjectId(SESSION_ID),
        "user_id": USER_ID,
        "title": "Test Chat",
        "messages": []
    })
    
    session = await get_chat_session(SESSION_ID, USER_ID)
    assert session["id"] == SESSION_ID
    assert session["title"] == "Test Chat"
    
@pytest.mark.asyncio
async def test_create_chat_session(mock_db):
    mock_insert_result = MagicMock()
    mock_insert_result.inserted_id = ObjectId(SESSION_ID)
    mock_db.chat_sessions.insert_one = AsyncMock(return_value=mock_insert_result)
    
    mock_db.chat_sessions.find_one = AsyncMock(return_value={
        "_id": ObjectId(SESSION_ID),
        "user_id": USER_ID,
        "title": "New Chat",
        "target_namespaces": []
    })
    
    session = await create_chat_session(USER_ID, "New Chat", [])
    assert session["id"] == SESSION_ID
    assert session["title"] == "New Chat"

@pytest.mark.asyncio
async def test_delete_chat_session(mock_db):
    mock_delete_result = MagicMock()
    mock_delete_result.deleted_count = 1
    mock_db.chat_sessions.delete_one = AsyncMock(return_value=mock_delete_result)
    
    result = await delete_chat_session(SESSION_ID, USER_ID)
    assert result is True
    
@pytest.mark.asyncio
async def test_delete_chat_session_not_found(mock_db):
    mock_delete_result = MagicMock()
    mock_delete_result.deleted_count = 0
    mock_db.chat_sessions.delete_one = AsyncMock(return_value=mock_delete_result)
    
    with pytest.raises(HTTPException) as exc:
        await delete_chat_session(SESSION_ID, USER_ID)
    assert exc.value.status_code == 404

@pytest.mark.asyncio
async def test_add_message_to_session(mock_db):
    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_db.chat_sessions.update_one = AsyncMock(return_value=mock_update_result)
    
    message = MessageRecord(role="user", content="Hello", citations=[])
    result = await add_message_to_session(SESSION_ID, USER_ID, message)
    assert result is True
