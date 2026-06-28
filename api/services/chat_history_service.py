"""
api/services/chat_history_service.py
────────────────────────────────────
CRUD operations for Chat Sessions in MongoDB.
"""

from bson import ObjectId
from datetime import datetime
from fastapi import HTTPException
from api.database.connection import db_manager
from api.models.chat import ChatSessionDocument, MessageRecord

async def get_user_chat_sessions(user_id: str):
    """Retrieve all chat sessions for a user (without full messages for summary)."""
    # Use aggregation to get message_count
    
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"updated_at": -1}},
        {"$project": {
            "title": 1,
            "target_namespaces": 1,
            "created_at": 1,
            "updated_at": 1,
            "user_id": 1,
            "message_count": {"$size": "$messages"}
        }},
        {"$limit": 100}
    ]
    
    results = await db_manager.db.chat_sessions.aggregate(pipeline).to_list(length=100)
    
    # Convert _id to id string
    for r in results:
        r["id"] = str(r["_id"])
        
    return results

async def get_chat_session(session_id: str, user_id: str):
    """Retrieve a specific chat session with its messages."""
    if not ObjectId.is_valid(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    session_data = await db_manager.db.chat_sessions.find_one({
        "_id": ObjectId(session_id),
        "user_id": user_id
    })
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Chat session not found")
        
    session_data["id"] = str(session_data["_id"])
    return session_data

async def create_chat_session(user_id: str, title: str, target_namespaces: list[str]):
    """Create a new chat session."""
    session = ChatSessionDocument(
        user_id=user_id,
        title=title,
        target_namespaces=target_namespaces
    )
    
    result = await db_manager.db.chat_sessions.insert_one(
        session.model_dump(by_alias=True)
    )
    
    created_session = await db_manager.db.chat_sessions.find_one({"_id": result.inserted_id})
    created_session["id"] = str(created_session["_id"])
    return created_session

async def update_chat_session(session_id: str, user_id: str, title: str):
    """Update a chat session (rename)."""
    if not ObjectId.is_valid(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    result = await db_manager.db.chat_sessions.update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {"$set": {"title": title, "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Chat session not found")
        
    return await get_chat_session(session_id, user_id)

async def delete_chat_session(session_id: str, user_id: str):
    """Delete a chat session."""
    if not ObjectId.is_valid(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    result = await db_manager.db.chat_sessions.delete_one({
        "_id": ObjectId(session_id), 
        "user_id": user_id
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat session not found")
        
    return True

async def add_message_to_session(session_id: str, user_id: str, message: MessageRecord):
    """Append a message to a chat session."""
    if not ObjectId.is_valid(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
        
    result = await db_manager.db.chat_sessions.update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {
            "$push": {"messages": message.model_dump()},
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Chat session not found")
        
    return True
