"""
api/models/chat.py
──────────────────
MongoDB models for Chat Sessions and Messages.
"""

from typing import Annotated, Any, List
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from bson import ObjectId
from datetime import datetime

class PyObjectId(str):
    """Custom type to handle MongoDB ObjectIds in Pydantic models."""
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ])
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )
        
    @classmethod
    def validate(cls, value) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise ValueError("Invalid ObjectId")
        return ObjectId(value)


class CitationRecord(BaseModel):
    """Record for a single source citation."""
    source_file: str
    page_number: int | None = None
    chunk_text: str


class MessageRecord(BaseModel):
    """A single message within a chat session."""
    role: str  # "user" or "assistant"
    content: str
    citations: list[CitationRecord] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSessionDocument(BaseModel):
    """
    MongoDB document shape for a Chat Session.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str  # Store as string or PyObjectId depending on how it's handled in the DB
    title: str = "New Chat"
    messages: list[MessageRecord] = Field(default_factory=list)
    target_namespaces: list[str] = Field(default_factory=list)  # the context of this chat
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
