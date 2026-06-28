"""
api/models/usage.py
───────────────────
MongoDB schema for tracking user usage (queries per day).
"""

from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
from api.models.user import PyObjectId

def get_today_str() -> str:
    """Returns today's date as a string (YYYY-MM-DD)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

class UserUsageDocument(BaseModel):
    """
    MongoDB document shape for tracking daily user usage.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    date: str = Field(default_factory=get_today_str)
    query_count: int = 0
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
