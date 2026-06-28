"""
api/schemas/user.py
───────────────────
User profile schemas.
"""

from pydantic import BaseModel
from datetime import datetime

class UserProfile(BaseModel):
    id: str
    email: str
    username: str
    full_name: str | None
    created_at: datetime
    last_login: datetime | None

class UpdateProfileRequest(BaseModel):
    full_name: str | None = None
    username: str | None = None
