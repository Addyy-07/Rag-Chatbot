"""
api/models/user.py
──────────────────
MongoDB user model using Pydantic.
"""

from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema
from bson import ObjectId
from datetime import datetime

class PyObjectId(str):
    """
    Custom type to handle MongoDB ObjectIds in Pydantic models.
    Allows validation of ObjectId and serialization to string.
    """
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

class UserDocument(BaseModel):
    """
    MongoDB document shape for users.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    email: str
    username: str
    hashed_password: str = ""
    auth_provider: str = Field(default="credentials", description="Authentication provider: 'credentials' or 'google'")
    provider_id: str | None = None
    avatar_url: str | None = None
    full_name: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime | None = None
    is_active: bool = True
    tier: str = Field(default="free", description="User subscription tier: 'free' or 'pro'")
    
    # Billing fields
    subscription_id: str | None = None
    subscription_status: str | None = None
    subscription_end: datetime | None = None
    cancel_at_period_end: bool = False
    
    # Auth fields
    reset_password_token: str | None = None
    reset_password_expires: datetime | None = None
    
    # Email verification fields
    is_email_verified: bool = False
    otp_code: str | None = None
    otp_expires: datetime | None = None
    otp_resend_count: int = 0
    otp_verify_attempts: int = 0
    otp_last_sent_at: datetime | None = None
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
