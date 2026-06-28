"""
api/models/billing.py
─────────────────────
MongoDB schemas for Subscriptions and Payments.
"""

from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from api.models.user import PyObjectId

class SubscriptionDocument(BaseModel):
    """
    MongoDB document representing a Razorpay Subscription.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    razorpay_subscription_id: str
    plan_id: str
    status: str
    current_start: datetime | None = None
    current_end: datetime | None = None
    cancel_at_period_end: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

class PaymentDocument(BaseModel):
    """
    MongoDB document representing a successful payment via webhook.
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    razorpay_payment_id: str
    razorpay_subscription_id: str | None = None
    amount: int
    currency: str
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
