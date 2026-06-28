"""
api/routes/billing_router.py
────────────────────────────
Endpoints for managing subscriptions and viewing billing history.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

from api.database.connection import get_database
from api.routes.auth_router import get_current_user
from api.models.user import UserDocument
from api.models.billing import PaymentDocument
from api.config import api_settings
from api.services.razorpay_service import create_subscription, cancel_subscription
import logging

log = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

class CreateSubscriptionRequest(BaseModel):
    interval: str # "monthly" or "yearly"

class CreateSubscriptionResponse(BaseModel):
    subscription_id: str
    plan_id: str

@router.get("/plans")
async def get_plans():
    """Returns available Razorpay Plan IDs from settings."""
    return {
        "monthly": api_settings.razorpay_plan_monthly,
        "yearly": api_settings.razorpay_plan_yearly,
    }

@router.post("/create-subscription", response_model=CreateSubscriptionResponse)
async def api_create_subscription(
    req: CreateSubscriptionRequest,
    current_user: UserDocument = Depends(get_current_user),
    db = Depends(get_database)
):
    """Creates a Razorpay subscription for the user."""
    # Only allow free users or cancelled users to create new subscriptions
    if current_user.tier == "pro" and current_user.subscription_status in ["active", "authenticated"]:
        raise HTTPException(status_code=400, detail="User already has an active subscription.")
        
    plan_id = api_settings.razorpay_plan_monthly if req.interval == "monthly" else api_settings.razorpay_plan_yearly
    
    try:
        sub = create_subscription(plan_id)
        
        # Save intent in DB (optional, but good for tracking)
        await db.users.update_one(
            {"_id": current_user.id},
            {"$set": {
                "subscription_id": sub["id"],
                "subscription_status": sub["status"] # usually 'created'
            }}
        )
        
        return CreateSubscriptionResponse(subscription_id=sub["id"], plan_id=plan_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel-subscription")
async def api_cancel_subscription(
    current_user: UserDocument = Depends(get_current_user),
    db = Depends(get_database)
):
    """Cancels the active subscription at period end."""
    if not current_user.subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription found.")
        
    try:
        sub = cancel_subscription(current_user.subscription_id, cancel_at_cycle_end=True)
        
        await db.users.update_one(
            {"_id": current_user.id},
            {"$set": {
                "cancel_at_period_end": True
            }}
        )
        return {"status": "ok", "message": "Subscription cancelled. It will remain active until the billing period ends."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[dict])
async def get_billing_history(
    current_user: UserDocument = Depends(get_current_user),
    db = Depends(get_database)
):
    """Returns the user's payment history."""
    cursor = db.payments.find({"user_id": str(current_user.id)}).sort("created_at", -1)
    payments = []
    for doc in await cursor.to_list(length=100):
        doc["_id"] = str(doc["_id"])
        payments.append(doc)
    return payments
