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
from api.services.razorpay_service import create_subscription, cancel_subscription, create_order, verify_payment_signature
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

class CreateOrderRequest(BaseModel):
    amount: int
    currency: str = "INR"

@router.post("/create-order")
async def api_create_order(
    req: CreateOrderRequest,
    current_user: UserDocument = Depends(get_current_user)
):
    """Creates a Razorpay order for Standard Checkout."""
    if req.amount < 100:
        raise HTTPException(status_code=400, detail="Amount must be at least 100 paise.")
        
    try:
        # Pass user ID as receipt for tracking
        order = create_order(amount_paise=req.amount, currency=req.currency, receipt=str(current_user.id))
        return {
            "order_id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create order.")

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

@router.post("/verify-payment")
async def api_verify_payment(
    req: VerifyPaymentRequest,
    current_user: UserDocument = Depends(get_current_user),
    db = Depends(get_database)
):
    """Verifies Razorpay payment signature."""
    is_valid = verify_payment_signature(req.razorpay_order_id, req.razorpay_payment_id, req.razorpay_signature)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid payment signature.")
        
    # Signature is valid, update user tier or record payment
    try:
        from datetime import datetime, timezone, timedelta
        
        # Example logic: granting Pro tier for 1 month for standard checkout payment
        await db.users.update_one(
            {"_id": current_user.id},
            {"$set": {
                "tier": "pro",
                "subscription_end": datetime.now(timezone.utc) + timedelta(days=30)
            }}
        )
        
        # Record payment
        await db.payments.insert_one({
            "user_id": str(current_user.id),
            "razorpay_payment_id": req.razorpay_payment_id,
            "razorpay_order_id": req.razorpay_order_id,
            "amount": 0, # In a real system, you'd fetch the actual amount charged from Razorpay
            "currency": "INR",
            "status": "captured",
            "created_at": datetime.utcnow()
        })
        
        return {"status": "success", "message": "Payment verified successfully."}
    except Exception as e:
        log.error(f"Error updating user after payment: {e}")
        raise HTTPException(status_code=500, detail="Payment verified but failed to update user status.")
