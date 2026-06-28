"""
api/routes/webhook_router.py
────────────────────────────
Handles asynchronous webhooks from Razorpay (e.g. payment success, subscription halted).
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from datetime import datetime, timezone
import logging

from api.database.connection import get_database
from api.services.razorpay_service import verify_webhook_signature

log = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

@router.post("/razorpay")
async def razorpay_webhook(request: Request, db = Depends(get_database)):
    """
    Razorpay Webhook Endpoint.
    Expects x-razorpay-signature header.
    """
    body = await request.body()
    signature = request.headers.get("x-razorpay-signature")
    
    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature")
        
    # Verify Signature
    try:
        is_valid = verify_webhook_signature(body, signature)
        if not is_valid:
            log.warning("Invalid Razorpay signature.")
            raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        log.error(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Verification failed")
        
    payload = await request.json()
    event = payload.get("event")
    
    log.info(f"Received Razorpay Webhook Event: {event}")
    
    try:
        if event == "subscription.charged":
            # A payment was successful for a subscription
            sub_entity = payload["payload"]["subscription"]["entity"]
            payment_entity = payload["payload"]["payment"]["entity"]
            
            sub_id = sub_entity["id"]
            user_doc = await db.users.find_one({"subscription_id": sub_id})
            
            if user_doc:
                # Update User
                await db.users.update_one(
                    {"_id": user_doc["_id"]},
                    {"$set": {
                        "tier": "pro",
                        "subscription_status": sub_entity["status"],
                        "subscription_end": datetime.fromtimestamp(sub_entity["current_end"], tz=timezone.utc),
                        "cancel_at_period_end": bool(sub_entity.get("cancel_at_cycle_end", 0))
                    }}
                )
                
                # Log Payment
                await db.payments.insert_one({
                    "user_id": str(user_doc["_id"]),
                    "razorpay_payment_id": payment_entity["id"],
                    "razorpay_subscription_id": sub_id,
                    "amount": payment_entity["amount"],
                    "currency": payment_entity["currency"],
                    "status": payment_entity["status"],
                    "created_at": datetime.utcnow()
                })
                
        elif event in ["subscription.halted", "subscription.cancelled"]:
            # Subscription stopped
            sub_entity = payload["payload"]["subscription"]["entity"]
            sub_id = sub_entity["id"]
            
            await db.users.update_one(
                {"subscription_id": sub_id},
                {"$set": {
                    "tier": "free",
                    "subscription_status": sub_entity["status"],
                }}
            )
            
    except Exception as e:
        log.error(f"Error processing webhook event {event}: {e}")
        # Return 200 anyway so Razorpay doesn't retry infinitely on our bug
        return {"status": "error"}
        
    return {"status": "ok"}
