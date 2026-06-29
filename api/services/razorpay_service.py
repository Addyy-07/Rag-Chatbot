"""
api/services/razorpay_service.py
────────────────────────────────
Razorpay SDK integration.
"""

import razorpay
import hmac
import hashlib
from api.config import api_settings
import logging

log = logging.getLogger(__name__)

# Initialize Razorpay Client
client = razorpay.Client(auth=(api_settings.razorpay_key_id, api_settings.razorpay_key_secret))

def create_subscription(plan_id: str, total_count: int = 120) -> dict:
    """
    Create a new subscription for a plan in Razorpay.
    total_count: number of billing cycles (120 months = 10 years by default for 'unlimited' feel).
    """
    try:
        sub = client.subscription.create({
            "plan_id": plan_id,
            "total_count": total_count,
            "customer_notify": 1,
        })
        return sub
    except Exception as e:
        log.error(f"Failed to create subscription: {e}")
        raise e

def cancel_subscription(subscription_id: str, cancel_at_cycle_end: bool = True) -> dict:
    """
    Cancel an active subscription.
    """
    try:
        # Razorpay API doesn't support 'cancel_at_cycle_end' directly in the cancel call exactly like Stripe.
        # Calling cancel() immediately halts it.
        # But we will use cancel(cancel_at_cycle_end=True) if supported, else just cancel.
        # Actually Razorpay cancel API accepts cancel_at_cycle_end as a parameter.
        sub = client.subscription.cancel(subscription_id, {"cancel_at_cycle_end": int(cancel_at_cycle_end)})
        return sub
    except Exception as e:
        log.error(f"Failed to cancel subscription {subscription_id}: {e}")
        raise e

def verify_webhook_signature(payload_body: bytes, signature: str) -> bool:
    """
    Verify the HMAC SHA256 signature from Razorpay.
    """
    secret = api_settings.razorpay_webhook_secret
    expected_signature = hmac.new(
        key=secret.encode('utf-8'),
        msg=payload_body,
        digestmod=hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_signature, signature)

def create_order(amount_paise: int, currency: str = "INR", receipt: str = None) -> dict:
    """
    Create an order for standard checkout.
    """
    try:
        data = {
            "amount": amount_paise,
            "currency": currency,
        }
        if receipt:
            data["receipt"] = receipt
            
        order = client.order.create(data=data)
        return order
    except Exception as e:
        log.error(f"Failed to create razorpay order: {e}")
        raise e

def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Verify payment signature for Standard Checkout.
    """
    try:
        secret = api_settings.razorpay_key_secret
        msg = f"{order_id}|{payment_id}"
        expected_signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=msg.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
    except Exception as e:
        log.error(f"Failed to verify payment signature: {e}")
        return False
