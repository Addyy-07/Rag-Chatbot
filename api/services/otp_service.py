"""
api/services/otp_service.py
────────────────────────────
OTP generation, storage, verification, and rate-limiting logic.
"""

import random
import string
import logging
from datetime import datetime, timedelta

from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.config import api_settings
from api.services.email_service import send_otp_email

log = logging.getLogger(__name__)


def generate_otp() -> str:
    """Generate a cryptographically random 6-digit OTP."""
    return "".join(random.choices(string.digits, k=6))


async def send_verification_otp(email: str, db: AsyncIOMotorDatabase) -> dict:
    """
    Generate an OTP, persist it, and send it via email.
    Enforces:
      - 60-second cooldown between sends
      - Maximum resend attempts (default 5)
    """
    email_lower = email.lower()
    user = await db.users.find_one({"email": email_lower})

    if not user:
        # Don't reveal whether the email exists
        return {"message": "If that email is registered, an OTP has been sent."}

    if user.get("is_email_verified", False):
        return {"message": "Email is already verified."}

    # ── Rate limiting: cooldown ───────────────────────────────────────────────
    last_sent = user.get("otp_last_sent_at")
    if last_sent:
        elapsed = (datetime.utcnow() - last_sent).total_seconds()
        if elapsed < api_settings.otp_cooldown_seconds:
            remaining = int(api_settings.otp_cooldown_seconds - elapsed)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Please wait {remaining} seconds before requesting another OTP.",
            )

    # ── Rate limiting: max resends ────────────────────────────────────────────
    resend_count = user.get("otp_resend_count", 0)
    if resend_count >= api_settings.otp_max_resend_attempts:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Maximum OTP resend attempts reached. Please contact support.",
        )

    # ── Generate and store OTP ────────────────────────────────────────────────
    otp_code = generate_otp()
    expires = datetime.utcnow() + timedelta(minutes=api_settings.otp_expiry_minutes)

    await db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "otp_code": otp_code,
                "otp_expires": expires,
                "otp_last_sent_at": datetime.utcnow(),
                "otp_verify_attempts": 0,  # Reset verify attempts on new OTP
            },
            "$inc": {"otp_resend_count": 1},
        },
    )

    # ── Send email ────────────────────────────────────────────────────────────
    sent = await send_otp_email(email_lower, otp_code)
    if not sent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP email. Please try again.",
        )

    log.info(f"OTP sent to {email_lower} (resend #{resend_count + 1})")
    return {"message": "Verification code sent to your email."}


async def verify_otp(
    email: str, otp_code: str, db: AsyncIOMotorDatabase
) -> dict:
    """
    Validate the OTP against the stored value.
    Enforces:
      - Maximum verification attempts (default 5)
      - Expiry check (default 10 minutes)
    On success: marks user as verified and clears all OTP fields.
    """
    email_lower = email.lower()
    user = await db.users.find_one({"email": email_lower})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or OTP.",
        )

    if user.get("is_email_verified", False):
        return {"message": "Email is already verified.", "is_verified": True}

    # ── Rate limiting: max verify attempts ────────────────────────────────────
    verify_attempts = user.get("otp_verify_attempts", 0)
    if verify_attempts >= api_settings.otp_max_verify_attempts:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Maximum verification attempts reached. Please request a new OTP.",
        )

    # Increment attempt counter immediately (before checking correctness)
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$inc": {"otp_verify_attempts": 1}},
    )

    # ── Check OTP exists ──────────────────────────────────────────────────────
    stored_otp = user.get("otp_code")
    if not stored_otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No OTP has been generated. Please request one first.",
        )

    # ── Check expiry ──────────────────────────────────────────────────────────
    otp_expires = user.get("otp_expires")
    if not otp_expires or datetime.utcnow() > otp_expires:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired. Please request a new one.",
        )

    # ── Check match ───────────────────────────────────────────────────────────
    if otp_code != stored_otp:
        remaining = api_settings.otp_max_verify_attempts - (verify_attempts + 1)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Incorrect OTP. {remaining} attempt(s) remaining.",
        )

    # ── Success: mark verified and clear OTP fields ───────────────────────────
    await db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"is_email_verified": True},
            "$unset": {
                "otp_code": "",
                "otp_expires": "",
                "otp_resend_count": "",
                "otp_verify_attempts": "",
                "otp_last_sent_at": "",
            },
        },
    )

    log.info(f"Email verified successfully: {email_lower}")
    return {"message": "Email verified successfully!", "is_verified": True}
