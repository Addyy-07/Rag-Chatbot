"""
api/routes/auth_router.py
─────────────────────────
Endpoints for user authentication (signup, login, logout, password reset, OTP verification).
"""

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.database.connection import get_database
from api.schemas.auth import (
    SignupRequest, LoginRequest, TokenResponse,
    ForgotPasswordRequest, ResetPasswordRequest,
    VerifyOTPRequest, ResendOTPRequest, VerifyOTPResponse,
)
from api.services import auth_service
from api.services import otp_service
from api.middleware.auth_middleware import get_current_user
from api.models.user import UserDocument

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup", response_model=TokenResponse, status_code=201)
async def signup(
    request: SignupRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Register a new user and return an access token."""
    return await auth_service.signup(request, db)

@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Authenticate user and return an access token."""
    return await auth_service.login(request, db)

@router.post("/logout")
async def logout():
    """
    Logout is handled client-side by discarding the JWT token.
    This endpoint exists for semantic completeness.
    """
    return {"message": "Successfully logged out"}

@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Request a password reset link."""
    return await auth_service.forgot_password(request.email, db)

@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Reset password using a valid token."""
    return await auth_service.reset_password(request.token, request.new_password, db)

# ── OTP Verification ─────────────────────────────────────────────────────────

@router.post("/verify-otp", response_model=VerifyOTPResponse)
async def verify_otp(
    request: VerifyOTPRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Verify a 6-digit OTP code."""
    result = await otp_service.verify_otp(request.email, request.otp_code, db)
    return VerifyOTPResponse(**result)

@router.post("/resend-otp")
async def resend_otp(
    request: ResendOTPRequest,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Resend the OTP verification email."""
    return await otp_service.send_verification_otp(request.email, db)

@router.get("/verification-status")
async def verification_status(
    current_user: UserDocument = Depends(get_current_user),
):
    """Check whether the current user's email is verified."""
    return {
        "email": current_user.email,
        "is_email_verified": current_user.is_email_verified,
    }
