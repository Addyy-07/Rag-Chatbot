"""
api/routes/auth_router.py
─────────────────────────
Endpoints for user authentication (signup, login, logout).
"""

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.database.connection import get_database
from api.schemas.auth import SignupRequest, LoginRequest, TokenResponse, ForgotPasswordRequest, ResetPasswordRequest
from api.services import auth_service

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
