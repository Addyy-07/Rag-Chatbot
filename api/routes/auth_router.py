"""
api/routes/auth_router.py
─────────────────────────
Endpoints for user authentication (signup, login, logout).
"""

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.database.connection import get_database
from api.schemas.auth import SignupRequest, LoginRequest, TokenResponse
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
