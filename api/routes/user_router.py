"""
api/routes/user_router.py
─────────────────────────
Endpoints for user profile management.
"""

from fastapi import APIRouter, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.database.connection import get_database
from api.middleware.auth_middleware import get_current_user
from api.models.user import UserDocument
from api.schemas.user import UserProfile, UpdateProfileRequest
from api.services import user_service

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/me", response_model=UserProfile)
async def get_my_profile(
    current_user: UserDocument = Depends(get_current_user)
):
    """Get the profile of the currently authenticated user."""
    return user_service.get_profile(current_user)

@router.patch("/me", response_model=UserProfile)
async def update_my_profile(
    request: UpdateProfileRequest,
    current_user: UserDocument = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Update profile information."""
    return await user_service.update_profile(current_user, request, db)

@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_account(
    current_user: UserDocument = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Delete user account."""
    await user_service.delete_account(current_user, db)
