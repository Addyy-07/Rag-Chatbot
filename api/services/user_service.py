"""
api/services/user_service.py
────────────────────────────
User management business logic: view profile, update, delete account.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from fastapi import HTTPException, status
import logging

from api.models.user import UserDocument
from api.schemas.user import UserProfile, UpdateProfileRequest

log = logging.getLogger(__name__)

def get_profile(user: UserDocument) -> UserProfile:
    """Convert UserDocument to UserProfile schema."""
    return UserProfile(
        id=str(user.id),
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        created_at=user.created_at,
        last_login=user.last_login
    )

async def update_profile(
    user: UserDocument, 
    request: UpdateProfileRequest, 
    db: AsyncIOMotorDatabase
) -> UserProfile:
    """Update user's profile information."""
    update_data = {}
    if request.full_name is not None:
        update_data["full_name"] = request.full_name
        
    if request.username is not None and request.username != user.username:
        # Check if username is already taken
        existing = await db.users.find_one({"username": request.username})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        update_data["username"] = request.username
        
    if update_data:
        await db.users.update_one(
            {"_id": ObjectId(user.id)},
            {"$set": update_data}
        )
        log.info(f"User {user.email} updated profile: {update_data.keys()}")
        
        # Fetch updated user to return
        updated_dict = await db.users.find_one({"_id": ObjectId(user.id)})
        user = UserDocument(**updated_dict)
        
    return get_profile(user)

async def delete_account(user: UserDocument, db: AsyncIOMotorDatabase) -> None:
    """Delete the user account."""
    # Note: In a complete SaaS, this would also delete their Pinecone namespaces
    # and S3/local files. We will handle document isolation later.
    await db.users.delete_one({"_id": ObjectId(user.id)})
    log.info(f"User {user.email} deleted their account")
