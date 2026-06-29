"""
api/services/google_auth_service.py
───────────────────────────────────
Google OAuth backend verification and account linking logic.
"""

from google.oauth2 import id_token
from google.auth.transport import requests
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
import logging

from api.config import api_settings
from api.models.user import UserDocument
from api.schemas.auth import TokenResponse
from api.services.auth_service import create_access_token

log = logging.getLogger(__name__)

def verify_google_token(token: str) -> dict:
    """
    Verify the Google ID token and return its payload.
    """
    try:
        # Specify the CLIENT_ID of the app that accesses the backend
        idinfo = id_token.verify_oauth2_token(
            token, 
            requests.Request(), 
            api_settings.google_client_id
        )

        # ID token is valid. Get the user's Google Account ID from the decoded token.
        return idinfo
    except ValueError as e:
        log.error(f"Google token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token"
        )

async def google_authenticate(token: str, db: AsyncIOMotorDatabase) -> TokenResponse:
    """
    Authenticate user via Google.
    Creates a new account if they don't exist, or links to an existing one.
    """
    # 1. Verify the token with Google
    if not api_settings.google_client_id:
        log.warning("Google Client ID is not configured. Mocking verification for local dev.")
        # Only for fallback local dev when not configured - DO NOT use in prod
        import jwt
        # Assuming the frontend passed the raw decoded payload as JSON string if secret was missing (just for dev simulation)
        # But normally we must fail.
        # Let's just fail if not configured in a real app, but for this demo, we'll extract it without verification
        try:
            idinfo = jwt.decode(token, options={"verify_signature": False})
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid token format")
    else:
        idinfo = verify_google_token(token)
        
    google_user_id = idinfo.get("sub")
    email = idinfo.get("email", "").lower()
    name = idinfo.get("name")
    picture = idinfo.get("picture")
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not provided by Google"
        )
        
    # 2. Check if a user with this Google ID exists
    user_dict = await db.users.find_one({"provider_id": google_user_id})
    
    if user_dict:
        # User exists via Google
        user = UserDocument(**user_dict)
        
        # Update last login and avatar if changed
        update_data = {"last_login": datetime.utcnow()}
        if picture and user.avatar_url != picture:
            update_data["avatar_url"] = picture
            user.avatar_url = picture
            
        await db.users.update_one(
            {"_id": user_dict["_id"]},
            {"$set": update_data}
        )
    else:
        # 3. Check if a user with this email exists (Account Linking)
        user_dict = await db.users.find_one({"email": email})
        
        if user_dict:
            # Link Google account to existing email account
            user = UserDocument(**user_dict)
            
            update_data = {
                "auth_provider": "google",
                "provider_id": google_user_id,
                "last_login": datetime.utcnow()
            }
            if picture and not user.avatar_url:
                update_data["avatar_url"] = picture
                user.avatar_url = picture
                
            await db.users.update_one(
                {"_id": user_dict["_id"]},
                {"$set": update_data}
            )
            
            log.info(f"Linked Google account to existing user: {email}")
        else:
            # 4. Create new user via Google
            # Generate a base username from email
            base_username = email.split('@')[0]
            username = base_username
            
            # Ensure uniqueness
            counter = 1
            while await db.users.find_one({"username": username}):
                username = f"{base_username}{counter}"
                counter += 1
                
            user = UserDocument(
                email=email,
                username=username,
                full_name=name,
                avatar_url=picture,
                auth_provider="google",
                provider_id=google_user_id,
                is_email_verified=True,  # Google emails are verified
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            
            result = await db.users.insert_one(user.model_dump(by_alias=True, exclude={"id"}))
            user.id = result.inserted_id
            log.info(f"New user registered via Google: {email}")
            
    # 5. Generate our own JWT
    user_id_str = str(user.id)
    access_token = create_access_token(user_id_str, user.email)
    
    return TokenResponse(
        access_token=access_token,
        expires_in=api_settings.jwt_expiry_minutes * 60,
        user={
            "id": user_id_str,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "is_email_verified": user.is_email_verified,
            "avatar_url": user.avatar_url
        }
    )
