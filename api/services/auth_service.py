"""
api/services/auth_service.py
────────────────────────────
Authentication business logic: password hashing, JWT generation, login, signup.
"""

from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
import logging

from api.config import api_settings
from api.models.user import UserDocument
from api.schemas.auth import SignupRequest, LoginRequest, TokenResponse

log = logging.getLogger(__name__)

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(user_id: str, email: str) -> str:
    """Generate a JWT token for the authenticated user."""
    expire = datetime.utcnow() + timedelta(minutes=api_settings.jwt_expiry_minutes)
    to_encode = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    encoded_jwt = jwt.encode(
        to_encode, 
        api_settings.jwt_secret_key, 
        algorithm=api_settings.jwt_algorithm
    )
    return encoded_jwt

async def signup(request: SignupRequest, db: AsyncIOMotorDatabase) -> TokenResponse:
    """Register a new user and return a token."""
    # Check if email exists
    existing_email = await db.users.find_one({"email": request.email.lower()})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
        
    # Check if username exists
    existing_username = await db.users.find_one({"username": request.username})
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
        
    # Create user document
    new_user = UserDocument(
        email=request.email.lower(),
        username=request.username,
        hashed_password=hash_password(request.password),
        full_name=request.full_name,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow()
    )
    
    # Insert to DB
    result = await db.users.insert_one(new_user.model_dump(by_alias=True, exclude={"id"}))
    user_id = str(result.inserted_id)
    
    # Generate token
    token = create_access_token(user_id, new_user.email)
    
    log.info(f"New user registered: {new_user.email} ({user_id})")
    
    return TokenResponse(
        access_token=token,
        expires_in=api_settings.jwt_expiry_minutes * 60,
        user={
            "id": user_id,
            "email": new_user.email,
            "username": new_user.username,
            "full_name": new_user.full_name
        }
    )

async def login(request: LoginRequest, db: AsyncIOMotorDatabase) -> TokenResponse:
    """Authenticate user and return a token."""
    # Find user by email
    user_dict = await db.users.find_one({"email": request.email.lower()})
    if not user_dict:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
        
    user = UserDocument(**user_dict)
    
    # Verify password
    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
        
    # Check active status
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
        
    # Update last login
    await db.users.update_one(
        {"_id": ObjectId(user.id)},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    # Generate token
    token = create_access_token(str(user.id), user.email)
    
    log.info(f"User logged in: {user.email}")
    
    return TokenResponse(
        access_token=token,
        expires_in=api_settings.jwt_expiry_minutes * 60,
        user={
            "id": str(user.id),
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name
        }
    )

import secrets

async def forgot_password(email: str, db: AsyncIOMotorDatabase) -> dict:
    """Generate a reset token and mock sending an email."""
    user = await db.users.find_one({"email": email.lower()})
    if not user:
        # Don't reveal that the user doesn't exist for security
        return {"message": "If that email is in our system, we've sent a reset link."}
        
    # Generate token (in real app, use a secure URL-safe token)
    reset_token = secrets.token_urlsafe(32)
    
    # Store token and expiry (e.g., 1 hour)
    expires = datetime.utcnow() + timedelta(hours=1)
    
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "reset_password_token": reset_token,
            "reset_password_expires": expires
        }}
    )
    
    # MOCK SENDING EMAIL
    log.info(f"========== PASSWORD RESET MOCK EMAIL ==========")
    log.info(f"To: {email}")
    log.info(f"Link: http://localhost:3000/reset-password?token={reset_token}")
    log.info(f"===============================================")
    
    return {"message": "If that email is in our system, we've sent a reset link."}

async def reset_password(token: str, new_password: str, db: AsyncIOMotorDatabase) -> dict:
    """Validate token and reset the user's password."""
    user = await db.users.find_one({
        "reset_password_token": token,
        "reset_password_expires": {"$gt": datetime.utcnow()}
    })
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
        
    # Hash new password
    hashed_pwd = hash_password(new_password)
    
    # Update user and clear token
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"hashed_password": hashed_pwd},
         "$unset": {"reset_password_token": "", "reset_password_expires": ""}}
    )
    
    return {"message": "Password successfully reset."}
