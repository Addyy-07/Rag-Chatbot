import pytest
from jose import jwt
from passlib.context import CryptContext

from api.services.auth_service import hash_password, verify_password, create_access_token
from api.config import api_settings

class TestAuthService:
    def test_hash_password(self):
        plain_password = "my_secret_password"
        hashed = hash_password(plain_password)
        
        assert hashed != plain_password
        assert isinstance(hashed, str)
        assert len(hashed) > 20
        
    def test_verify_password(self):
        plain_password = "my_secret_password"
        hashed = hash_password(plain_password)
        
        assert verify_password(plain_password, hashed) is True
        assert verify_password("wrong_password", hashed) is False
        
    def test_create_access_token(self):
        user_id = "507f1f77bcf86cd799439011"
        email = "test@example.com"
        
        token = create_access_token(user_id, email)
        
        assert isinstance(token, str)
        
        # Verify token contents
        payload = jwt.decode(
            token, 
            api_settings.jwt_secret_key, 
            algorithms=[api_settings.jwt_algorithm]
        )
        
        assert payload["sub"] == user_id
        assert payload["email"] == email
        assert "exp" in payload
        assert "iat" in payload
