"""
api/config.py
─────────────
Configuration settings for the FastAPI backend.
Reads from environment variables and `.env` file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class APISettings(BaseSettings):
    """
    FastAPI and Auth configurations.
    """
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "rag_chatbot_db"
    
    jwt_secret_key: str = "super-secret-jwt-key-please-change-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 1440  # 24 hours
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

@lru_cache(maxsize=1)
def get_api_settings() -> APISettings:
    return APISettings()

api_settings = get_api_settings()
