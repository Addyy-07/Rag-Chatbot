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
    
    # Razorpay Settings
    razorpay_key_id: str = "rzp_test_change_me"
    razorpay_key_secret: str = "change_me"
    razorpay_webhook_secret: str = "change_me"
    razorpay_plan_monthly: str = "plan_change_me"
    razorpay_plan_yearly: str = "plan_change_me"
    
    # SMTP Settings
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    
    # OTP Settings
    otp_expiry_minutes: int = 10
    otp_max_resend_attempts: int = 5
    otp_max_verify_attempts: int = 5
    otp_cooldown_seconds: int = 60
    
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
