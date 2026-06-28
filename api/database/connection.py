"""
api/database/connection.py
──────────────────────────
MongoDB connection management using Motor (asyncio).
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from api.config import api_settings
import logging

log = logging.getLogger(__name__)

class DatabaseManager:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None

db_manager = DatabaseManager()

async def connect_to_mongo():
    """Connect to MongoDB."""
    log.info("Connecting to MongoDB...")
    db_manager.client = AsyncIOMotorClient(api_settings.mongodb_url)
    db_manager.db = db_manager.client[api_settings.mongodb_db_name]
    
    # Create indexes here if necessary
    await db_manager.db.users.create_index("email", unique=True)
    await db_manager.db.users.create_index("username", unique=True)
    
    log.info("Connected to MongoDB and ensured indexes.")

async def close_mongo_connection():
    """Close MongoDB connection."""
    if db_manager.client:
        log.info("Closing MongoDB connection...")
        db_manager.client.close()
        log.info("MongoDB connection closed.")

def get_database() -> AsyncIOMotorDatabase:
    """Dependency to get the database instance."""
    if db_manager.db is None:
        raise RuntimeError("Database not initialized")
    return db_manager.db
