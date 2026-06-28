"""
backend/db/mongo_client.py
──────────────────────────
Synchronous MongoDB client for the Streamlit backend.
Provides access to raw document chunks for on-the-fly BM25 retrieval.
"""

from pymongo import MongoClient
from backend.config.settings import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None

    def connect(self):
        if self.client is None:
            log.info("Connecting to MongoDB Atlas from Streamlit backend...")
            self.client = MongoClient(settings.mongodb_url)
            self.db = self.client[settings.mongodb_db_name]
            # Ensure indexes
            self.db.document_chunks.create_index([("namespace", 1)])
            log.info("Connected to MongoDB successfully.")

    def get_db(self):
        if self.db is None:
            self.connect()
        return self.db

db_manager = MongoDBManager()
