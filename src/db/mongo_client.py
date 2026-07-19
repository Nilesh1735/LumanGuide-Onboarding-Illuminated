"""
MongoDB client initialization.
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient

# Read the MongoDB connection string from environment variables (Render)
# Fallback to localhost for local development
MONGO_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "adaptive_rag")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]