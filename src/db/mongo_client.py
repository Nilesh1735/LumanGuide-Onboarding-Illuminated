"""
MongoDB client initialization.
"""
import os
import re
from urllib.parse import quote_plus
from motor.motor_asyncio import AsyncIOMotorClient

# Read the MongoDB connection string from environment variables (Render)
MONGO_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# Fix for passwords with special characters (e.g., @, #, :)
# This finds the username:password section and safely encodes it.
if MONGO_URL.startswith("mongodb+srv://") or MONGO_URL.startswith("mongodb://"):
    try:
        # Extract protocol
        protocol = "mongodb+srv://" if "mongodb+srv://" in MONGO_URL else "mongodb://"
        remainder = MONGO_URL.split(protocol, 1)[1]
        
        # Extract credentials and host
        if "@" in remainder:
            credentials, host_part = remainder.split("@", 1)
            if ":" in credentials:
                username, password = credentials.split(":", 1)
                # URL encode the username and password
                safe_username = quote_plus(username)
                safe_password = quote_plus(password)
                MONGO_URL = f"{protocol}{safe_username}:{safe_password}@{host_part}"
    except Exception as e:
        print(f"Warning: Could not parse MongoDB URI for special characters: {e}")

DB_NAME = os.getenv("MONGO_DB_NAME", "adaptive_rag")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]