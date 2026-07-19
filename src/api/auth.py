from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import bcrypt
import os
import jwt
import logging
from src.db.mongo_client import db  # Import the db instance directly

auth_router = APIRouter()
logger = logging.getLogger(__name__)

class UserSignup(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    login_id: str  # Can be email or username
    password: str

@auth_router.post("/auth/signup")
async def signup(user: UserSignup):
    """Create a new user account in MongoDB."""
    users_col = db.users
    
    # Check if username or email already exists
    existing_user = await users_col.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
        
    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    hashed_password_str = hashed_password.decode('utf-8')
    
    # Save to MongoDB
    try:
        result = await users_col.insert_one({
            "username": user.username,
            "email": user.email,
            "hashed_password": hashed_password_str
        })
        return {"ok": True, "message": "User created successfully", "user_id": str(result.inserted_id)}
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user account.")

@auth_router.post("/auth/login")
async def login(user: UserLogin):
    """Verify credentials against MongoDB and issue a real JWT."""
    users_col = db.users
    
    # Look for the user by username OR email
    db_user = await users_col.find_one({"$or": [{"username": user.login_id}, {"email": user.login_id}]})

    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials. Access denied.")
    
    hashed_password_str = db_user.get('hashed_password')
    if not hashed_password_str or not bcrypt.checkpw(user.password.encode('utf-8'), hashed_password_str.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials. Access denied.")
        
    # GENERATE REAL JWT TOKEN
    secret = os.getenv("JWT_SECRET", "supersecretjwt12345")
    token = jwt.encode({"sub": str(db_user['_id'])}, secret, algorithm="HS256")
    
    return {"ok": True, "token": token, "user_id": str(db_user['_id'])}