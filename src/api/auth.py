from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import bcrypt
import os
import jwt
import sqlite3
import logging

auth_router = APIRouter()
logger = logging.getLogger(__name__)

# Use a local SQLite database for users
DB_PATH = os.getenv("SQLITE_DB_PATH", "users.db")

def init_db():
    """Create the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            hashed_password TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database on startup
init_db()

class UserSignup(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    login_id: str  # Can be email or username
    password: str

@auth_router.post("/auth/signup")
def signup(user: UserSignup):
    """Create a new user account in the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if username or email already exists
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (user.username, user.email))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username or email already registered")
        
    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    hashed_password_str = hashed_password.decode('utf-8')
    
    # Save to database
    try:
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
            (user.username, user.email, hashed_password_str)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {"ok": True, "message": "User created successfully", "user_id": str(user_id)}
    except Exception as e:
        conn.close()
        logger.error(f"Signup failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user account.")

@auth_router.post("/auth/login")
def login(user: UserLogin):
    """Verify credentials against the database and issue a real JWT."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Look for the user by username OR email
    cursor.execute("SELECT id, hashed_password FROM users WHERE username = ? OR email = ?", (user.login_id, user.login_id))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials. Access denied.")
    
    user_id, hashed_password_str = row
    
    # Verify the provided password against the stored hash
    if not bcrypt.checkpw(user.password.encode('utf-8'), hashed_password_str.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials. Access denied.")
        
    # GENERATE REAL JWT TOKEN
    secret = os.getenv("JWT_SECRET", "supersecretjwt12345")
    token = jwt.encode({"sub": str(user_id)}, secret, algorithm="HS256")
    
    return {"ok": True, "token": token, "user_id": str(user_id)}