from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

auth_router = APIRouter()

class UserCredentials(BaseModel):
    username: str
    password: str

@auth_router.post("/login")
async def login(credentials: UserCredentials):
    """Mock login endpoint that accepts any credentials for local testing."""
    return {
        "access_token": "mock_jwt_token_123",
        "token_type": "bearer",
        "username": credentials.username
    }

@auth_router.post("/signup")
async def signup(credentials: UserCredentials):
    """Mock signup endpoint."""
    return {
        "status": "ok",
        "message": f"User {credentials.username} created successfully."
    }