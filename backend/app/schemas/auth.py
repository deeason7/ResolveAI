"""
Pydantic schemas for auth request bodies and responses.

These are SEPARATE from the ORM models on purpose:
- ORM models define the DB table shape (hashed_password stored).
- Schemas define what the API accepts/returns (plain password in, never out).
Mixing them leaks internals and makes it easy to accidentally return
the hashed_password field in an API response.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from app.models.user import UserRole


class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserPublic(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    created_at: datetime
