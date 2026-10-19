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

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.user import UserRole


class PasswordRequest(BaseModel):
    password: str

    @field_validator("password")
    @classmethod
    def bcrypt_byte_boundary(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Passwords must not exceed 72 UTF-8 bytes")
        return value


class RegisterRequest(PasswordRequest):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(PasswordRequest):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthCapabilities(BaseModel):
    public_registration_enabled: bool


class UserPublic(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    created_at: datetime
