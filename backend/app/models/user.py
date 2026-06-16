"""User ORM model."""

import uuid
from datetime import datetime
from enum import Enum

from sqlmodel import Field, SQLModel

from app.models.types import EnumString


class UserRole(str, Enum):
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(unique=True, index=True, max_length=255)
    full_name: str = Field(max_length=255)
    hashed_password: str
    # EnumString stores the enum as VARCHAR (matching the migration's column)
    # instead of a native Postgres `userrole` type that was never created — same
    # treatment as Complaint.status, and reads come back as the enum.
    role: UserRole = Field(default=UserRole.analyst, sa_type=EnumString(UserRole))
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
