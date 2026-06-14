"""Custom SQLAlchemy column types shared across the ORM models."""

from __future__ import annotations

from enum import Enum

from sqlalchemy import String
from sqlalchemy.types import TypeDecorator


class EnumString(TypeDecorator):
    """Persist a ``(str, Enum)`` member as VARCHAR but round-trip it as the Enum.

    SQLModel's default enum handling emits a *native* Postgres ENUM type, which
    makes every added value a migration. We deliberately store these as
    VARCHAR(50) instead — adding a status is then a code change, not a schema
    change. The plain ``String`` mapping had one wart: a row loaded from the DB
    came back as a bare ``str``, not the Enum, so ``.value`` raised and call
    sites grew ``isinstance(...)`` guards. This decorator centralizes the mapping
    and guarantees the attribute is the Enum in *both* directions.

    Bind also validates: an unknown string raises here rather than letting a
    status no code path expects reach the column.
    """

    impl = String
    cache_ok = True

    def __init__(self, enum_cls: type[Enum], length: int = 50) -> None:
        # Attribute names match the constructor params so SQLAlchemy's compiled-
        # query cache key (cache_ok=True) is derived correctly — distinct enum
        # classes => distinct cache keys, never a cross-type cache hit.
        self.enum_cls = enum_cls
        self.length = length
        super().__init__(length=length)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, self.enum_cls):
            return value.value
        # A raw string (or a value from elsewhere) — validate it maps to a real
        # member so an unknown status can never be written.
        return self.enum_cls(value).value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return self.enum_cls(value)
