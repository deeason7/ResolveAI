"""
slowapi rate limiting setup.

slowapi wraps the limits library and integrates with FastAPI/Starlette.
Rate limit state is stored in Redis so it works across multiple API
replicas (unlike the default in-memory backend which is per-process).
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

# In production we store limit state in Redis so it's shared across replicas.
# In dev/test we use the in-process memory backend to avoid requiring Redis.
_storage = settings.redis_url if settings.environment == "production" else "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_storage,
    default_limits=["200/minute"],
)
