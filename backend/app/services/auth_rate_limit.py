"""Account credential ceiling shared across replicas without storing email keys."""

import hashlib
import hmac
import time

import redis.asyncio as aioredis
from fastapi import HTTPException
from redis.exceptions import RedisError

from app.config import settings

ACCOUNT_ATTEMPTS_PER_MINUTE = 5
ACCOUNT_WINDOW_SECONDS = 60


def account_key(email: str, window: int) -> str:
    digest = hmac.new(
        settings.jwt_secret_key.encode(),
        f"credential-account:{email.casefold()}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"limit:credentials:{window}:{digest}"


async def check_account_limit(r: aioredis.Redis, email: str) -> None:
    """Count all attempts, including unknown users, before expensive bcrypt work.

    A fixed window has a bounded boundary burst (twice the minute allowance).
    INCR + EXPIRE execute in one transaction, so a crash cannot leave an
    immortal counter. Each window key expires within two minutes of creation.
    The peer-address endpoint ceiling remains a separate aggregate CPU guard.
    """
    now = int(time.time())
    key = account_key(email, now // ACCOUNT_WINDOW_SECONDS)
    ttl = ACCOUNT_WINDOW_SECONDS * 2 - now % ACCOUNT_WINDOW_SECONDS
    try:
        async with r.pipeline(transaction=True) as pipeline:
            pipeline.incr(key)
            pipeline.expire(key, ttl)
            count, _ = await pipeline.execute()
    except RedisError:
        raise HTTPException(503, "Authentication service temporarily unavailable") from None
    if count > ACCOUNT_ATTEMPTS_PER_MINUTE:
        raise HTTPException(
            429,
            "Too many credential attempts; try again shortly",
            headers={"Retry-After": str(ACCOUNT_WINDOW_SECONDS - now % ACCOUNT_WINDOW_SECONDS)},
        )
