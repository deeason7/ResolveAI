"""Shared Redis-stream helpers for the consumer-group workers.

Both the classification and resolution workers consume their stream the same
way: ``XREADGROUP '>'`` for new messages, process, ``XACK``. That delivery mode
never re-delivers a message already handed to a (now-dead) consumer — it sits in
the group's pending-entries list (PEL) and stays there. ``reclaim_stale_messages``
is the sweep that rescues those orphans, so a worker crash mid-message doesn't
strand a complaint forever.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from app.config import settings

logger = logging.getLogger(__name__)


def trim_kwargs() -> dict:
    """MAXLEN arguments for a producer's XADD, or nothing when trimming is off.

    XACK removes an entry from the pending list but leaves it in the stream, so
    an uncapped stream grows for the life of the deployment. ``approximate``
    lets Redis trim on node boundaries, which is what makes a capped XADD cost
    roughly the same as an uncapped one.

    The cap must stay well above any plausible in-flight backlog: trimming
    evicts entries even while a consumer still holds them pending, and a PEL row
    pointing at a message that no longer exists can never be reclaimed.
    """
    maxlen = settings.stream_maxlen
    return {"maxlen": maxlen, "approximate": True} if maxlen else {}


# Reclaim one page per call; the run loop invokes this each cycle, so a large
# orphan backlog drains over successive ticks instead of blocking one iteration.
RECLAIM_COUNT = 10

# A handler with the same shape as each worker's ``_handle``: process the message
# then ack iff it's settled. Reusing it means reclaimed messages take the exact
# same success / poison / transient path a freshly-read message does.
MessageHandler = Callable[[str, dict[str, str]], Awaitable[None]]


async def reclaim_stale_messages(
    redis: aioredis.Redis,
    *,
    stream: str,
    group: str,
    consumer: str,
    handle: MessageHandler,
    min_idle_ms: int,
    count: int = RECLAIM_COUNT,
) -> int:
    """Reassign PEL entries stranded by a dead/slow consumer and re-handle them.

    ``XREADGROUP '>'`` only delivers NEW messages, so an entry a crashed worker
    never acked would sit in the pending-entries list forever. ``XAUTOCLAIM``
    atomically transfers entries idle longer than ``min_idle_ms`` to ``consumer``
    and returns them, so we finish them via the same ``handle`` path a fresh read
    uses (which acks on success / poison and leaves transient failures pending
    for the next sweep).

    The idle threshold is the safety margin: a message merely in-flight on a
    healthy worker — e.g. a multi-second LLM call — is younger than ``min_idle_ms``
    and is left alone, so two live workers never fight over the same message.

    Returns the number of messages reclaimed (for tests / metrics).
    """
    try:
        # redis-py 4.2+/fakeredis return (next_cursor, [(id, fields)...], [deleted]).
        _cursor, messages, _deleted = await redis.xautoclaim(
            name=stream,
            groupname=group,
            consumername=consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
    except ResponseError:
        # NOGROUP (no group yet) or a transient Redis error — never fatal; the
        # next cycle tries again. ensure_group() runs before the loop in prod.
        logger.exception("XAUTOCLAIM failed on %s/%s; will retry next cycle", stream, group)
        return 0
    for message_id, fields in messages:
        logger.info("reclaimed stale message %s on %s from the PEL", message_id, stream)
        await handle(message_id, fields)
    return len(messages)
