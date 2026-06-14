"""Tests for the shared stream reclaimer.

Driven against real fakeredis streams: produce, create a group, have one
consumer read (parking entries in the PEL), then verify reclaim behaviour.
"""

from __future__ import annotations

import fakeredis.aioredis
import pytest
from redis.exceptions import ResponseError

from app.workers.stream_utils import reclaim_stale_messages


@pytest.fixture()
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


async def _strand(redis, stream, group, *ids):
    """Produce ids, create the group, read them as a consumer that then 'dies'.

    The entries end up in the PEL owned by consumerA with no ack — exactly the
    state a crash mid-processing leaves behind.
    """
    for cid in ids:
        await redis.xadd(stream, {"complaint_id": cid})
    await redis.xgroup_create(stream, group, id="0")
    await redis.xreadgroup(group, "consumerA", {stream: ">"}, count=100)


async def test_reclaims_and_handles_stranded(redis_client):
    await _strand(redis_client, "s", "g", "c1", "c2")
    handled: list[str] = []

    async def handle(message_id, fields):
        handled.append(fields["complaint_id"])
        await redis_client.xack("s", "g", message_id)  # mirror _handle's ack-on-success

    n = await reclaim_stale_messages(
        redis_client, stream="s", group="g", consumer="consumerB", handle=handle, min_idle_ms=0
    )
    assert n == 2
    assert sorted(handled) == ["c1", "c2"]
    assert (await redis_client.xpending("s", "g"))["pending"] == 0  # all acked, PEL clear


async def test_leaves_fresh_in_flight_messages_alone(redis_client):
    await _strand(redis_client, "s", "g", "c1")
    handled: list[str] = []

    async def handle(message_id, fields):
        handled.append(message_id)

    # A huge idle threshold: the just-read message isn't stale enough to steal,
    # so a healthy worker's in-flight message is never yanked mid-flight.
    n = await reclaim_stale_messages(
        redis_client, stream="s", group="g", consumer="consumerB", handle=handle, min_idle_ms=10**9
    )
    assert n == 0
    assert handled == []
    assert (await redis_client.xpending("s", "g"))["pending"] == 1  # still consumerA's


async def test_transient_failure_leaves_message_pending(redis_client):
    # _handle returns without acking on a transient failure; the reclaimed
    # message must then stay in the PEL (now owned by us) for the next sweep.
    await _strand(redis_client, "s", "g", "c1")

    async def handle(message_id, fields):
        return  # processed, but not settled — no ack

    n = await reclaim_stale_messages(
        redis_client, stream="s", group="g", consumer="consumerB", handle=handle, min_idle_ms=0
    )
    assert n == 1
    assert (await redis_client.xpending("s", "g"))["pending"] == 1  # not lost, retried later


async def test_survives_xautoclaim_error(redis_client):
    async def boom(*a, **k):
        raise ResponseError("NOGROUP No such key or consumer group")

    redis_client.xautoclaim = boom
    called = False

    async def handle(message_id, fields):
        nonlocal called
        called = True

    n = await reclaim_stale_messages(
        redis_client, stream="s", group="g", consumer="c", handle=handle, min_idle_ms=0
    )
    assert n == 0
    assert called is False
