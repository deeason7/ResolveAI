"""Tests for the shared stream reclaimer.

Driven against real fakeredis streams: produce, create a group, have one
consumer read (parking entries in the PEL), then verify reclaim behaviour.
"""

from __future__ import annotations

import fakeredis.aioredis
import pytest
from redis.exceptions import ResponseError

from app.config import settings
from app.workers.stream_utils import reclaim_stale_messages, trim_kwargs


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


class TestStreamRetention:
    """XACK clears the PEL, not the stream — so producers have to cap it."""

    def test_returns_maxlen_when_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "stream_maxlen", 500)
        assert trim_kwargs() == {"maxlen": 500, "approximate": True}

    def test_zero_disables_trimming(self, monkeypatch):
        monkeypatch.setattr(settings, "stream_maxlen", 0)
        assert trim_kwargs() == {}

    async def test_producers_bound_the_stream(self, redis_client, monkeypatch):
        """The regression that matters: an acked entry still occupies the stream."""
        from app.workers.classification_worker import enqueue_complaint

        monkeypatch.setattr(settings, "stream_maxlen", 5)
        for i in range(40):
            await enqueue_complaint(redis_client, f"00000000-0000-0000-0000-{i:012d}", stream="s")

        length = await redis_client.xlen("s")
        assert length <= 40, "trimming should not grow the stream"
        assert length < 40, "40 XADDs against maxlen=5 must have trimmed something"

    async def test_acked_entries_still_count_toward_length(self, redis_client, monkeypatch):
        """Documents *why* the cap exists, so nobody removes it as redundant."""
        monkeypatch.setattr(settings, "stream_maxlen", 0)
        from app.workers.classification_worker import enqueue_complaint

        for i in range(3):
            await enqueue_complaint(redis_client, f"00000000-0000-0000-0000-{i:012d}", stream="s")
        await redis_client.xgroup_create("s", "g", id="0")
        msgs = await redis_client.xreadgroup("g", "c", {"s": ">"}, count=10)
        for _stream, entries in msgs:
            for message_id, _fields in entries:
                await redis_client.xack("s", "g", message_id)

        groups = await redis_client.xinfo_groups("s")
        assert groups[0]["pending"] == 0, "everything acked"
        assert await redis_client.xlen("s") == 3, "yet the entries remain in the stream"

    async def test_resolution_producer_is_capped_too(self, redis_client, monkeypatch):
        from app.workers.resolution_worker import enqueue_resolution

        monkeypatch.setattr(settings, "stream_maxlen", 5)
        for i in range(40):
            await enqueue_resolution(redis_client, f"00000000-0000-0000-0000-{i:012d}", stream="r")

        assert await redis_client.xlen("r") < 40
