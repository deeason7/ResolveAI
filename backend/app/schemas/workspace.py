"""
Schemas for the pipeline Workspace — the live view of the
classify -> resolve pipeline and the controls that feed it.
"""

from __future__ import annotations

from pydantic import BaseModel


class StreamInfo(BaseModel):
    """Live state of one Redis work stream, summed across its consumer groups.

    Best-effort: before any worker has created a group the stream doesn't exist,
    so these read as zero / None rather than erroring — the board still renders
    from the durable database counts.
    """

    name: str
    in_flight: int  # delivered to a worker, not yet acked (XPENDING)
    consumers: int  # worker processes currently attached to the group(s)
    lag: int | None  # enqueued but not yet delivered (Redis 7+; None otherwise)


class WorkspaceBoard(BaseModel):
    """One count per pipeline stage (from complaint.status) plus stream state.

    Stage truth is the database — status is the durable signal the workers write
    transactionally. The streams are the transient in-flight layer on top.
    """

    pending: int  # awaiting classification
    classified: int  # classified, low priority — no action needed
    escalated: int  # high priority, queued for the resolution agent
    agent_triggered: int  # resolution agent working on it
    draft_ready: int  # guardrail-passed draft awaiting human review
    needs_review: int  # agent failed / unavailable — human takes over
    resolved: int  # human approved — closed
    total: int
    classification_stream: StreamInfo
    resolution_stream: StreamInfo


class EnqueueResult(BaseModel):
    enqueued: int
    stream: str
