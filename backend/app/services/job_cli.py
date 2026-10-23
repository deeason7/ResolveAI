"""Bounded operator access to the same transactional batch path as the API."""

import argparse
import asyncio
import json
import time

from fastapi import HTTPException
from sqlalchemy import func
from sqlmodel import select

from app.api.routes.workspace import job_health
from app.database import AsyncSessionLocal
from app.models.job import Job, JobKind, JobStatus
from app.services.job_batches import BATCH_MAX, enqueue_batch


def bounded_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= BATCH_MAX:
        raise argparse.ArgumentTypeError(f"count must be between 1 and {BATCH_MAX}")
    return count


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("classify", "resolve"):
        command = commands.add_parser(name)
        command.add_argument("count", type=bounded_count, nargs="?", default=50)
    commands.add_parser("monitor")
    wait = commands.add_parser("wait")
    wait.add_argument("kind", choices=("classify", "resolve"))
    wait.add_argument("timeout_minutes", type=int, nargs="?", default=30, choices=range(1, 1441))
    return result


async def main(args: argparse.Namespace) -> int:
    if args.command in {"classify", "resolve"}:
        kind = JobKind.classification if args.command == "classify" else JobKind.resolution
        async with AsyncSessionLocal() as session:
            try:
                jobs = await enqueue_batch(session, kind, args.count)
                await session.commit()
            except HTTPException as exc:
                await session.rollback()
                print(exc.detail)
                return 1
        print(json.dumps({"kind": kind.value, "enqueued": len(jobs)}))
        return 0
    if args.command == "monitor":
        async with AsyncSessionLocal() as session:
            print(json.dumps([item.model_dump(mode="json") for item in await job_health(session)]))
        return 0
    kind = JobKind.classification if args.kind == "classify" else JobKind.resolution
    deadline = time.monotonic() + args.timeout_minutes * 60
    while time.monotonic() < deadline:
        async with AsyncSessionLocal() as session:
            rows = (
                await session.exec(
                    select(Job.status, func.count(Job.id))
                    .where(Job.kind == kind)
                    .group_by(Job.status)
                )
            ).all()
        counts = dict(rows)
        pending = counts.get(JobStatus.queued, 0) + counts.get(JobStatus.running, 0)
        failed = counts.get(JobStatus.failed, 0)
        print(
            json.dumps({"kind": kind.value, "outstanding": pending, "failed": failed}), flush=True
        )
        if pending == 0:
            return 1 if failed else 0
        await asyncio.sleep(20)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parser().parse_args())))
