from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Literal, Optional

from app.config import get_settings

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    queue_position: int = 0


class JobQueue:
    """Single-worker async queue. GPU tasks run serially.

    Streaming endpoints share the same GPU lock (`gpu_lock`) to avoid OOM
    while letting interactive chunks bypass the queue ordering.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=s.max_queue_size)
        self._jobs: Dict[str, Job] = {}
        self._tasks: Dict[str, Callable[[], Awaitable[Any]]] = {}
        self._ttl = s.job_ttl
        self._worker_task: Optional[asyncio.Task] = None
        self.gpu_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="job-queue-worker")

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def submit(self, task: Callable[[], Awaitable[Any]]) -> Job:
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job
        self._tasks[job.id] = task
        try:
            self._queue.put_nowait(job.id)
        except asyncio.QueueFull:
            del self._jobs[job.id]
            del self._tasks[job.id]
            raise
        self._recompute_positions()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        self._purge_expired()
        return self._jobs.get(job_id)

    def _recompute_positions(self) -> None:
        queued_ids = list(self._queue._queue)  # type: ignore[attr-defined]
        for idx, jid in enumerate(queued_ids):
            j = self._jobs.get(jid)
            if j is not None:
                j.queue_position = idx + 1

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            jid for jid, j in self._jobs.items()
            if j.finished_at is not None and now - j.finished_at > self._ttl
        ]
        for jid in expired:
            self._jobs.pop(jid, None)
            self._tasks.pop(jid, None)

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            task = self._tasks.pop(job_id, None)
            if job is None or task is None:
                continue
            job.status = "running"
            job.started_at = time.time()
            job.queue_position = 0
            try:
                async with self.gpu_lock:
                    job.result = await task()
                job.status = "succeeded"
            except Exception as e:  # noqa: BLE001
                log.exception("job %s failed", job_id)
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.finished_at = time.time()
                self._recompute_positions()
