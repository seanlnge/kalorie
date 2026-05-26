from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4


class IdempotencyConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobSubmissionRequest:
    idempotency_key: str
    payload_hash: str
    market_ticker: str
    cpu_slots: int = 1
    openai_slots: int = 0
    provider_slots: int = 0


@dataclass
class JobRecord:
    job_id: str
    idempotency_key: str
    payload_hash: str
    market_ticker: str
    created_at: datetime
    status: str
    wait_reason: str | None
    cpu_slots: int
    openai_slots: int
    provider_slots: int


class JobRegistry:
    def __init__(
        self,
        *,
        max_active_jobs: int,
        max_cpu_slots: int,
        max_openai_slots: int,
        max_provider_slots: int,
    ) -> None:
        self._max_active_jobs = max_active_jobs
        self._max_cpu_slots = max_cpu_slots
        self._max_openai_slots = max_openai_slots
        self._max_provider_slots = max_provider_slots

        self._jobs: dict[str, JobRecord] = {}
        self._idempotency_to_job_id: dict[str, str] = {}
        self._queued_job_ids: deque[str] = deque()

        self._active_jobs = 0
        self._active_cpu_slots = 0
        self._active_openai_slots = 0
        self._active_provider_slots = 0

    @property
    def total_jobs(self) -> int:
        return len(self._jobs)

    def submit(self, request: JobSubmissionRequest) -> JobRecord:
        existing_job_id = self._idempotency_to_job_id.get(request.idempotency_key)
        if existing_job_id is not None:
            existing = self._jobs[existing_job_id]
            if existing.payload_hash != request.payload_hash:
                raise IdempotencyConflictError(
                    "Idempotency key already exists with different payload hash"
                )
            return existing

        job_id = uuid4().hex
        record = JobRecord(
            job_id=job_id,
            idempotency_key=request.idempotency_key,
            payload_hash=request.payload_hash,
            market_ticker=request.market_ticker,
            created_at=datetime.now(tz=UTC),
            status="queued",
            wait_reason=None,
            cpu_slots=request.cpu_slots,
            openai_slots=request.openai_slots,
            provider_slots=request.provider_slots,
        )
        self._jobs[job_id] = record
        self._idempotency_to_job_id[request.idempotency_key] = job_id

        if self._can_run(record):
            self._start_running(record)
        else:
            record.status = "queued"
            record.wait_reason = "waiting_for_resources"
            self._queued_job_ids.append(job_id)
        return record

    def mark_completed(self, job_id: str) -> None:
        record = self._jobs[job_id]
        if record.status == "running":
            self._release(record)
        record.status = "completed"
        record.wait_reason = None
        self._promote_queued()

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def _can_run(self, record: JobRecord) -> bool:
        return (
            self._active_jobs + 1 <= self._max_active_jobs
            and self._active_cpu_slots + record.cpu_slots <= self._max_cpu_slots
            and self._active_openai_slots + record.openai_slots <= self._max_openai_slots
            and self._active_provider_slots + record.provider_slots <= self._max_provider_slots
        )

    def _start_running(self, record: JobRecord) -> None:
        self._active_jobs += 1
        self._active_cpu_slots += record.cpu_slots
        self._active_openai_slots += record.openai_slots
        self._active_provider_slots += record.provider_slots
        record.status = "running"
        record.wait_reason = None

    def _release(self, record: JobRecord) -> None:
        self._active_jobs = max(0, self._active_jobs - 1)
        self._active_cpu_slots = max(0, self._active_cpu_slots - record.cpu_slots)
        self._active_openai_slots = max(0, self._active_openai_slots - record.openai_slots)
        self._active_provider_slots = max(0, self._active_provider_slots - record.provider_slots)

    def _promote_queued(self) -> None:
        remaining_queue: deque[str] = deque()
        while self._queued_job_ids:
            job_id = self._queued_job_ids.popleft()
            record = self._jobs[job_id]
            if record.status != "queued":
                continue
            if self._can_run(record):
                self._start_running(record)
            else:
                remaining_queue.append(job_id)
        self._queued_job_ids = remaining_queue

