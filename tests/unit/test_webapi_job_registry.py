from kalorie.webapi.job_registry import (
    IdempotencyConflictError,
    JobRegistry,
    JobSubmissionRequest,
)


def _request(
    *,
    key: str,
    payload_hash: str,
    market_ticker: str = "KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL",
) -> JobSubmissionRequest:
    return JobSubmissionRequest(
        idempotency_key=key,
        payload_hash=payload_hash,
        market_ticker=market_ticker,
        cpu_slots=1,
        openai_slots=1,
        provider_slots=1,
    )


def test_same_idempotency_key_same_payload_reuses_existing_job() -> None:
    registry = JobRegistry(
        max_active_jobs=3,
        max_cpu_slots=3,
        max_openai_slots=3,
        max_provider_slots=3,
    )

    first = registry.submit(_request(key="abc", payload_hash="hash-1"))
    second = registry.submit(_request(key="abc", payload_hash="hash-1"))

    assert first.job_id == second.job_id
    assert registry.total_jobs == 1


def test_same_idempotency_key_different_payload_raises_conflict() -> None:
    registry = JobRegistry(
        max_active_jobs=3,
        max_cpu_slots=3,
        max_openai_slots=3,
        max_provider_slots=3,
    )
    registry.submit(_request(key="abc", payload_hash="hash-1"))

    try:
        registry.submit(_request(key="abc", payload_hash="hash-2"))
    except IdempotencyConflictError:
        return
    raise AssertionError("Expected IdempotencyConflictError")


def test_budget_limit_queues_when_resources_exhausted() -> None:
    registry = JobRegistry(
        max_active_jobs=1,
        max_cpu_slots=1,
        max_openai_slots=1,
        max_provider_slots=1,
    )

    first = registry.submit(_request(key="first", payload_hash="hash-1"))
    second = registry.submit(_request(key="second", payload_hash="hash-2"))

    assert first.status == "running"
    assert second.status == "queued"
    assert second.wait_reason == "waiting_for_resources"


def test_complete_running_job_promotes_oldest_queued_job() -> None:
    registry = JobRegistry(
        max_active_jobs=1,
        max_cpu_slots=1,
        max_openai_slots=1,
        max_provider_slots=1,
    )
    first = registry.submit(_request(key="first", payload_hash="hash-1"))
    second = registry.submit(_request(key="second", payload_hash="hash-2"))

    registry.mark_completed(first.job_id)
    refreshed = registry.get(second.job_id)

    assert refreshed is not None
    assert refreshed.status == "running"
    assert refreshed.wait_reason is None
