"""AWS Lambda entrypoint for the 6h Kalorie S3 snapshot job."""

from __future__ import annotations

from typing import Any

from kalorie2.s3_snapshot import handler_from_event


def handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    del context
    return handler_from_event(event)
