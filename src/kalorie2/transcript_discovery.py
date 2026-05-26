from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptDiscoveryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptSourceCandidate(TranscriptDiscoveryModel):
    fiscal_period: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    published_at: datetime | None
    transcript_candidate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)

    @field_validator("published_at")
    @classmethod
    def normalize_published_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class TranscriptDiscoveryPacket(TranscriptDiscoveryModel):
    company_name: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    candidates: list[TranscriptSourceCandidate] = Field(default_factory=list)

    def transcript_candidates(self) -> list[TranscriptSourceCandidate]:
        return [
            candidate
            for candidate in self.candidates
            if candidate.transcript_candidate and candidate.confidence >= 0.5
        ]


def build_transcript_discovery_prompt(
    *,
    company_name: str,
    ticker: str,
    fiscal_periods: list[str],
) -> str:
    payload = {
        "company_name": company_name,
        "ticker": ticker,
        "fiscal_periods": fiscal_periods,
        "instructions": [
            "Use web search to find public earnings-call transcript source candidates.",
            "Do not rely on memory; every candidate must come from a discoverable source URL.",
            "Return transcript_candidate=false for previews, recaps, audio pages, or pages that "
            "are not likely to contain the raw transcript text.",
            "The downstream matcher will verify cached transcript text deterministically.",
        ],
        "required_schema": _transcript_discovery_schema(),
    }
    return (
        "Find earnings-call transcript source candidates for mention-market base-rate features. "
        "Return only strict JSON matching `required_schema`.\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def build_openai_transcript_discovery_payload(*, prompt: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "tools": [{"type": "web_search"}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "transcript_discovery_packet",
                "strict": True,
                "schema": _transcript_discovery_schema(),
            }
        },
    }


def parse_transcript_discovery_response(text: str) -> TranscriptDiscoveryPacket:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("transcript discovery response must be strict JSON") from exc
    return TranscriptDiscoveryPacket.model_validate(payload)


def _transcript_discovery_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
            "ticker": {"type": "string"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fiscal_period": {"type": "string"},
                        "source_url": {"type": "string"},
                        "source_name": {"type": "string"},
                        "published_at": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "transcript_candidate": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "fiscal_period",
                        "source_url",
                        "source_name",
                        "published_at",
                        "transcript_candidate",
                        "confidence",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["company_name", "ticker", "candidates"],
        "additionalProperties": False,
    }
