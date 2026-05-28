from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WebEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebEvidenceItem(WebEvidenceModel):
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_type: Literal["company", "sec", "news", "analyst", "other"] = "other"
    published_at: datetime | None
    snippet: str = Field(min_length=1)
    target_phrases: list[str] = Field(default_factory=list)
    evidence_strength: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_direction: Literal["support", "against", "neutral"] = "neutral"

    @field_validator("target_phrases")
    @classmethod
    def normalize_target_phrases(cls, values: list[str]) -> list[str]:
        return sorted({value.strip().lower() for value in values if value.strip()})

    @field_validator("published_at", mode="before")
    @classmethod
    def ignore_partial_published_at(cls, value: datetime | str | None) -> datetime | str | None:
        if isinstance(value, str):
            cleaned = value.strip()
            date_prefix = cleaned.split("T", 1)[0]
            if (
                "?" in cleaned
                or re.search(r"[a-zA-Z]", date_prefix)
                or re.fullmatch(r"\d{4}(-\d{2})?", cleaned)
                or re.match(r"\d{4}-00-", cleaned)
                or re.match(r"\d{4}-\d{2}-00", cleaned)
            ):
                return None
            try:
                datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
            except ValueError:
                return None
        return value

    @field_validator("published_at")
    @classmethod
    def normalize_published_at(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value) if value is not None else None


class WebEvidencePacket(WebEvidenceModel):
    event_ticker: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    cutoff_time: datetime
    items: list[WebEvidenceItem] = Field(default_factory=list)

    @field_validator("cutoff_time")
    @classmethod
    def normalize_cutoff_time(cls, value: datetime) -> datetime:
        return _as_utc(value)

    def cutoff_safe_items(self, cutoff_time: datetime | None = None) -> list[WebEvidenceItem]:
        effective_cutoff = (
            min(self.cutoff_time, _as_utc(cutoff_time)) if cutoff_time else self.cutoff_time
        )
        return [
            item
            for item in self.items
            if item.published_at is not None and item.published_at <= effective_cutoff
        ]

    def features_for_target(
        self,
        target_phrase: str,
        cutoff_time: datetime | None = None,
    ) -> dict[str, float]:
        effective_cutoff = (
            min(self.cutoff_time, _as_utc(cutoff_time)) if cutoff_time else self.cutoff_time
        )
        retained = self.cutoff_safe_items(effective_cutoff)
        normalized_target = target_phrase.strip().lower()
        matching = [
            item
            for item in retained
            if item.relevance_score >= 0.35
            and (
                normalized_target in item.target_phrases
                or _token_overlap(normalized_target, item.snippet) > 0.0
            )
        ]
        matching_strengths = [item.evidence_strength for item in matching]
        matching_relevances = [item.relevance_score for item in matching]
        recency_hours = [
            (effective_cutoff - item.published_at).total_seconds() / 3600.0
            for item in matching
            if item.published_at is not None
        ]
        source_flags = _source_type_flags(matching)
        return {
            "web_evidence_available": 1.0 if retained else 0.0,
            "web_evidence_item_count": float(len(retained)),
            "web_evidence_target_overlap": 1.0 if matching else 0.0,
            "web_evidence_strength_max": max(
                (item.evidence_strength for item in matching),
                default=0.0,
            ),
            "web_evidence_cutoff_safe_count": float(len(retained)),
            "web_evidence_target_match_count": float(len(matching)),
            "web_evidence_target_match_share": (
                len(matching) / len(retained) if retained else 0.0
            ),
            "web_evidence_strength_mean": (
                sum(matching_strengths) / len(matching_strengths)
                if matching_strengths
                else 0.0
            ),
            "web_evidence_strength_sum": sum(matching_strengths),
            "web_evidence_relevance_mean": (
                sum(matching_relevances) / len(matching_relevances)
                if matching_relevances
                else 0.0
            ),
            "web_evidence_relevance_max": max(matching_relevances, default=0.0),
            "web_evidence_high_relevance_count": float(
                sum(1 for item in matching if item.relevance_score >= 0.75)
            ),
            "web_evidence_support_count": float(
                sum(1 for item in matching if item.evidence_direction == "support")
            ),
            "web_evidence_against_count": float(
                sum(1 for item in matching if item.evidence_direction == "against")
            ),
            "web_evidence_neutral_count": float(
                sum(1 for item in matching if item.evidence_direction == "neutral")
            ),
            "web_evidence_recency_min_hours": min(recency_hours, default=0.0),
            "web_evidence_recency_mean_hours": (
                sum(recency_hours) / len(recency_hours) if recency_hours else 0.0
            ),
            **source_flags,
        }


def build_web_evidence_prompt(
    *,
    event: dict[str, Any],
    target_phrases: list[str],
) -> str:
    payload = {
        "event": event,
        "target_phrases": target_phrases,
        "instructions": [
            "Find company-specific web/news evidence published before or at the cutoff.",
            "Only include sources worth using as forecasting evidence; skip generic results "
            "that do not materially inform any target phrase.",
            "Do not use earnings-call transcripts, call audio, post-call recaps, "
            "or resolution pages.",
            "Prefer company releases, reputable financial news, SEC filings, and analyst previews.",
            "If a source publication time is unavailable, include published_at as null.",
            "Set relevance_score to how specifically the source informs the target phrase, "
            "and evidence_direction to support, against, or neutral.",
            "Set source_type to one of company, sec, news, analyst, or other.",
        ],
        "required_schema": _web_evidence_schema(),
    }
    return (
        "Collect cutoff-safe web evidence for earnings mention forecasting. "
        "Return only strict JSON matching `required_schema`.\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def build_openai_web_search_payload(*, prompt: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "tools": [{"type": "web_search"}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "web_evidence_packet",
                "strict": True,
                "schema": _web_evidence_schema(),
            }
        },
    }


def parse_web_evidence_response(text: str) -> WebEvidencePacket:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("web evidence response must be strict JSON") from exc
    return WebEvidencePacket.model_validate(payload)


def _web_evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "event_ticker": {"type": "string"},
            "company_name": {"type": "string"},
            "cutoff_time": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "source_type": {
                            "type": "string",
                            "enum": ["company", "sec", "news", "analyst", "other"],
                        },
                        "published_at": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "snippet": {"type": "string"},
                        "target_phrases": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence_strength": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "relevance_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "evidence_direction": {
                            "type": "string",
                            "enum": ["support", "against", "neutral"],
                        },
                    },
                    "required": [
                        "title",
                        "url",
                        "source",
                        "source_type",
                        "published_at",
                        "snippet",
                        "target_phrases",
                        "evidence_strength",
                        "relevance_score",
                        "evidence_direction",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["event_ticker", "company_name", "cutoff_time", "items"],
        "additionalProperties": False,
    }


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _source_type_flags(items: list[WebEvidenceItem]) -> dict[str, float]:
    source_types = {_source_type(item) for item in items}
    return {
        "web_evidence_source_company": 1.0 if "company" in source_types else 0.0,
        "web_evidence_source_sec": 1.0 if "sec" in source_types else 0.0,
        "web_evidence_source_news": 1.0 if "news" in source_types else 0.0,
        "web_evidence_source_analyst": 1.0 if "analyst" in source_types else 0.0,
        "web_evidence_source_other": 1.0 if "other" in source_types else 0.0,
    }


def _source_type(item: WebEvidenceItem) -> str:
    if item.source_type != "other":
        return item.source_type
    text = " ".join([item.source, item.url, item.title]).lower()
    if "sec.gov" in text or re.search(r"\bsec\b", text) or "10-q" in text or "10-k" in text:
        return "sec"
    if any(term in text for term in ("investor", "ir.", "press release", "news release")):
        return "company"
    if any(term in text for term in ("analyst", "rating", "price target", "research note")):
        return "analyst"
    if any(
        term in text
        for term in (
            "news",
            "reuters",
            "bloomberg",
            "cnbc",
            "wsj",
            "barron",
            "yahoo",
            "ap ",
        )
    ):
        return "news"
    return "other"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
