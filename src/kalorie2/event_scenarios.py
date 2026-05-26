import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScenarioModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TargetPhraseContext(ScenarioModel):
    target_phrase: str = Field(min_length=1)
    likely_contexts: list[str] = Field(min_length=1)
    unlikely_contexts: list[str]
    evidence_rationale: str = Field(min_length=1)

    def texts(self) -> list[str]:
        return [
            self.target_phrase,
            *self.likely_contexts,
            *self.unlikely_contexts,
            self.evidence_rationale,
        ]


class EventScenarioCatalog(ScenarioModel):
    event_ticker: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1)
    analyst_questions: list[str]
    management_language_patterns: list[str]
    source_rationales: list[str]
    target_phrase_contexts: list[TargetPhraseContext]

    def scenario_texts(self) -> list[str]:
        texts = [
            *self.topics,
            *self.analyst_questions,
            *self.management_language_patterns,
            *self.source_rationales,
        ]
        for context in self.target_phrase_contexts:
            texts.extend(context.texts())
        return texts


def build_event_scenario_prompt(
    *,
    event: dict[str, Any],
    target_phrases: list[str],
    evidence_snippets: list[dict[str, Any]],
) -> str:
    payload = {
        "event": event,
        "target_phrases": target_phrases,
        "evidence_snippets": evidence_snippets,
        "required_schema": {
            "event_ticker": "string",
            "company_name": "string",
            "topics": ["string"],
            "analyst_questions": ["string"],
            "management_language_patterns": ["string"],
            "source_rationales": ["string"],
            "target_phrase_contexts": [
                {
                    "target_phrase": "string",
                    "likely_contexts": ["string"],
                    "unlikely_contexts": ["string"],
                    "evidence_rationale": "string",
                }
            ],
        },
    }
    return (
        "Create a bounded pre-call earnings mention scenario catalog. "
        "Use only the supplied pre-call evidence snippets and target phrases. "
        "Do not use transcript text, call audio, post-call articles, or settlement results. "
        "Output strict JSON with exactly the keys in `required_schema`; do not include "
        "markdown or commentary.\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def parse_event_scenario_response(text: str) -> EventScenarioCatalog:
    payload = _parse_json_payload(text)
    catalog = EventScenarioCatalog.model_validate(payload)
    reject_transcript_like_output(catalog)
    return catalog


def reject_transcript_like_output(catalog: EventScenarioCatalog) -> None:
    for text in catalog.scenario_texts():
        if _TRANSCRIPT_MARKER_RE.search(text):
            raise ValueError("scenario catalog appears to contain transcript-like output")


def _parse_json_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("event scenario response must be strict JSON") from exc
    if isinstance(payload, dict):
        return payload
    preview = text.strip().replace("\n", " ")[:200]
    raise ValueError(f"event scenario response must be a strict JSON object: {preview}")


_TRANSCRIPT_MARKER_RE = re.compile(
    r"\b(operator|moderator|analyst|speaker\s+\d+|prepared remarks|transcript begins)\s*:",
    flags=re.IGNORECASE,
)
