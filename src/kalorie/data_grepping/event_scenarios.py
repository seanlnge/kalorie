import json
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventScenarioCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    company_symbol: str
    company_name: str
    llm_model: str
    source_digest: str | None = None
    prompt_version: str = "event-dossier-v1"
    topics: list[str] = Field(default_factory=list)
    analyst_questions: list[str] = Field(default_factory=list)
    management_answers: list[str] = Field(default_factory=list)
    synthetic_call_snippets: list[str] = Field(default_factory=list)
    target_phrase_variants: dict[str, list[str]] = Field(default_factory=dict)
    source_rationales: list[str] = Field(default_factory=list)

    @field_validator("company_symbol")
    @classmethod
    def normalize_company_symbol(cls, value: str) -> str:
        return value.upper()

    def scenario_texts(self) -> list[str]:
        return [
            *self.topics,
            *self.analyst_questions,
            *self.management_answers,
            *self.synthetic_call_snippets,
        ]

    def has_transcript_like_output(self) -> bool:
        transcript_markers = re.compile(
            r"\b(operator|moderator|speaker\s+\d+|prepared remarks|transcript begins)\s*:",
            re.IGNORECASE,
        )
        return any(transcript_markers.search(text) for text in self.scenario_texts())


class OpenAIEventScenarioGenerator:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._client = client or OpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        *,
        event_id: str,
        company_symbol: str,
        company_name: str,
        target_phrases: list[str],
        material_snippets: list[str],
        max_items: int = 8,
    ) -> EventScenarioCatalog:
        prompt = _scenario_prompt(
            event_id=event_id,
            company_symbol=company_symbol,
            company_name=company_name,
            target_phrases=target_phrases,
            material_snippets=material_snippets,
            max_items=max_items,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = _parse_json_payload(content)
        catalog = EventScenarioCatalog(
            event_id=event_id,
            company_symbol=company_symbol,
            company_name=company_name,
            llm_model=self._model,
            topics=_coerce_string_list(payload.get("topics", []), max_items),
            analyst_questions=_coerce_string_list(
                payload.get("analyst_questions", []),
                max_items,
            ),
            management_answers=_coerce_string_list(
                payload.get("management_answers", []),
                max_items,
            ),
            synthetic_call_snippets=_coerce_string_list(
                payload.get("synthetic_call_snippets", []),
                max_items,
            ),
            target_phrase_variants=_coerce_phrase_variants(
                payload.get("target_phrase_variants", {}),
                max_items,
            ),
            source_rationales=_coerce_string_list(payload.get("source_rationales", []), max_items),
        )
        if catalog.has_transcript_like_output():
            raise ValueError("Scenario catalog looks like a transcript excerpt")
        return catalog


class _PromptPayload(BaseModel):
    event_id: str
    company_symbol: str
    company_name: str
    target_phrases: list[str]
    snippets: list[str]
    max_items: int = Field(ge=1, le=25)


def _scenario_prompt(
    *,
    event_id: str,
    company_symbol: str,
    company_name: str,
    target_phrases: list[str],
    material_snippets: list[str],
    max_items: int,
) -> str:
    payload = _PromptPayload(
        event_id=event_id,
        company_symbol=company_symbol,
        company_name=company_name,
        target_phrases=target_phrases,
        snippets=material_snippets,
        max_items=max_items,
    )
    return (
        "Generate a compact pre-call event scenario catalog for an earnings mention-market "
        "model. Use only the supplied pre-call snippets and target phrases. Do not quote or "
        "infer from the real transcript. Output strict JSON with keys `topics`, "
        "`analyst_questions`, `management_answers`, `synthetic_call_snippets`, "
        "`target_phrase_variants`, and `source_rationales`. "
        "`target_phrase_variants` must be an object keyed by normalized target phrase, "
        "with each value an array of concise alternate wordings.\n"
        + payload.model_dump_json(indent=2)
    )


def _coerce_string_list(values: object, max_items: int) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()][:max_items]


def _parse_json_payload(content: str) -> dict[str, Any]:
    candidates = [content]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    brace_index = content.find("{")
    if brace_index >= 0:
        try:
            payload, _ = json.JSONDecoder().raw_decode(content[brace_index:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    preview = content.strip().replace("\n", " ")[:200]
    raise ValueError(f"OpenAI event scenario response was not valid JSON: {preview}")


def _coerce_phrase_variants(values: object, max_items: int) -> dict[str, list[str]]:
    if not isinstance(values, dict):
        return {}
    variants: dict[str, list[str]] = {}
    for key, raw_items in values.items():
        phrase = str(key).strip().lower()
        items = _coerce_string_list(raw_items, max_items)
        if phrase and items:
            variants[phrase] = items
    return variants


_SYSTEM_PROMPT = (
    "You create bounded, evidence-grounded earnings-call scenario catalogs. "
    "Never use real transcript text; transcripts are label-only."
)
