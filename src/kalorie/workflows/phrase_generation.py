import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kalorie.ml.labeling import find_kalshi_settlement_mentions, normalize_phrase
from kalorie.ml.synthetic_phrases import STOPWORDS, TOKEN_PATTERN
from kalorie.workflows.models import (
    PhraseCatalog,
    PhraseCatalogEntry,
    TranscriptInventoryRow,
    WorkflowSkippedRecord,
)

GENERIC_PHRASES = {
    "business",
    "company",
    "earnings",
    "financial",
    "growth",
    "quarter",
    "revenue",
    "sales",
}
ALLOWED_THREE_WORD_PHRASES = {"brick and mortar"}
ABSENT_FALLBACK_CANDIDATES = [
    "OpenAI",
    "omnichannel",
    "automotive",
    "brick and mortar",
    "robotaxi",
    "tariff",
    "cloud",
    "crypto",
    "automation",
    "inflation",
    "inventory",
    "pricing",
    "capex",
    "advertising",
    "streaming",
    "semiconductor",
]


class OpenAIPhraseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    present_phrases: list[str] = Field(default_factory=list)
    absent_phrases: list[str] = Field(default_factory=list)

    @field_validator("present_phrases", "absent_phrases")
    @classmethod
    def values_must_be_strings(cls, value: list[str]) -> list[str]:
        return [phrase.strip() for phrase in value if phrase.strip()]


def is_simple_kalshi_phrase(phrase: str) -> bool:
    normalized = normalize_phrase(phrase)
    if not normalized:
        return False
    words = normalized.split()
    if len(words) > 2 and normalized not in ALLOWED_THREE_WORD_PHRASES:
        return False
    if len(words) == 1 and (words[0] in STOPWORDS or words[0] in GENERIC_PHRASES):
        return False
    if len(normalized) > 40:
        return False
    if not re.fullmatch(r"[a-z0-9+.\-]+(?: [a-z0-9+.\-]+)*", normalized):
        return False
    return any(character.isalpha() for character in normalized)


def parse_openai_phrase_response(content: str) -> OpenAIPhraseResponse:
    payload = json.loads(content)
    return OpenAIPhraseResponse.model_validate(payload)


def build_openai_phrase_prompt(
    *,
    company_name: str,
    transcript_text: str,
    max_per_label: int = 12,
) -> str:
    clipped = transcript_text[:12000]
    return (
        "Generate simple Kalshi-style target words or short phrases for an earnings "
        f"mention market for {company_name}. Return strict JSON with keys "
        "`present_phrases` and `absent_phrases`. Each value must be an array of "
        f"6-{max_per_label} simple terms like OpenAI, omnichannel, automotive, or "
        "brick and mortar. Avoid long conceptual phrases.\n\n"
        f"Transcript:\n{clipped}"
    )


def generate_openai_phrase_response(
    *,
    client: Any,
    model: str,
    company_name: str,
    transcript_text: str,
    max_per_label: int = 12,
) -> OpenAIPhraseResponse:
    prompt = build_openai_phrase_prompt(
        company_name=company_name,
        transcript_text=transcript_text,
        max_per_label=max_per_label,
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        text={"format": {"type": "json_object"}},
    )
    return parse_openai_phrase_response(response.output_text)


def generate_validated_phrase_entries(
    row: TranscriptInventoryRow,
    *,
    transcript_text: str,
    openai_response: str | None = None,
    max_per_label: int = 12,
) -> list[PhraseCatalogEntry]:
    mined_present_candidates = _mine_present_phrases(
        transcript_text,
        max_candidates=max_per_label * 4,
    )
    present_candidates = mined_present_candidates
    absent_candidates: list[str] = []
    present_source = "deterministic"
    absent_source = "deterministic"

    if openai_response:
        parsed = parse_openai_phrase_response(openai_response)
        present_candidates = [*parsed.present_phrases, *mined_present_candidates]
        absent_candidates = parsed.absent_phrases
        present_source = "openai"
        absent_source = "openai"
    else:
        absent_candidates = ABSENT_FALLBACK_CANDIDATES

    present_entries = _validate_candidates(
        row,
        transcript_text=transcript_text,
        candidates=present_candidates,
        expected_present=True,
        max_per_label=max_per_label,
        source=present_source,
    )
    absent_entries = _validate_candidates(
        row,
        transcript_text=transcript_text,
        candidates=absent_candidates,
        expected_present=False,
        max_per_label=max_per_label,
        source=absent_source,
    )
    return [*present_entries, *absent_entries]


OpenAIResponseProvider = Callable[[TranscriptInventoryRow, str], str | None]
PhraseCatalogCheckpointWriter = Callable[[PhraseCatalog], None]


def build_phrase_catalog(
    *,
    rows: list[TranscriptInventoryRow],
    openai_response_provider: OpenAIResponseProvider | None = None,
    max_per_label: int = 12,
    max_workers: int = 1,
    checkpoint_writer: PhraseCatalogCheckpointWriter | None = None,
) -> PhraseCatalog:
    entries: list[PhraseCatalogEntry] = []
    skipped: list[WorkflowSkippedRecord] = []

    def catalog_snapshot() -> PhraseCatalog:
        return PhraseCatalog(entries=list(entries), skipped_records=list(skipped))

    def process_row(
        row: TranscriptInventoryRow,
    ) -> tuple[list[PhraseCatalogEntry], WorkflowSkippedRecord | None]:
        transcript_text = row.transcript_path.read_text(encoding="utf-8", errors="ignore")
        openai_response = None
        skip = None
        if openai_response_provider is not None:
            try:
                openai_response = openai_response_provider(row, transcript_text)
            except Exception as exc:
                skip = WorkflowSkippedRecord(
                    company_symbol=row.company_symbol,
                    company_name=row.company_name,
                    fiscal_year=row.fiscal_year,
                    fiscal_quarter=row.fiscal_quarter,
                    path=row.transcript_path,
                    reason="openai_phrase_generation_failed",
                    detail=str(exc),
                )
        try:
            row_entries = generate_validated_phrase_entries(
                row,
                transcript_text=transcript_text,
                openai_response=openai_response,
                max_per_label=max_per_label,
            )
        except Exception as exc:
            skipped.append(
                WorkflowSkippedRecord(
                    company_symbol=row.company_symbol,
                    company_name=row.company_name,
                    fiscal_year=row.fiscal_year,
                    fiscal_quarter=row.fiscal_quarter,
                    path=row.transcript_path,
                    reason="phrase_validation_failed",
                    detail=str(exc),
                )
            )
            row_entries = generate_validated_phrase_entries(
                row,
                transcript_text=transcript_text,
                max_per_label=max_per_label,
            )
        if not row_entries:
            skip = WorkflowSkippedRecord(
                company_symbol=row.company_symbol,
                company_name=row.company_name,
                fiscal_year=row.fiscal_year,
                fiscal_quarter=row.fiscal_quarter,
                path=row.transcript_path,
                reason="missing_valid_phrases",
            )
        return row_entries, skip

    if max_workers > 1 and len(rows) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(process_row, row): row for row in rows}
            for future in as_completed(future_to_row):
                row_entries, skip = future.result()
                entries.extend(row_entries)
                if skip is not None:
                    skipped.append(skip)
                if checkpoint_writer is not None:
                    checkpoint_writer(catalog_snapshot())
    else:
        for row in rows:
            row_entries, skip = process_row(row)
            entries.extend(row_entries)
            if skip is not None:
                skipped.append(skip)
            if checkpoint_writer is not None:
                checkpoint_writer(catalog_snapshot())

    return catalog_snapshot()


def _mine_present_phrases(transcript_text: str, *, max_candidates: int) -> list[str]:
    tokens = TOKEN_PATTERN.findall(transcript_text)
    counts: Counter[str] = Counter()
    for token in tokens:
        if is_simple_kalshi_phrase(token):
            counts[token] += 1

    # Keep one-word terms first, with a light allowance for explicitly market-like phrases.
    for phrase in ALLOWED_THREE_WORD_PHRASES:
        if find_kalshi_settlement_mentions(transcript_text, phrase):
            counts[phrase] += 1

    return [
        phrase
        for phrase, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    ][:max_candidates]


def _validate_candidates(
    row: TranscriptInventoryRow,
    *,
    transcript_text: str,
    candidates: list[str],
    expected_present: bool,
    max_per_label: int,
    source: str,
) -> list[PhraseCatalogEntry]:
    entries: list[PhraseCatalogEntry] = []
    seen: set[str] = set()
    for phrase in candidates:
        normalized = normalize_phrase(phrase)
        if normalized in seen or not is_simple_kalshi_phrase(phrase):
            continue
        matches = find_kalshi_settlement_mentions(transcript_text, normalized)
        if bool(matches) != expected_present:
            continue
        seen.add(normalized)
        entries.append(
            PhraseCatalogEntry(
                company_symbol=row.company_symbol,
                fiscal_year=row.fiscal_year,
                fiscal_quarter=row.fiscal_quarter,
                phrase=phrase.strip(),
                label="present" if expected_present else "absent",
                source=source,
                match_count=len(matches),
            )
        )
        if len(entries) >= max_per_label:
            break
    return entries
