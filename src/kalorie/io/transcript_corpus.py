import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

TRANSCRIPT_FILENAME = re.compile(
    r"^(?P<year>\d{4})_Q(?P<quarter>[1-4])_(?P<symbol>.+?)_processed\.txt$",
    re.IGNORECASE,
)


class TranscriptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str
    company_symbol: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    path: Path
    event_ticker: str | None = None
    call_start_at: datetime | None = None
    call_time_source: str | None = None

    @field_validator("company_symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()


def scan_transcript_corpus(root: Path) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    for path in sorted(root.glob("*/*_processed.txt")):
        match = TRANSCRIPT_FILENAME.match(path.name)
        if not match:
            continue
        records.append(
            TranscriptRecord(
                company_name=path.parent.name,
                company_symbol=match.group("symbol"),
                fiscal_year=int(match.group("year")),
                fiscal_quarter=int(match.group("quarter")),
                path=path,
            )
        )
    return records


def transcript_period_index(record: TranscriptRecord) -> int:
    return (record.fiscal_year * 4) + (record.fiscal_quarter - 1)


def sort_transcript_records(records: list[TranscriptRecord]) -> list[TranscriptRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.company_symbol,
            transcript_period_index(record),
            str(record.path),
        ),
    )


def prior_transcript_records(
    records: list[TranscriptRecord],
    current: TranscriptRecord,
) -> list[TranscriptRecord]:
    current_index = transcript_period_index(current)
    current_symbol = current.company_symbol.upper()
    return [
        record
        for record in sort_transcript_records(records)
        if record.company_symbol.upper() == current_symbol
        and transcript_period_index(record) < current_index
    ]
