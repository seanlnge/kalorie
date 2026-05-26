from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator, model_validator

from kalorie.domain.models import KalorieModel
from kalorie.ml.datasets import HistoricalTrainingExample

BenchmarkSplit = Literal["blind", "validation", "training"]
BenchmarkModelFamily = Literal[
    "global_base",
    "company_niched",
    "hierarchical",
    "ensemble_research",
]


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime fields must be timezone-aware")
    return value


class BenchmarkEvent(KalorieModel):
    event_ticker: str
    company_symbol: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    call_start_at: datetime
    evidence_cutoff_at: datetime

    @field_validator("company_symbol")
    @classmethod
    def normalize_company_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("call_start_at", "evidence_cutoff_at")
    @classmethod
    def datetimes_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def cutoff_must_not_follow_call_start(self) -> "BenchmarkEvent":
        if self.evidence_cutoff_at > self.call_start_at:
            raise ValueError("evidence_cutoff_at must be before or equal to call_start_at")
        return self


class BenchmarkMarket(KalorieModel):
    event_ticker: str
    market_id: str
    target_phrase: str
    title: str
    result: Literal["yes", "no"]
    aliases: list[str] = Field(default_factory=list)


class BenchmarkSnapshot(KalorieModel):
    event_ticker: str
    market_id: str
    preclose_yes_bid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    preclose_yes_ask: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    snapshot_target_time: datetime
    candle_end_ts: int = Field(ge=0)
    raw_candle: dict | None = None

    @field_validator("snapshot_target_time")
    @classmethod
    def snapshot_target_time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def quote_and_candle_must_be_valid(self) -> "BenchmarkSnapshot":
        if self.preclose_yes_bid > self.preclose_yes_ask:
            raise ValueError("preclose_yes_bid must be less than or equal to preclose_yes_ask")
        candle_end = datetime.fromtimestamp(self.candle_end_ts, tz=self.snapshot_target_time.tzinfo)
        if candle_end > self.snapshot_target_time:
            raise ValueError("candle_end_ts must not be after snapshot_target_time")
        return self


class BenchmarkEvidenceDocument(KalorieModel):
    event_ticker: str
    source_id: str
    company_symbol: str
    document_type: str
    source_path: str
    published_at: datetime
    content_hash: str
    cutoff_eligible: bool
    role: Literal["event_baseline", "time_sensitive"] = "time_sensitive"

    @field_validator("company_symbol")
    @classmethod
    def normalize_company_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("published_at")
    @classmethod
    def published_at_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class BenchmarkPackManifest(KalorieModel):
    pack_id: str
    split: BenchmarkSplit
    created_at: datetime
    description: str
    source_paths: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class BenchmarkPack(KalorieModel):
    manifest: BenchmarkPackManifest
    events: list[BenchmarkEvent]
    markets: list[BenchmarkMarket]
    snapshots: list[BenchmarkSnapshot]
    evidence: list[BenchmarkEvidenceDocument] = Field(default_factory=list)
    examples: list[HistoricalTrainingExample]


class BenchmarkRunMetadata(KalorieModel):
    model_family: BenchmarkModelFamily
    model_path: str
    pack_path: str
    excluded_events: list[str] = Field(default_factory=list)
    calibration: str | None = None


def validate_benchmark_pack(pack: BenchmarkPack) -> None:
    event_ids = {event.event_ticker for event in pack.events}
    market_ids = {market.market_id for market in pack.markets}
    snapshot_market_ids = {snapshot.market_id for snapshot in pack.snapshots}
    example_market_ids = {example.market_id for example in pack.examples}

    duplicate_markets = _duplicates([market.market_id for market in pack.markets])
    if duplicate_markets:
        raise ValueError(f"duplicate market metadata rows: {sorted(duplicate_markets)}")

    duplicate_snapshots = _duplicates([snapshot.market_id for snapshot in pack.snapshots])
    if duplicate_snapshots:
        raise ValueError(f"duplicate snapshots: {sorted(duplicate_snapshots)}")

    markets_missing_events = {
        market.event_ticker for market in pack.markets if market.event_ticker not in event_ids
    }
    if markets_missing_events:
        raise ValueError(f"markets reference missing events: {sorted(markets_missing_events)}")

    examples_missing_market_metadata = example_market_ids - market_ids
    if examples_missing_market_metadata:
        raise ValueError(
            "examples reference missing market metadata: "
            f"{sorted(examples_missing_market_metadata)}"
        )

    examples_missing_snapshots = example_market_ids - snapshot_market_ids
    if examples_missing_snapshots:
        raise ValueError(f"examples are missing snapshots: {sorted(examples_missing_snapshots)}")

    snapshots_missing_market_metadata = snapshot_market_ids - market_ids
    if snapshots_missing_market_metadata:
        raise ValueError(
            "snapshots reference missing market metadata: "
            f"{sorted(snapshots_missing_market_metadata)}"
        )


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
