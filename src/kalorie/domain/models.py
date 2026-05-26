from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime fields must be timezone-aware")
    return value


class KalorieModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Company(KalorieModel):
    symbol: str
    name: str

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()


class EarningsEvent(KalorieModel):
    company_symbol: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    event_date: date

    @field_validator("company_symbol")
    @classmethod
    def normalize_company_symbol(cls, value: str) -> str:
        return value.upper()


class SourceDocument(KalorieModel):
    source_id: str
    company_symbol: str
    document_type: str
    source_path: str
    published_at: datetime
    content_hash: str

    @field_validator("company_symbol")
    @classmethod
    def normalize_company_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("published_at")
    @classmethod
    def published_at_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class DocumentChunk(KalorieModel):
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    section: str | None = None
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)

    @model_validator(mode="after")
    def token_range_must_be_ordered(self) -> "DocumentChunk":
        if self.token_end < self.token_start:
            raise ValueError("token_end must be greater than or equal to token_start")
        return self


class TargetPhrase(KalorieModel):
    phrase: str
    normalized_phrase: str
    aliases: list[str] = Field(default_factory=list)


class MarketSnapshot(KalorieModel):
    venue: str
    market_id: str
    title: str
    yes_bid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    yes_ask: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def bid_ask_must_be_ordered(self) -> "MarketSnapshot":
        if self.yes_bid > self.yes_ask:
            raise ValueError("yes_bid must be less than or equal to yes_ask")
        return self


class MentionMarketContract(KalorieModel):
    venue: str
    market_id: str
    event_ticker: str
    title: str
    rules_text: str
    target_phrase: TargetPhrase
    yes_bid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    yes_ask: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def bid_ask_must_be_ordered(self) -> "MentionMarketContract":
        if self.yes_bid > self.yes_ask:
            raise ValueError("yes_bid must be less than or equal to yes_ask")
        return self


class MatchSpan(KalorieModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    match_type: Literal["exact", "lexical"]

    @model_validator(mode="after")
    def span_must_be_ordered(self) -> "MatchSpan":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class MentionLabel(KalorieModel):
    target_phrase: str
    exact_mentioned: bool
    lexical_mentioned: bool
    match_spans: list[MatchSpan] = Field(default_factory=list)


class FeatureVector(KalorieModel):
    target_phrase: str
    features: dict[str, float]


class Prediction(KalorieModel):
    target_phrase: str
    model_version: str
    probability: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class PaperTradeComparison(KalorieModel):
    target_phrase: str
    model_probability: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    market_probability: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    edge: Decimal
    side: Literal["yes", "no", "skip"]
    reasons: list[str] = Field(default_factory=list)
    spread: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
