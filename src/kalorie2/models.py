from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MarketCategory = Literal["earnings", "politics", "sports", "other"]


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime fields must be timezone-aware")
    return value


class Kalorie2Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrecloseSnapshot(Kalorie2Model):
    yes_bid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    yes_ask: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    yes_mid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    candle_end_ts: int
    staleness_seconds: int = Field(ge=0)
    volume: int = Field(default=0, ge=0)
    open_interest: int = Field(default=0, ge=0)
    yes_bid_size: int = Field(default=0, ge=0)
    yes_ask_size: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def bid_ask_must_be_ordered(self) -> "PrecloseSnapshot":
        if self.yes_bid > self.yes_ask:
            raise ValueError("yes_bid must be less than or equal to yes_ask")
        return self


class HistoricalMentionMarketRow(Kalorie2Model):
    market_ticker: str
    event_ticker: str
    series_ticker: str
    market_category: MarketCategory
    event_phrase: str
    market_name: str
    word_said: str
    normalized_word_said: str
    final_outcome: Literal["yes", "no"]
    status: str | None = None
    close_time: datetime
    snapshot_target_time: datetime
    preclose_yes_bid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    preclose_yes_ask: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    preclose_yes_mid: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    candle_end_ts: int
    snapshot_staleness_seconds: int = Field(ge=0)
    preclose_volume: int = Field(default=0, ge=0)
    preclose_open_interest: int = Field(default=0, ge=0)
    preclose_yes_bid_size: int = Field(default=0, ge=0)
    preclose_yes_ask_size: int = Field(default=0, ge=0)
    company_prior_call_count: int = Field(default=0, ge=0)
    company_avg_call_duration_minutes_prior: float = Field(default=0.0, ge=0.0)
    company_avg_qa_question_count_prior: float = Field(default=0.0, ge=0.0)
    company_avg_prepared_remarks_minutes_prior: float = Field(default=0.0, ge=0.0)
    company_qa_share_prior: float = Field(default=0.0, ge=0.0, le=1.0)
    company_question_count_trend_prior: float = 0.0
    company_transcript_coverage_count: int = Field(default=0, ge=0)
    company_transcript_style_available: int = Field(default=0, ge=0, le=1)
    company_avg_transcript_word_count_prior: float = Field(default=0.0, ge=0.0)
    company_avg_phrase_mentions_prior: float = Field(default=0.0, ge=0.0)
    settlement_ts: datetime | None = None
    source: str

    @field_validator("close_time", "snapshot_target_time", "settlement_ts")
    @classmethod
    def datetimes_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _require_timezone(value)

    @model_validator(mode="after")
    def bid_ask_must_be_ordered(self) -> "HistoricalMentionMarketRow":
        if self.preclose_yes_bid > self.preclose_yes_ask:
            raise ValueError("preclose_yes_bid must be less than or equal to preclose_yes_ask")
        return self


class SkippedMarket(Kalorie2Model):
    market_ticker: str | None = None
    event_ticker: str | None = None
    reason: str


class CollectionResult(Kalorie2Model):
    rows: list[HistoricalMentionMarketRow] = Field(default_factory=list)
    skipped_markets: list[SkippedMarket] = Field(default_factory=list)
    stats: dict[str, int | dict[str, int]]
