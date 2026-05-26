from datetime import timedelta
from decimal import Decimal
from typing import Protocol

from pydantic import Field

from kalorie.benchmarking.packs import BenchmarkEvent, BenchmarkMarket, BenchmarkSnapshot
from kalorie.domain.models import KalorieModel


class CandlestickClient(Protocol):
    def get_market_candlesticks(
        self,
        *,
        series_ticker: str,
        market_id: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> dict:
        pass


class SnapshotSkip(KalorieModel):
    market_id: str
    reason: str


class SnapshotHydrationResult(KalorieModel):
    snapshots: list[BenchmarkSnapshot] = Field(default_factory=list)
    skipped_markets: list[SnapshotSkip] = Field(default_factory=list)


def hydrate_event_snapshots(
    client: CandlestickClient,
    event: BenchmarkEvent,
    markets: list[BenchmarkMarket],
    *,
    lookback_minutes: int = 120,
    series_ticker: str = "KXEARNINGSMENTION",
    period_interval: int = 1,
) -> SnapshotHydrationResult:
    target_time = event.evidence_cutoff_at
    start_time = target_time - timedelta(minutes=lookback_minutes)
    start_ts = int(start_time.timestamp())
    end_ts = int(target_time.timestamp())
    snapshots: list[BenchmarkSnapshot] = []
    skipped_markets: list[SnapshotSkip] = []

    for market in markets:
        try:
            payload = client.get_market_candlesticks(
                series_ticker=series_ticker,
                market_id=market.market_id,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=period_interval,
            )
        except Exception as exc:
            skipped_markets.append(
                SnapshotSkip(
                    market_id=market.market_id,
                    reason=f"fetch_failed:{type(exc).__name__}",
                )
            )
            continue

        candle = _select_latest_eligible_candle(payload.get("candlesticks", []), end_ts=end_ts)
        if candle is None:
            skipped_markets.append(
                SnapshotSkip(market_id=market.market_id, reason="no_eligible_candle")
            )
            continue

        snapshots.append(
            BenchmarkSnapshot(
                event_ticker=event.event_ticker,
                market_id=market.market_id,
                preclose_yes_bid=_candle_close_dollars(candle, "yes_bid"),
                preclose_yes_ask=_candle_close_dollars(candle, "yes_ask"),
                snapshot_target_time=target_time,
                candle_end_ts=int(candle["end_period_ts"]),
                raw_candle=candle,
            )
        )

    return SnapshotHydrationResult(snapshots=snapshots, skipped_markets=skipped_markets)


def _select_latest_eligible_candle(candles: list[dict], *, end_ts: int) -> dict | None:
    eligible = [
        candle
        for candle in candles
        if int(candle.get("end_period_ts", end_ts + 1)) <= end_ts
        and _has_candle_close(candle, "yes_bid")
        and _has_candle_close(candle, "yes_ask")
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda candle: int(candle["end_period_ts"]))


def _has_candle_close(candle: dict, key: str) -> bool:
    value = candle.get(key)
    return isinstance(value, dict) and value.get("close_dollars") is not None


def _candle_close_dollars(candle: dict, key: str) -> Decimal:
    return Decimal(str(candle[key]["close_dollars"]))
