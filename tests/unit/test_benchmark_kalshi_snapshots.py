from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kalorie.benchmarking.kalshi_snapshots import hydrate_event_snapshots
from kalorie.benchmarking.packs import BenchmarkEvent, BenchmarkMarket


class FakeKalshiClient:
    def __init__(self, payloads: dict[str, dict]):
        self.payloads = payloads
        self.calls = []

    def get_market_candlesticks(
        self,
        *,
        series_ticker: str,
        market_id: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> dict:
        self.calls.append(
            {
                "series_ticker": series_ticker,
                "market_id": market_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            }
        )
        return self.payloads[market_id]


def _event() -> BenchmarkEvent:
    return BenchmarkEvent(
        event_ticker="KXEARNINGSMENTIONTGT-26MAY20",
        company_symbol="TGT",
        company_name="Target Corporation",
        fiscal_year=2026,
        fiscal_quarter=1,
        call_start_at=datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
        evidence_cutoff_at=datetime(2026, 5, 20, 11, 50, tzinfo=UTC),
    )


def _market(market_id: str = "KXEARNINGSMENTIONTGT-26MAY20-BEAU") -> BenchmarkMarket:
    return BenchmarkMarket(
        event_ticker="KXEARNINGSMENTIONTGT-26MAY20",
        market_id=market_id,
        target_phrase="beauty",
        title="What will Target Corporation say during their next earnings call?",
        result="yes",
    )


def _candle(offset_minutes: int, *, bid: str | None = "0.93", ask: str | None = "0.95") -> dict:
    target = datetime(2026, 5, 20, 11, 50, tzinfo=UTC)
    candle = {
        "end_period_ts": int((target + timedelta(minutes=offset_minutes)).timestamp()),
        "price": {"close_dollars": "0.94"},
    }
    if bid is not None:
        candle["yes_bid"] = {"close_dollars": bid}
    if ask is not None:
        candle["yes_ask"] = {"close_dollars": ask}
    return candle


def test_hydrate_event_snapshots_chooses_latest_candle_not_after_target():
    market = _market()
    client = FakeKalshiClient(
        {
            market.market_id: {
                "candlesticks": [
                    _candle(-10, bid="0.70", ask="0.80"),
                    _candle(-1, bid="0.91", ask="0.94"),
                    _candle(1, bid="0.99", ask="1.00"),
                ]
            }
        }
    )

    result = hydrate_event_snapshots(client, _event(), [market], lookback_minutes=120)

    assert len(result.snapshots) == 1
    snapshot = result.snapshots[0]
    assert snapshot.market_id == market.market_id
    assert snapshot.preclose_yes_bid == Decimal("0.91")
    assert snapshot.preclose_yes_ask == Decimal("0.94")
    assert snapshot.candle_end_ts == _candle(-1)["end_period_ts"]
    assert snapshot.snapshot_target_time == datetime(2026, 5, 20, 11, 50, tzinfo=UTC)
    assert result.skipped_markets == []
    assert client.calls[0]["end_ts"] == int(snapshot.snapshot_target_time.timestamp())
    assert client.calls[0]["period_interval"] == 1


def test_hydrate_event_snapshots_skips_candles_missing_bid_or_ask():
    market = _market()
    client = FakeKalshiClient(
        {
            market.market_id: {
                "candlesticks": [
                    _candle(-2, bid="0.91", ask=None),
                    _candle(-1, bid=None, ask="0.94"),
                ]
            }
        }
    )

    result = hydrate_event_snapshots(client, _event(), [market])

    assert result.snapshots == []
    assert result.skipped_markets[0].market_id == market.market_id
    assert result.skipped_markets[0].reason == "no_eligible_candle"


def test_hydrate_event_snapshots_skips_when_only_future_candles_exist():
    market = _market()
    client = FakeKalshiClient({market.market_id: {"candlesticks": [_candle(1)]}})

    result = hydrate_event_snapshots(client, _event(), [market])

    assert result.snapshots == []
    assert result.skipped_markets[0].reason == "no_eligible_candle"
