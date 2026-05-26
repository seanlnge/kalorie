from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from kalorie.domain.models import MentionMarketContract, TargetPhrase
from kalorie.workflows.kalshi_event_pack import (
    KalshiEventPackCandidate,
    build_event_pack,
    select_latest_pre_cutoff_candle,
)


def _contract() -> MentionMarketContract:
    return MentionMarketContract(
        venue="kalshi",
        market_id="KXEARNINGSMENTIONWMT-26AUG-TFFIC",
        event_ticker="KXEARNINGSMENTIONWMT-26AUG",
        title='Will Walmart mention "traffic"?',
        rules_text='Settles yes if Walmart says "traffic".',
        target_phrase=TargetPhrase(phrase="traffic", normalized_phrase="traffic"),
        yes_bid=Decimal("0.40"),
        yes_ask=Decimal("0.45"),
        observed_at=datetime(2026, 8, 21, 15, 0, tzinfo=UTC),
    )


def test_select_latest_pre_cutoff_candle_never_selects_post_cutoff():
    cutoff = datetime(2026, 8, 21, 15, 50, tzinfo=UTC)
    before = {"end_period_ts": int((cutoff - timedelta(minutes=1)).timestamp()), "yes_bid": 40}
    after = {"end_period_ts": int((cutoff + timedelta(minutes=1)).timestamp()), "yes_bid": 60}

    selected = select_latest_pre_cutoff_candle([before, after], cutoff=cutoff)

    assert selected == before


def test_build_event_pack_writes_contracts_snapshots_and_summary(tmp_path: Path):
    class FakeKalshiClient:
        def get_event_mention_markets(self, event_ticker: str):
            assert event_ticker == "KXEARNINGSMENTIONWMT-26AUG"
            return [_contract()]

        def get_market_candlesticks(self, **kwargs):
            cutoff = datetime(2026, 8, 21, 15, 50, tzinfo=UTC)
            return {
                "candlesticks": [
                    {
                        "end_period_ts": int((cutoff - timedelta(minutes=1)).timestamp()),
                        "yes_bid": 41,
                        "yes_ask": 46,
                    },
                    {
                        "end_period_ts": int((cutoff + timedelta(minutes=1)).timestamp()),
                        "yes_bid": 99,
                        "yes_ask": 100,
                    },
                ]
            }

    candidate = KalshiEventPackCandidate(
        event_ticker="KXEARNINGSMENTIONWMT-26AUG",
        company_symbol="WMT",
        company_name="Walmart",
        fiscal_year=2026,
        fiscal_quarter=2,
        call_start_at=datetime(2026, 8, 21, 16, 0, tzinfo=UTC),
    )

    summary = build_event_pack(
        candidates=[candidate],
        kalshi_client=FakeKalshiClient(),
        output_dir=tmp_path,
    )

    event_dir = tmp_path / "KXEARNINGSMENTIONWMT-26AUG"
    assert summary["ready_count"] == 0
    assert (event_dir / "event.json").exists()
    assert (event_dir / "contracts.json").exists()
    assert (event_dir / "snapshots.json").exists()
    assert (tmp_path / "candidate-events-summary.json").exists()
