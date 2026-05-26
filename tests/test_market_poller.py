from datetime import UTC, datetime
from pathlib import Path

import httpx
from tests.test_saved_models import _write_bundle
from typer.testing import CliRunner

from kalorie2.market_poller import (
    ActiveMarketPoller,
    ActiveMarketRow,
    KalshiActiveMarketSource,
    MarketPollCacheStore,
    PollPredictionRow,
    app,
    default_poll_cache_root,
    normalize_active_market,
    preferred_model_name,
)


class StubMarketSource:
    def list_active_markets(self) -> list[ActiveMarketRow]:
        return [
            ActiveMarketRow(
                market_ticker="KXEARNINGSMENTIONAAPL-26APR30-AI",
                event_ticker="KXEARNINGSMENTIONAAPL-26APR30",
                series_ticker="KXEARNINGSMENTIONAAPL",
                event_title="What will Apple say during their next earnings call?",
                market_title="What will Apple say during their next earnings call? - AI",
                target_phrase="AI",
                yes_bid=0.37,
                yes_ask=0.4,
                yes_mid=0.385,
                volume=123,
            )
        ]


class StubScorer:
    def score_active_markets(
        self,
        markets: list[ActiveMarketRow],
        *,
        model_name: str,
    ) -> list[PollPredictionRow]:
        market = markets[0]
        return [
            PollPredictionRow(
                market_ticker=market.market_ticker,
                event_ticker=market.event_ticker,
                target_phrase=market.target_phrase,
                model_name=model_name,
                model_probability=0.31,
                market_probability=market.yes_mid,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                residual_delta=-0.29,
                side="NO",
                edge=0.06,
                cost=0.63,
                volume=market.volume,
            )
        ]


def test_normalize_active_market_builds_runtime_compatible_market_row() -> None:
    row = normalize_active_market(
        event_payload={
            "event_ticker": "KXEARNINGSMENTIONAAPL-26APR30",
            "series_ticker": "KXEARNINGSMENTIONAAPL",
            "event_title": "What will Apple say during their next earnings call?",
        },
        market_payload={
            "ticker": "KXEARNINGSMENTIONAAPL-26APR30-AI",
            "title": "What will Apple say during their next earnings call? - AI",
            "custom_strike": {"Word": "AI"},
            "yes_bid": 37,
            "yes_ask": 40,
            "volume": 123,
        },
    )

    assert row.market_ticker == "KXEARNINGSMENTIONAAPL-26APR30-AI"
    assert row.event_ticker == "KXEARNINGSMENTIONAAPL-26APR30"
    assert row.target_phrase == "AI"
    assert row.yes_bid == 0.37
    assert row.yes_ask == 0.4
    assert row.yes_mid == 0.385
    assert row.to_runtime_row()["preclose_yes_mid"] == "0.385"
    assert row.to_runtime_row()["word_said"] == "AI"


def test_poll_cache_store_writes_latest_history_and_trade_files(tmp_path: Path) -> None:
    store = MarketPollCacheStore(root=tmp_path)
    snapshot = ActiveMarketPoller(
        market_source=StubMarketSource(),
        scorer=StubScorer(),
        cache_store=store,
        now=lambda: datetime(2026, 5, 26, 4, 0, tzinfo=UTC),
    ).run_once(model_name="unit-model")

    latest = store.read_latest_snapshot()
    trades = store.read_latest_trades()

    assert snapshot.poll_id == "20260526-040000"
    assert latest is not None
    assert latest.model_name == "unit-model"
    assert latest.market_count == 1
    assert latest.trade_count == 1
    assert trades == snapshot.trade_rows
    assert (tmp_path / "history" / "20260526-040000.json").exists()


def test_poll_snapshot_separates_all_predictions_from_trade_opportunities(tmp_path: Path) -> None:
    snapshot = ActiveMarketPoller(
        market_source=StubMarketSource(),
        scorer=StubScorer(),
        cache_store=MarketPollCacheStore(root=tmp_path),
        now=lambda: datetime(2026, 5, 26, 4, 0, tzinfo=UTC),
    ).run_once(model_name="unit-model")

    assert len(snapshot.prediction_rows) == 1
    assert len(snapshot.trade_rows) == 1
    assert snapshot.trade_rows[0].side == "NO"
    assert snapshot.trade_rows[0].edge == 0.06


def test_market_poller_cli_exposes_once_and_loop_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "once" in result.stdout
    assert "loop" in result.stdout


def test_active_market_source_falls_back_to_event_markets_when_search_has_no_markets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(
                200,
                json={
                    "current_page": [
                        {
                            "event_ticker": "KXEARNINGSMENTIONAAPL-26APR30",
                            "series_ticker": "KXEARNINGSMENTIONAAPL",
                            "event_title": "What will Apple say during their next earnings call?",
                        }
                    ]
                },
            )
        if request.url.path == "/trade-api/v2/markets":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONAAPL-26APR30-AI",
                            "event_ticker": "KXEARNINGSMENTIONAAPL-26APR30",
                            "series_ticker": "KXEARNINGSMENTIONAAPL",
                            "title": "What will Apple say during their next earnings call? - AI",
                            "custom_strike": {"Word": "AI"},
                            "yes_bid": 37,
                            "yes_ask": 40,
                            "volume": 123,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    source = KalshiActiveMarketSource(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_pages=1,
    )

    rows = source.list_active_markets()

    assert [row.market_ticker for row in rows] == ["KXEARNINGSMENTIONAAPL-26APR30-AI"]


def test_default_poll_cache_root_is_inside_kalorie2_artifacts_runtime() -> None:
    assert default_poll_cache_root().parts[-3:] == ("artifacts", "runtime", "workstation")
    assert default_poll_cache_root().parent.parent.parent.name == "kalorie2"


def test_preferred_model_name_uses_kalorie_v2_when_available(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "zzz-model")
    _write_bundle(tmp_path, "kalorie-v2")

    assert preferred_model_name(tmp_path, None) == "kalorie-v2"
    assert preferred_model_name(tmp_path, "explicit-model") == "explicit-model"
