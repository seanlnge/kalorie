from datetime import UTC, datetime
from pathlib import Path

from kalorie2.execution.rescore import Rescorer
from kalorie2.market_poller import (
    ActiveMarketRow,
    MarketPollCacheStore,
    PollPredictionRow,
)

NOW = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)


def _market(
    event: str, ticker: str, *, yes_bid: float = 0.42, yes_ask: float = 0.45
) -> ActiveMarketRow:
    return ActiveMarketRow(
        market_ticker=ticker,
        event_ticker=event,
        series_ticker="KXEARNINGSMENTION",
        event_title="Acme Q1",
        market_title=ticker,
        target_phrase="AI",
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_mid=(yes_bid + yes_ask) / 2.0,
        volume=100,
    )


class FakeMarketSource:
    def __init__(self, markets: list[ActiveMarketRow]) -> None:
        self._markets = markets

    def list_active_markets(self) -> list[ActiveMarketRow]:
        return list(self._markets)


class FakeScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def score_active_markets(
        self, markets: list[ActiveMarketRow], *, model_name: str
    ) -> list[PollPredictionRow]:
        self.calls.append((tuple(m.market_ticker for m in markets), model_name))
        return [
            PollPredictionRow(
                market_ticker=m.market_ticker,
                event_ticker=m.event_ticker,
                target_phrase=m.target_phrase,
                model_name=model_name,
                model_probability=0.31,
                market_probability=0.435,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                residual_delta=-0.12,
                side="NO",
                edge=0.06,
                cost=0.58,
                volume=m.volume,
            )
            for m in markets
        ]


def _rescorer(markets: list[ActiveMarketRow], root: Path) -> tuple[Rescorer, FakeScorer]:
    scorer = FakeScorer()
    rescorer = Rescorer(
        market_source=FakeMarketSource(markets),
        scorer=scorer,
        cache_store=MarketPollCacheStore(root=root),
        now=lambda: NOW,
    )
    return rescorer, scorer


def test_rescore_all_scores_every_market_and_writes_cache(tmp_path: Path) -> None:
    markets = [_market("EVT-A", "EVT-A-AI"), _market("EVT-B", "EVT-B-AI")]
    rescorer, scorer = _rescorer(markets, tmp_path)
    store = MarketPollCacheStore(root=tmp_path)

    snapshot = rescorer.rescore_all(model_name="kalorie-v2")

    assert snapshot.market_count == 2
    assert scorer.calls[0][0] == ("EVT-A-AI", "EVT-B-AI")
    written = store.read_latest_snapshot()
    assert written is not None
    assert {row.market_ticker for row in written.prediction_rows} == {"EVT-A-AI", "EVT-B-AI"}


def test_rescore_event_only_scores_that_event_and_merges(tmp_path: Path) -> None:
    markets = [_market("EVT-A", "EVT-A-AI"), _market("EVT-B", "EVT-B-AI")]
    rescorer, _ = _rescorer(markets, tmp_path)
    store = MarketPollCacheStore(root=tmp_path)
    rescorer.rescore_all(model_name="kalorie-v2")

    # Now only EVT-B is re-scored; EVT-A rows must be preserved.
    rescorer2, scorer2 = _rescorer(
        [_market("EVT-A", "EVT-A-AI"), _market("EVT-B", "EVT-B-AI", yes_bid=0.50, yes_ask=0.52)],
        tmp_path,
    )
    snapshot = rescorer2.rescore_event(model_name="kalorie-v2", event_ticker="EVT-B")

    assert snapshot is not None
    # Only EVT-B markets were sent to the scorer.
    assert scorer2.calls == [(("EVT-B-AI",), "kalorie-v2")]
    written = store.read_latest_snapshot()
    assert written is not None
    by_ticker = {row.market_ticker: row for row in written.prediction_rows}
    assert set(by_ticker) == {"EVT-A-AI", "EVT-B-AI"}
    # The rescored EVT-B row reflects the new quote.
    assert by_ticker["EVT-B-AI"].yes_bid == 0.50


def test_rescore_event_returns_none_when_event_absent(tmp_path: Path) -> None:
    markets = [_market("EVT-A", "EVT-A-AI")]
    rescorer, scorer = _rescorer(markets, tmp_path)

    result = rescorer.rescore_event(model_name="kalorie-v2", event_ticker="EVT-MISSING")

    assert result is None
    assert scorer.calls == []


def test_rescore_event_drops_stale_other_model_rows(tmp_path: Path) -> None:
    markets = [_market("EVT-A", "EVT-A-AI"), _market("EVT-B", "EVT-B-AI")]
    rescorer, _ = _rescorer(markets, tmp_path)
    rescorer.rescore_all(model_name="old-model")
    store = MarketPollCacheStore(root=tmp_path)

    rescorer2, _ = _rescorer(markets, tmp_path)
    snapshot = rescorer2.rescore_event(model_name="new-model", event_ticker="EVT-B")

    assert snapshot is not None
    written = store.read_latest_snapshot()
    assert written is not None
    # EVT-A rows came from old-model, so they are dropped (only EVT-B remains).
    assert {row.market_ticker for row in written.prediction_rows} == {"EVT-B-AI"}
    assert written.model_name == "new-model"
