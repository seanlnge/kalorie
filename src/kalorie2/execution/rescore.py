from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from kalorie2.market_poller import (
    ActiveMarketPoller,
    ActiveMarketSource,
    MarketPollCacheStore,
    MarketPollSnapshot,
    MarketScorer,
    PollPredictionRow,
)


class Rescorer:
    """Scores active markets with a model and writes the live trader's poll cache.

    The live trader only consumes pre-scored rows from ``MarketPollCacheStore``;
    it never runs a model itself. This service is the bridge that lets the web app
    trigger a fresh score on demand:

    * ``rescore_all`` re-scores every active market (used synchronously on
      start/restart so the bot trades only on signals from its committed model).
    * ``rescore_event`` re-scores a single event and merges the result back into
      the latest snapshot (used when a giant +EV order block appears).
    """

    def __init__(
        self,
        *,
        market_source: ActiveMarketSource,
        scorer: MarketScorer,
        cache_store: MarketPollCacheStore,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._market_source = market_source
        self._scorer = scorer
        self._cache_store = cache_store
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._poller = ActiveMarketPoller(
            market_source=market_source,
            scorer=scorer,
            cache_store=cache_store,
            now=self._now,
        )

    def rescore_all(self, *, model_name: str) -> MarketPollSnapshot:
        return self._poller.run_once(model_name=model_name)

    def rescore_event(
        self, *, model_name: str, event_ticker: str
    ) -> MarketPollSnapshot | None:
        started_at = self._now()
        markets = self._market_source.list_active_markets()
        event_markets = [m for m in markets if m.event_ticker == event_ticker]
        if not event_markets:
            return None

        rows = self._scorer.score_active_markets(event_markets, model_name=model_name)
        merged: list[PollPredictionRow] = list(rows)
        existing = self._cache_store.read_latest_snapshot()
        # Only preserve other events' rows when they came from the same model;
        # otherwise drop them so a stale, mixed-model snapshot can't leak through.
        if existing is not None and existing.model_name == model_name:
            merged.extend(
                row for row in existing.prediction_rows if row.event_ticker != event_ticker
            )

        completed_at = self._now()
        trade_rows = [row for row in merged if row.side in {"YES", "NO"}]
        snapshot = MarketPollSnapshot(
            poll_id=f"{started_at:%Y%m%d-%H%M%S}",
            model_name=model_name,
            started_at=started_at,
            completed_at=completed_at,
            market_count=len(merged),
            prediction_count=len(merged),
            trade_count=len(trade_rows),
            prediction_rows=merged,
            trade_rows=trade_rows,
        )
        self._cache_store.write_snapshot(snapshot)
        return snapshot
