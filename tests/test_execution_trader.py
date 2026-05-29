from datetime import UTC, datetime, timedelta
from pathlib import Path

from kalorie2.execution.client import MarketQuote, OrderbookDepth
from kalorie2.execution.config import LIVE_CONFIRMATION_TOKEN, LiveTradingConfig
from kalorie2.execution.state import ExecutionStateStore
from kalorie2.execution.trader import LiveTrader
from kalorie2.market_poller import MarketPollSnapshot, PollPredictionRow
from kalorie2.risk_presets import get_risk_preset

NOW = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
EVENT_TICKER = "KXEARNINGSMENTIONAAPL-26APR30"
MARKET_TICKER = "KXEARNINGSMENTIONAAPL-26APR30-AI"


def _row(**overrides: object) -> PollPredictionRow:
    base: dict[str, object] = {
        "market_ticker": MARKET_TICKER,
        "event_ticker": EVENT_TICKER,
        "event_datetime": (NOW + timedelta(hours=3)).isoformat(),
        "target_phrase": "AI",
        "model_name": "kalorie-v2",
        "model_probability": 0.31,
        "market_probability": 0.435,
        "yes_bid": 0.42,
        "yes_ask": 0.45,
        "residual_delta": -0.12,
        "side": "NO",
        "edge": 0.06,
        "cost": 0.58,
        "volume": 100,
    }
    base.update(overrides)
    return PollPredictionRow(**base)


class FakeSignalSource:
    def __init__(self, rows: list[PollPredictionRow]) -> None:
        self._rows = rows

    def read_latest_trades(self) -> list[PollPredictionRow]:
        return list(self._rows)

    def read_latest_snapshot(self) -> MarketPollSnapshot | None:
        return MarketPollSnapshot(
            poll_id="20260529-120000",
            model_name="kalorie-v2",
            started_at=NOW,
            completed_at=NOW,
            market_count=len(self._rows),
            prediction_count=len(self._rows),
            trade_count=len(self._rows),
            prediction_rows=self._rows,
            trade_rows=self._rows,
        )


class FakeClient:
    def __init__(
        self,
        *,
        quote: MarketQuote,
        balance: dict | None = None,
        positions: dict | None = None,
        resting: list[dict] | None = None,
        orderbook: OrderbookDepth | None = None,
    ) -> None:
        self._quote = quote
        self._balance = balance if balance is not None else {
            "balance": {"available_balance": 10000, "portfolio_value": 10000}
        }
        self._positions = positions if positions is not None else {"market_positions": []}
        self._resting = resting if resting is not None else []
        # Default ladder: NO asks (from yes bids) sit at 0.58 with ample depth.
        self._orderbook = orderbook or OrderbookDepth(
            market_ticker=MARKET_TICKER,
            yes_bids=[(42, 50)],
            no_bids=[(45, 50)],
        )
        self.submitted: list[dict] = []
        self.cancelled: list[str] = []
        self.balance_calls = 0
        self.position_calls = 0

    def get_market_quote(self, ticker: str) -> MarketQuote:
        return self._quote

    def get_orderbook(self, ticker: str) -> OrderbookDepth:
        return self._orderbook

    def get_balance(self) -> dict:
        self.balance_calls += 1
        return self._balance

    def list_positions(self) -> dict:
        self.position_calls += 1
        return self._positions

    def list_resting_orders(self, *, ticker: str | None = None) -> list[dict]:
        if ticker is None:
            return list(self._resting)
        return [order for order in self._resting if order.get("ticker") == ticker]

    def submit_limit_order(self, **kwargs: object) -> str:
        self.submitted.append(kwargs)
        return f"ord-{len(self.submitted)}"

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True


def _quote(yes_bid: float = 0.42, yes_ask: float = 0.45) -> MarketQuote:
    return MarketQuote(
        market_ticker=MARKET_TICKER,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=round(1 - yes_ask, 2),
        no_ask=round(1 - yes_bid, 2),
    )


def _trader(
    *,
    config: LiveTradingConfig,
    client: FakeClient,
    state: ExecutionStateStore,
    rows: list[PollPredictionRow],
) -> LiveTrader:
    return LiveTrader(
        config=config,
        client=client,
        state=state,
        risk_preset=get_risk_preset("balanced"),
        signal_source=FakeSignalSource(rows),
        now=lambda: NOW,
    )


def test_dry_run_approves_without_submitting_any_orders(tmp_path: Path) -> None:
    client = FakeClient(quote=_quote())
    state = ExecutionStateStore(root=tmp_path)
    trader = _trader(
        config=LiveTradingConfig(mode="dry_run"),
        client=client,
        state=state,
        rows=[_row()],
    )

    summary = trader.run_once()

    assert summary.dry_run_approved == 1
    assert summary.submitted == 0
    assert client.submitted == []
    last = state.last_audit()
    assert last["event"] == "dry_run_approved"
    assert last["order_contracts"] == 8


def test_live_mode_submits_and_then_dedupes_on_repeat(tmp_path: Path) -> None:
    client = FakeClient(quote=_quote())
    state = ExecutionStateStore(root=tmp_path)
    # Lift the per-contract daily cap so the dedupe (idempotent client_order_id)
    # guard is what blocks the repeat, not the separate daily-count safeguard.
    config = LiveTradingConfig(
        mode="live",
        live_confirmation=LIVE_CONFIRMATION_TOKEN,
        max_orders_per_contract_per_day=5,
    )

    first = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()
    assert first.submitted == 1
    assert len(client.submitted) == 1
    submitted = client.submitted[0]
    assert submitted["side"] == "no"
    assert submitted["limit_price_cents"] == 58
    assert submitted["count"] == 8

    second = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()
    assert second.submitted == 0
    assert second.duplicates == 1
    assert len(client.submitted) == 1


def test_kill_switch_cancels_resting_orders_and_skips_trading(tmp_path: Path) -> None:
    client = FakeClient(
        quote=_quote(),
        resting=[{"order_id": "r1", "ticker": MARKET_TICKER}],
    )
    state = ExecutionStateStore(root=tmp_path)
    state.activate_kill_switch(reason="test")
    config = LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN)

    summary = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()

    assert summary.kill_switch_active is True
    assert summary.submitted == 0
    assert client.submitted == []
    assert client.cancelled == ["r1"]


def test_event_cutoff_rejects_and_cancels_resting_orders(tmp_path: Path) -> None:
    client = FakeClient(
        quote=_quote(),
        resting=[{"order_id": "r2", "ticker": MARKET_TICKER}],
    )
    state = ExecutionStateStore(root=tmp_path)
    config = LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN)
    row = _row(event_datetime=(NOW + timedelta(hours=1)).isoformat())

    summary = _trader(config=config, client=client, state=state, rows=[row]).run_once()

    assert summary.submitted == 0
    assert summary.rejected == 1
    assert client.cancelled == ["r2"]
    assert summary.outcomes[0].reason == "event_cutoff"


def test_price_swing_halts_contract_and_persists_halt(tmp_path: Path) -> None:
    state = ExecutionStateStore(root=tmp_path)
    state.set_observed_mid(MARKET_TICKER, 0.30)
    client = FakeClient(
        quote=_quote(yes_bid=0.44, yes_ask=0.46),
        resting=[{"order_id": "r3", "ticker": MARKET_TICKER}],
    )
    config = LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN)

    summary = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()

    assert summary.halted == 1
    assert summary.submitted == 0
    assert client.cancelled == ["r3"]
    assert state.is_halted(MARKET_TICKER) is True

    follow_up = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()
    assert follow_up.outcomes[0].reason == "contract_halted"


def test_missing_event_datetime_blocks_execution(tmp_path: Path) -> None:
    client = FakeClient(quote=_quote())
    state = ExecutionStateStore(root=tmp_path)
    config = LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN)
    row = _row(event_datetime=None)

    summary = _trader(config=config, client=client, state=state, rows=[row]).run_once()

    assert summary.submitted == 0
    assert summary.rejected == 1
    assert client.submitted == []
    assert summary.outcomes[0].reason == "missing_event_time"


def test_depth_aware_fill_walks_multiple_price_levels(tmp_path: Path) -> None:
    # NO buy: win prob = 1 - 0.31 = 0.69. NO asks come from YES bids:
    # yes bid 42 -> NO ask 0.58 (qty 4); yes bid 40 -> NO ask 0.60 (qty 100).
    orderbook = OrderbookDepth(
        market_ticker=MARKET_TICKER,
        yes_bids=[(42, 4), (40, 100)],
        no_bids=[(45, 50)],
    )
    client = FakeClient(quote=_quote(), orderbook=orderbook)
    state = ExecutionStateStore(root=tmp_path)
    config = LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN)

    summary = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()

    assert summary.submitted == 1
    order = client.submitted[0]
    # 4 contracts available at 0.58, remainder filled at 0.60 up to the budget;
    # the limit is set to the worst accepted price.
    assert order["limit_price_cents"] == 60
    assert order["count"] >= 5


def test_depth_anomaly_at_better_price_skips_pending_reeval(tmp_path: Path) -> None:
    # A huge resting block appears at a price well better than the scored ask
    # (scored NO ask = 1 - 0.42 = 0.58). yes bid 50 -> NO ask 0.50 with 5000 qty.
    orderbook = OrderbookDepth(
        market_ticker=MARKET_TICKER,
        yes_bids=[(50, 5000)],
        no_bids=[(45, 50)],
    )
    client = FakeClient(quote=_quote(), orderbook=orderbook)
    state = ExecutionStateStore(root=tmp_path)
    config = LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN)

    summary = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()

    assert summary.submitted == 0
    assert summary.rejected == 1
    assert summary.outcomes[0].reason == "depth_anomaly_skip"
    assert state.is_halted(MARKET_TICKER) is False


def test_no_liquidity_skips_without_submitting(tmp_path: Path) -> None:
    orderbook = OrderbookDepth(market_ticker=MARKET_TICKER, yes_bids=[], no_bids=[])
    client = FakeClient(quote=_quote(), orderbook=orderbook)
    state = ExecutionStateStore(root=tmp_path)
    config = LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN)

    summary = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()

    assert summary.submitted == 0
    assert summary.outcomes[0].reason == "no_liquidity"


def test_paper_account_blocks_live_trading(tmp_path: Path) -> None:
    client = FakeClient(quote=_quote(), balance={})
    state = ExecutionStateStore(root=tmp_path)
    config = LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN)

    summary = _trader(config=config, client=client, state=state, rows=[_row()]).run_once()

    assert summary.submitted == 0
    assert summary.rejected == 1
    assert summary.outcomes[0].reason == "paper_account"


def test_sizes_off_total_portfolio_value_including_open_positions(tmp_path: Path) -> None:
    # Same $100 cash, but one account also holds $500 of open positions in an
    # unrelated event. Sizing compounds off cash + positions, so it buys more.
    base_client = FakeClient(quote=_quote())
    base_state = ExecutionStateStore(root=tmp_path / "a")
    base = _trader(
        config=LiveTradingConfig(mode="dry_run"),
        client=base_client,
        state=base_state,
        rows=[_row()],
    ).run_once()

    with_positions = FakeClient(
        quote=_quote(),
        positions={
            "market_positions": [
                {"ticker": "OTHEREVENT-X", "position": 10, "market_value": 50000}
            ]
        },
    )
    state = ExecutionStateStore(root=tmp_path / "b")
    grown = _trader(
        config=LiveTradingConfig(mode="dry_run"),
        client=with_positions,
        state=state,
        rows=[_row()],
    ).run_once()

    assert grown.outcomes[0].order_contracts > base.outcomes[0].order_contracts


def test_caps_and_sizing_scale_down_for_a_small_account(tmp_path: Path) -> None:
    # $30 account: percentage caps shrink with it and sizing stays tiny.
    client = FakeClient(
        quote=_quote(),
        balance={"balance": {"available_balance": 3000, "portfolio_value": 3000}},
    )
    state = ExecutionStateStore(root=tmp_path)
    summary = _trader(
        config=LiveTradingConfig(mode="dry_run"),
        client=client,
        state=state,
        rows=[_row()],
    ).run_once()

    outcome = summary.outcomes[0]
    assert outcome.action == "DRY_RUN"
    # max_order cap = 10% of $30 = $3, so the order must stay under it.
    assert outcome.order_dollars <= 3.0
    assert 0 < outcome.order_contracts < 8


def test_portfolio_fetched_from_kalshi_once_per_hour_then_refetched(tmp_path: Path) -> None:
    clock = {"now": NOW}
    client = FakeClient(quote=_quote())
    state = ExecutionStateStore(root=tmp_path)
    trader = LiveTrader(
        config=LiveTradingConfig(mode="dry_run"),
        client=client,
        state=state,
        risk_preset=get_risk_preset("balanced"),
        signal_source=FakeSignalSource([_row()]),
        now=lambda: clock["now"],
        portfolio_refresh_seconds=3600,
    )

    trader.run_once()
    clock["now"] = NOW + timedelta(seconds=20)
    trader.run_once()
    clock["now"] = NOW + timedelta(minutes=30)
    trader.run_once()
    # Start + two more passes inside the hour => one live portfolio fetch.
    assert client.balance_calls == 1
    assert client.position_calls == 1

    clock["now"] = NOW + timedelta(hours=1, seconds=1)
    trader.run_once()
    # Past the hour => refetched directly from Kalshi.
    assert client.balance_calls == 2
    assert client.position_calls == 2


def test_running_model_guard_skips_signals_from_other_models(tmp_path: Path) -> None:
    # The committed model is "new-model" but cached signals say "kalorie-v2".
    client = FakeClient(quote=_quote())
    state = ExecutionStateStore(root=tmp_path)
    trader = LiveTrader(
        config=LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN),
        client=client,
        state=state,
        risk_preset=get_risk_preset("balanced"),
        signal_source=FakeSignalSource([_row()]),
        now=lambda: NOW,
        model_name="new-model",
    )

    summary = trader.run_once()

    assert summary.submitted == 0
    assert summary.rejected == 1
    assert summary.outcomes[0].reason == "model_mismatch"
    assert client.submitted == []


def test_matching_model_name_allows_trading(tmp_path: Path) -> None:
    client = FakeClient(quote=_quote())
    state = ExecutionStateStore(root=tmp_path)
    trader = LiveTrader(
        config=LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN),
        client=client,
        state=state,
        risk_preset=get_risk_preset("balanced"),
        signal_source=FakeSignalSource([_row()]),
        now=lambda: NOW,
        model_name="kalorie-v2",
    )

    summary = trader.run_once()

    assert summary.submitted == 1


def test_giant_block_triggers_debounced_event_rescore(tmp_path: Path) -> None:
    orderbook = OrderbookDepth(
        market_ticker=MARKET_TICKER,
        yes_bids=[(50, 5000)],
        no_bids=[(45, 50)],
    )
    client = FakeClient(quote=_quote(), orderbook=orderbook)
    state = ExecutionStateStore(root=tmp_path)
    rescored: list[str] = []
    trader = LiveTrader(
        config=LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN),
        client=client,
        state=state,
        risk_preset=get_risk_preset("balanced"),
        signal_source=FakeSignalSource([_row()]),
        now=lambda: NOW,
        rescore_event=rescored.append,
    )

    first = trader.run_once()
    assert first.outcomes[0].reason == "depth_anomaly_skip"
    assert rescored == [EVENT_TICKER]

    # A second pass within the debounce window must NOT trigger another re-score.
    trader.run_once()
    assert rescored == [EVENT_TICKER]


def test_rescore_failure_is_non_fatal(tmp_path: Path) -> None:
    orderbook = OrderbookDepth(
        market_ticker=MARKET_TICKER,
        yes_bids=[(50, 5000)],
        no_bids=[(45, 50)],
    )
    client = FakeClient(quote=_quote(), orderbook=orderbook)
    state = ExecutionStateStore(root=tmp_path)

    def boom(_event: str) -> None:
        raise RuntimeError("scorer offline")

    trader = LiveTrader(
        config=LiveTradingConfig(mode="live", live_confirmation=LIVE_CONFIRMATION_TOKEN),
        client=client,
        state=state,
        risk_preset=get_risk_preset("balanced"),
        signal_source=FakeSignalSource([_row()]),
        now=lambda: NOW,
        rescore_event=boom,
    )

    summary = trader.run_once()
    assert summary.outcomes[0].reason == "depth_anomaly_skip"
    events = {entry["event"] for entry in state.read_audit()}
    assert "rescore_failed" in events


def test_paper_account_refetches_until_authenticated(tmp_path: Path) -> None:
    # A failed balance read must not be cached for an hour; the next pass retries.
    client = FakeClient(quote=_quote(), balance={})
    state = ExecutionStateStore(root=tmp_path)
    trader = LiveTrader(
        config=LiveTradingConfig(mode="dry_run"),
        client=client,
        state=state,
        risk_preset=get_risk_preset("balanced"),
        signal_source=FakeSignalSource([_row()]),
        now=lambda: NOW,
        portfolio_refresh_seconds=3600,
    )

    trader.run_once()
    trader.run_once()
    assert client.balance_calls == 2
