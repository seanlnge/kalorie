from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from kalorie2.execution.config import EffectiveCaps, LiveTradingConfig
from kalorie2.execution.safeguards import (
    FreshQuote,
    SafeguardContext,
    TradeSignal,
    evaluate_signal,
)
from kalorie2.execution.sizing import FillPlan, plan_fill
from kalorie2.execution.state import (
    ExecutionStateStore,
    date_key,
    deterministic_client_order_id,
)
from kalorie2.kalshi_account import (
    AccountSummary,
    OpenPositionsSummary,
    build_account_summary,
    build_open_positions_summary,
)
from kalorie2.market_poller import MarketPollSnapshot, PollPredictionRow
from kalorie2.risk_presets import RiskPreset, apply_risk_preset_to_market

# Pull the live portfolio (cash + open positions) from Kalshi at most this often
# during a long-running loop. Start/restart always refetch because a fresh trader
# instance starts with an empty cache.
PORTFOLIO_REFRESH_SECONDS = 3600


class ExecutionClient(Protocol):
    def get_market_quote(self, ticker: str) -> Any: ...
    def get_orderbook(self, ticker: str) -> Any: ...
    def get_balance(self) -> dict[str, Any]: ...
    def list_positions(self) -> dict[str, Any]: ...
    def list_resting_orders(self, *, ticker: str | None = None) -> list[dict[str, Any]]: ...
    def submit_limit_order(
        self,
        *,
        ticker: str,
        action: str,
        side: str,
        limit_price_cents: int,
        count: int,
        client_order_id: str,
    ) -> str: ...
    def cancel_order(self, order_id: str) -> bool: ...


class SignalSource(Protocol):
    def read_latest_trades(self) -> list[PollPredictionRow]: ...
    def read_latest_snapshot(self) -> MarketPollSnapshot | None: ...


@dataclass(frozen=True)
class _BudgetInputs:
    # Compounding base: cash + market value of open positions.
    portfolio_value: float
    free_cash: float | None
    caps: EffectiveCaps
    total_exposure: float
    event_exposure: float


@dataclass
class _PortfolioState:
    """Live portfolio snapshot, refreshed on start/restart and hourly.

    ``running_*_exposure`` are seeded from Kalshi positions on each refresh and
    then incremented in-process as orders are placed, so exposure caps stay
    correct between hourly refetches without re-reading positions every pass.
    """

    account: AccountSummary
    positions: OpenPositionsSummary
    portfolio_value: float
    running_total_exposure: float
    running_event_exposure: dict[str, float]
    fetched_at: datetime
    fresh: bool


class SignalOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_ticker: str
    event_ticker: str
    action: str
    reason: str
    detail: str = ""
    side: str = "NONE"
    order_contracts: int = 0
    limit_price: float = 0.0
    order_dollars: float = 0.0
    order_id: str | None = None


class TraderRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    poll_id: str | None
    kill_switch_active: bool
    evaluated: int = 0
    submitted: int = 0
    dry_run_approved: int = 0
    rejected: int = 0
    halted: int = 0
    failed: int = 0
    duplicates: int = 0
    outcomes: list[SignalOutcome] = []


class LiveTrader:
    """Single autonomous trading pass.

    Re-derives side and size from the configured risk preset against a fresh
    quote (the canonical gate), runs every safeguard, and only then places a
    real order when the config explicitly allows live trading. All paths emit
    audit records; dry-run and preview never POST orders.
    """

    def __init__(
        self,
        *,
        config: LiveTradingConfig,
        client: ExecutionClient,
        state: ExecutionStateStore,
        risk_preset: RiskPreset,
        signal_source: SignalSource,
        now: Callable[[], datetime] | None = None,
        portfolio_refresh_seconds: int = PORTFOLIO_REFRESH_SECONDS,
        model_name: str | None = None,
        rescore_event: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._state = state
        self._risk_preset = risk_preset
        self._signal_source = signal_source
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._portfolio_refresh_seconds = max(portfolio_refresh_seconds, 0)
        self._portfolio: _PortfolioState | None = None
        # The model this bot is committed to (the running spec). When set, signals
        # scored by any other model are skipped (fail-closed) until a re-score lands.
        self._model_name = model_name
        # Optional callback to request a fresh per-event score when a giant block
        # appears. Bound by the config debounce + ExecutionStateStore daily cap.
        self._rescore_event = rescore_event

    @property
    def config(self) -> LiveTradingConfig:
        return self._config

    def run_once(self) -> TraderRunSummary:
        now = self._now()
        snapshot = self._signal_source.read_latest_snapshot()
        poll_id = snapshot.poll_id if snapshot else f"{now:%Y%m%d-%H%M%S}"

        if self._state.kill_switch_active():
            self._cancel_resting_orders(ticker=None)
            self._state.record_audit(
                {"event": "kill_switch", "mode": self._config.mode, "poll_id": poll_id}
            )
            return TraderRunSummary(
                mode=self._config.mode,
                poll_id=poll_id,
                kill_switch_active=True,
            )

        rows = self._signal_source.read_latest_trades()
        portfolio = self._refresh_portfolio_if_due(now=now, poll_id=poll_id)
        account_summary = portfolio.account
        caps = self._config.effective_caps(portfolio.portfolio_value)
        day = date_key(now)

        summary = TraderRunSummary(
            mode=self._config.mode,
            poll_id=poll_id,
            kill_switch_active=False,
        )

        for row in rows:
            if self._model_name is not None and row.model_name != self._model_name:
                outcome = self._reject_outcome(
                    {
                        "market_ticker": row.market_ticker,
                        "event_ticker": row.event_ticker,
                        "side": "NONE",
                    },
                    "model_mismatch",
                    (
                        f"Signal scored by {row.model_name}; awaiting re-score for "
                        f"running model {self._model_name}"
                    ),
                    poll_id,
                )
                summary.outcomes.append(outcome)
                summary.evaluated += 1
                self._tally(summary, outcome.action)
                continue

            quote = self._client.get_market_quote(row.market_ticker)
            fresh = FreshQuote(yes_bid=quote.yes_bid, yes_ask=quote.yes_ask)
            previous_mid = self._state.observed_mid(row.market_ticker)

            risk_decision = apply_risk_preset_to_market(
                preset=self._risk_preset,
                model_probability=row.model_probability,
                yes_bid=fresh.yes_bid,
                yes_ask=fresh.yes_ask,
            )
            signal = TradeSignal(
                market_ticker=row.market_ticker,
                event_ticker=row.event_ticker,
                side=risk_decision.side,
                signal_yes_bid=row.yes_bid,
                signal_yes_ask=row.yes_ask,
                recommended_fraction=risk_decision.recommended_fraction,
                passes_risk_filter=risk_decision.passes_filter,
            )
            context = SafeguardContext(
                now=now,
                event_start=_parse_event_start(row.event_datetime),
                account_available=account_summary.available,
                previous_observed_mid=previous_mid,
                daily_orders=self._state.daily_orders(day),
                daily_orders_for_contract=self._state.daily_orders_for_contract(
                    day, row.market_ticker
                ),
                daily_loss=self._state.daily_loss(day),
                daily_loss_stop=caps.daily_loss_stop_dollars,
                is_halted=self._state.is_halted(row.market_ticker),
                kill_switch_active=False,
            )

            decision = evaluate_signal(
                signal=signal,
                quote=fresh,
                context=context,
                config=self._config,
            )
            self._state.set_observed_mid(row.market_ticker, fresh.yes_mid)

            budget = _BudgetInputs(
                portfolio_value=portfolio.portfolio_value,
                free_cash=account_summary.free_cash,
                caps=caps,
                total_exposure=portfolio.running_total_exposure,
                event_exposure=portfolio.running_event_exposure.get(row.event_ticker, 0.0),
            )
            outcome = self._handle_decision(
                row=row,
                signal=signal,
                decision=decision,
                budget=budget,
                poll_id=poll_id,
                day=day,
                now=now,
            )
            summary.outcomes.append(outcome)
            summary.evaluated += 1
            self._tally(summary, outcome.action)

            if outcome.action == "SUBMITTED" or outcome.action == "DRY_RUN":
                portfolio.running_total_exposure += outcome.order_dollars
                portfolio.running_event_exposure[row.event_ticker] = (
                    portfolio.running_event_exposure.get(row.event_ticker, 0.0)
                    + outcome.order_dollars
                )

        return summary

    def _refresh_portfolio_if_due(self, *, now: datetime, poll_id: str) -> _PortfolioState:
        cached = self._portfolio
        if cached is not None and cached.fresh:
            elapsed = (now - cached.fetched_at).total_seconds()
            if elapsed < self._portfolio_refresh_seconds:
                return cached

        account = build_account_summary(
            balance_payload=_safe_call(self._client.get_balance),
            positions_payload=None,
        )
        positions = build_open_positions_summary(_safe_call(self._client.list_positions))
        portfolio_value = _portfolio_value(account, positions)
        state = _PortfolioState(
            account=account,
            positions=positions,
            portfolio_value=portfolio_value,
            running_total_exposure=positions.total_exposure or 0.0,
            running_event_exposure=_event_exposure_map(positions.positions),
            fetched_at=now,
            # Only treat the snapshot as fresh (and therefore cacheable for the
            # hour) once we actually have an authenticated account; otherwise keep
            # retrying every pass so a transient balance failure self-heals.
            fresh=account.available,
        )
        self._portfolio = state
        self._audit(
            "portfolio_refresh",
            {
                "market_ticker": "",
                "event_ticker": "",
                "reason": "portfolio_refresh",
                "detail": (
                    f"portfolio_value={portfolio_value:.2f} "
                    f"free_cash={_fmt(account.free_cash)} "
                    f"positions_value={_fmt(positions.total_market_value)} "
                    f"open_positions={positions.open_position_count}"
                ),
            },
            poll_id,
        )
        return state

    def _handle_decision(
        self,
        *,
        row: PollPredictionRow,
        signal: TradeSignal,
        decision: Any,
        budget: _BudgetInputs,
        poll_id: str,
        day: str,
        now: datetime,
    ) -> SignalOutcome:
        base = {
            "market_ticker": row.market_ticker,
            "event_ticker": row.event_ticker,
            "reason": decision.reason,
            "detail": decision.detail,
            "side": decision.side,
        }

        if decision.action == "HALT_CONTRACT":
            self._state.halt_contract(row.market_ticker, reason=decision.reason, now=now)
            self._cancel_resting_orders(ticker=row.market_ticker)
            self._audit("halted", base, poll_id)
            return SignalOutcome(action="HALT_CONTRACT", **base)

        if decision.action == "REJECT":
            if decision.reason == "event_cutoff":
                self._cancel_resting_orders(ticker=row.market_ticker)
            self._audit("rejected", base, poll_id)
            return SignalOutcome(action="REJECT", **base)

        return self._size_and_submit(
            row=row,
            signal=signal,
            side=decision.side,
            budget=budget,
            poll_id=poll_id,
            day=day,
            now=now,
        )

    def _size_and_submit(
        self,
        *,
        row: PollPredictionRow,
        signal: TradeSignal,
        side: str,
        budget: _BudgetInputs,
        poll_id: str,
        day: str,
        now: datetime,
    ) -> SignalOutcome:
        base = {
            "market_ticker": row.market_ticker,
            "event_ticker": row.event_ticker,
            "side": side,
        }

        try:
            book = self._client.get_orderbook(row.market_ticker)
        except Exception as exc:  # noqa: BLE001 - no fresh depth -> do not trade
            return self._reject_outcome(
                base, "orderbook_unavailable", f"Could not fetch orderbook: {exc}", poll_id
            )

        ask_levels: list[tuple[float, int]] = book.ask_levels(side)
        if not ask_levels:
            return self._reject_outcome(
                base, "no_liquidity", "No resting depth on the buy side", poll_id
            )

        caps = budget.caps
        anomaly = self._depth_anomaly(
            side=side,
            signal=signal,
            ask_levels=ask_levels,
            max_order_dollars=caps.max_order_dollars,
        )
        if anomaly is not None:
            # Don't trade a possible information event/spoof; instead ask for a
            # fresh per-event score (debounced) so the next pass can act on it.
            self._maybe_request_rescore(event_ticker=row.event_ticker, now=now, poll_id=poll_id)
            return self._reject_outcome(base, "depth_anomaly_skip", anomaly, poll_id)

        win_probability = row.model_probability if side == "YES" else 1.0 - row.model_probability
        # Size off live portfolio value (cash + positions) so orders compound as
        # the account grows and scale down when it is small.
        target_dollars = max(signal.recommended_fraction, 0.0) * budget.portfolio_value
        event_room = max(caps.max_event_exposure_dollars - budget.event_exposure, 0.0)
        total_room = max(caps.max_total_exposure_dollars - budget.total_exposure, 0.0)
        max_budget = min(caps.max_order_dollars, event_room, total_room)
        if budget.free_cash is not None:
            max_budget = min(max_budget, budget.free_cash)

        plan = plan_fill(
            win_probability=win_probability,
            ask_levels=ask_levels,
            target_dollars=target_dollars,
            max_budget_dollars=max_budget,
            min_margin=self._risk_preset.min_margin,
            fee_rate=self._config.fee_rate,
        )
        if plan.contracts < 1:
            return self._reject_outcome(
                base,
                "insufficient_size",
                f"No +EV contracts affordable within budget {max_budget:.2f}",
                poll_id,
            )

        return self._submit_plan(row=row, side=side, plan=plan, base=base, poll_id=poll_id, day=day)

    def _submit_plan(
        self,
        *,
        row: PollPredictionRow,
        side: str,
        plan: FillPlan,
        base: dict[str, Any],
        poll_id: str,
        day: str,
    ) -> SignalOutcome:
        limit_price_cents = int(round(plan.limit_price * 100))
        sized = {
            **base,
            "detail": (
                f"{plan.contracts} {side} @<= {plan.limit_price:.2f} "
                f"(blended {plan.blended_price:.3f}, EV {plan.expected_value_dollars:.2f})"
            ),
            "reason": "ok",
            "order_contracts": plan.contracts,
            "limit_price": plan.limit_price,
            "order_dollars": plan.gross_cost_dollars,
        }
        client_order_id = deterministic_client_order_id(
            model_name=row.model_name,
            poll_id=poll_id,
            market_ticker=row.market_ticker,
            side=side,
            limit_price_cents=limit_price_cents,
            count=plan.contracts,
            date_key=day,
        )

        if self._state.has_seen_client_order_id(client_order_id):
            self._audit("duplicate_skipped", {**sized, "client_order_id": client_order_id}, poll_id)
            return SignalOutcome(action="DUPLICATE", **sized)

        if not self._config.allows_real_orders():
            self._audit("dry_run_approved", {**sized, "client_order_id": client_order_id}, poll_id)
            return SignalOutcome(action="DRY_RUN", **sized)

        # Clear any stale unfilled remainder for this contract before re-submitting.
        self._cancel_resting_orders(ticker=row.market_ticker)

        try:
            order_id = self._client.submit_limit_order(
                ticker=row.market_ticker,
                action="buy",
                side=side.lower(),
                limit_price_cents=limit_price_cents,
                count=plan.contracts,
                client_order_id=client_order_id,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed and reconcile next loop
            self._audit(
                "submit_failed",
                {**sized, "client_order_id": client_order_id, "error": str(exc)},
                poll_id,
            )
            return SignalOutcome(action="FAILED", **sized)

        self._state.record_client_order_id(client_order_id)
        self._state.record_order(day, row.market_ticker)
        self._audit(
            "submitted",
            {**sized, "client_order_id": client_order_id, "order_id": order_id},
            poll_id,
        )
        return SignalOutcome(action="SUBMITTED", order_id=order_id, **sized)

    def _depth_anomaly(
        self,
        *,
        side: str,
        signal: TradeSignal,
        ask_levels: list[tuple[float, int]],
        max_order_dollars: float,
    ) -> str | None:
        """Skip (do not halt) when the top of book is both anomalously deep and
        cheaper than the price the model scored against: a likely information
        event/spoof that should wait for the next model re-evaluation."""

        if self._config.depth_anomaly_multiple <= 0.0:
            return None
        best_price, best_qty = ask_levels[0]
        scored_ask = signal.signal_yes_ask if side == "YES" else 1.0 - signal.signal_yes_bid
        improved = best_price <= scored_ask - 0.01
        target_contracts = 1
        if best_price > 0.0:
            target_contracts = max(1, int(max_order_dollars / best_price))
        if improved and best_qty > self._config.depth_anomaly_multiple * target_contracts:
            return (
                f"Top-of-book depth {best_qty} at {best_price:.2f} improved past scored "
                f"{scored_ask:.2f}; awaiting re-evaluation"
            )
        return None

    def _maybe_request_rescore(self, *, event_ticker: str, now: datetime, poll_id: str) -> None:
        if self._rescore_event is None:
            return
        if not self._state.should_rescore_event(
            event_ticker,
            now=now,
            min_interval_seconds=self._config.rescore_min_interval_seconds,
            max_per_day=self._config.rescore_max_per_day,
        ):
            return
        # Record before invoking so a slow/failing score still consumes the
        # debounce window and never hammers the scorer.
        self._state.record_rescore(event_ticker, now=now)
        self._audit(
            "rescore_requested",
            {
                "market_ticker": "",
                "event_ticker": event_ticker,
                "reason": "giant_block_rescore",
                "detail": f"Giant +EV block on {event_ticker}; re-scoring event",
            },
            poll_id,
        )
        try:
            self._rescore_event(event_ticker)
        except Exception as exc:  # noqa: BLE001 - re-score is best-effort, never fatal
            self._audit(
                "rescore_failed",
                {
                    "market_ticker": "",
                    "event_ticker": event_ticker,
                    "reason": "rescore_failed",
                    "detail": str(exc),
                },
                poll_id,
            )

    def _reject_outcome(
        self, base: dict[str, Any], reason: str, detail: str, poll_id: str
    ) -> SignalOutcome:
        payload = {**base, "reason": reason, "detail": detail}
        self._audit("rejected", payload, poll_id)
        return SignalOutcome(action="REJECT", **payload)

    def _cancel_resting_orders(self, *, ticker: str | None) -> None:
        try:
            orders = self._client.list_resting_orders(ticker=ticker)
        except Exception:  # noqa: BLE001
            return
        for order in orders:
            order_id = order.get("order_id") or order.get("id")
            if order_id is None:
                continue
            try:
                self._client.cancel_order(str(order_id))
            except Exception:  # noqa: BLE001
                continue

    def _audit(self, event: str, base: dict[str, Any], poll_id: str) -> None:
        self._state.record_audit(
            {"event": event, "mode": self._config.mode, "poll_id": poll_id, **base}
        )

    @staticmethod
    def _tally(summary: TraderRunSummary, action: str) -> None:
        if action == "SUBMITTED":
            summary.submitted += 1
        elif action == "DRY_RUN":
            summary.dry_run_approved += 1
        elif action == "REJECT":
            summary.rejected += 1
        elif action == "HALT_CONTRACT":
            summary.halted += 1
        elif action == "FAILED":
            summary.failed += 1
        elif action == "DUPLICATE":
            summary.duplicates += 1


def _portfolio_value(account: AccountSummary, positions: OpenPositionsSummary) -> float:
    """Compounding base: free cash + market value of open positions.

    Falls back to the paper bankroll when no authenticated account is available,
    so sizing math stays well-defined (the gate blocks orders separately).
    """

    if account.free_cash is None:
        return account.bankroll
    return account.free_cash + (positions.total_market_value or 0.0)


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _event_exposure_map(positions: list[Any]) -> dict[str, float]:
    """Approximate per-event exposure by matching market tickers to event prefixes.

    Earnings-mention markets are tickered as ``<event_ticker>-<word>`` so a
    market belongs to the event whose ticker prefixes it.
    """

    exposures: dict[str, float] = {}
    for position in positions:
        exposure = getattr(position, "exposure", None)
        market_ticker = getattr(position, "market_ticker", "")
        if exposure is None or not market_ticker:
            continue
        event_ticker = market_ticker.rsplit("-", 1)[0]
        exposures[event_ticker] = exposures.get(event_ticker, 0.0) + abs(float(exposure))
    return exposures


def _parse_event_start(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _safe_call(func: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return func()
    except Exception:  # noqa: BLE001 - missing/failed account reads fail closed downstream
        return None
