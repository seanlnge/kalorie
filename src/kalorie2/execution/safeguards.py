from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kalorie2.execution.config import LiveTradingConfig

SafeguardAction = Literal["APPROVE", "REJECT", "HALT_CONTRACT"]
OrderSide = Literal["YES", "NO"]

MIN_LIMIT_PRICE = 0.01
MAX_LIMIT_PRICE = 0.99


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TradeSignal(_Base):
    market_ticker: str
    event_ticker: str
    side: str
    signal_yes_bid: float
    signal_yes_ask: float
    recommended_fraction: float
    passes_risk_filter: bool

    @property
    def signal_mid(self) -> float:
        return (self.signal_yes_bid + self.signal_yes_ask) / 2.0


class FreshQuote(_Base):
    yes_bid: float = Field(ge=0.0, le=1.0)
    yes_ask: float = Field(ge=0.0, le=1.0)

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2.0


class SafeguardContext(_Base):
    now: datetime
    event_start: datetime | None
    account_available: bool
    previous_observed_mid: float | None
    daily_orders: int
    daily_orders_for_contract: int
    daily_loss: float
    # Portfolio-scaled stop, computed per pass from live portfolio value.
    daily_loss_stop: float
    is_halted: bool
    kill_switch_active: bool


class SafeguardDecision(_Base):
    action: SafeguardAction
    reason: str
    detail: str = ""
    side: str = "NONE"


def evaluate_signal(
    *,
    signal: TradeSignal,
    quote: FreshQuote,
    context: SafeguardContext,
    config: LiveTradingConfig,
) -> SafeguardDecision:
    """Pure, fail-closed pre-trade gate (no sizing).

    Returns APPROVE only when every safeguard passes; sizing against orderbook
    depth happens downstream in the fill engine. REJECT skips this signal for
    this round; HALT_CONTRACT additionally marks the contract for a durable stop
    until an operator clears it.
    """

    if context.kill_switch_active:
        return _reject("kill_switch", "Kill switch is active; no orders permitted")
    if context.is_halted:
        return _reject("contract_halted", "Contract is halted from a prior price swing")
    if config.mode == "off":
        return _reject("trading_disabled", "Trading mode is off")
    if not signal.passes_risk_filter:
        return _reject("risk_filter", "Signal did not pass the risk preset filter")
    if signal.side not in ("YES", "NO"):
        return _reject("invalid_side", f"Signal side is not tradable: {signal.side}")
    if not context.account_available:
        return _reject("paper_account", "No authenticated Kalshi account available")
    if context.event_start is None:
        return _reject("missing_event_time", "Event start time is unknown")

    seconds_to_event = (context.event_start - context.now).total_seconds()
    if seconds_to_event <= config.event_cutoff_seconds:
        return _reject(
            "event_cutoff",
            f"Event starts in {seconds_to_event:.0f}s, within the "
            f"{config.event_cutoff_seconds}s cutoff",
        )

    fresh_mid = quote.yes_mid
    if context.previous_observed_mid is not None:
        swing = abs(fresh_mid - context.previous_observed_mid)
        if swing >= config.price_swing_threshold:
            return SafeguardDecision(
                action="HALT_CONTRACT",
                reason="price_swing",
                detail=(
                    f"yes_mid moved {swing:.3f} from {context.previous_observed_mid:.3f} "
                    f"to {fresh_mid:.3f}"
                ),
                side=signal.side,
            )

    # Model-staleness tiers: the scored probability is conditioned on the mid the
    # model saw. Small drift -> still valid (execute). Moderate drift -> skip and
    # wait for the next re-evaluation. Large drift -> durable halt.
    signal_drift = abs(fresh_mid - signal.signal_mid)
    if signal_drift >= config.price_swing_threshold:
        return SafeguardDecision(
            action="HALT_CONTRACT",
            reason="stale_model_halt",
            detail=(
                f"Fresh yes_mid {fresh_mid:.3f} drifted {signal_drift:.3f} from "
                f"scored mid {signal.signal_mid:.3f}"
            ),
            side=signal.side,
        )
    if signal_drift >= config.execution_drift_tolerance:
        return _reject(
            "stale_model_skip",
            f"Fresh yes_mid {fresh_mid:.3f} drifted {signal_drift:.3f} from "
            f"scored mid {signal.signal_mid:.3f}; awaiting re-evaluation",
        )

    if context.daily_loss >= context.daily_loss_stop:
        return _reject(
            "daily_loss_stop",
            f"Daily loss {context.daily_loss:.2f} reached stop "
            f"{context.daily_loss_stop:.2f}",
        )
    if context.daily_orders >= config.max_orders_per_day:
        return _reject(
            "max_orders_per_day",
            f"Daily order count {context.daily_orders} reached cap "
            f"{config.max_orders_per_day}",
        )
    if context.daily_orders_for_contract >= config.max_orders_per_contract_per_day:
        return _reject(
            "max_orders_per_contract",
            f"Contract already traded {context.daily_orders_for_contract} times today",
        )

    return SafeguardDecision(
        action="APPROVE",
        reason="ok",
        detail="Gate passed; sizing against orderbook depth",
        side=signal.side,
    )


def seconds_until_event(now: datetime, event_start: datetime | None) -> float | None:
    if event_start is None:
        return None
    return (event_start - now).total_seconds()


def within_cutoff(
    now: datetime,
    event_start: datetime | None,
    *,
    cutoff_seconds: int,
) -> bool:
    """True when the event is missing a start time or starts within the cutoff."""

    if event_start is None:
        return True
    return (event_start - now) <= timedelta(seconds=cutoff_seconds)


def _reject(reason: str, detail: str) -> SafeguardDecision:
    return SafeguardDecision(action="REJECT", reason=reason, detail=detail)


def _clip_price(value: float) -> float:
    return min(MAX_LIMIT_PRICE, max(MIN_LIMIT_PRICE, value))
