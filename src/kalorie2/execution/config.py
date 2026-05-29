from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TradingMode = Literal["off", "dry_run", "live"]

# Operator must set this exact value to acknowledge that real money is at risk.
LIVE_CONFIRMATION_TOKEN = "I_UNDERSTAND_LIVE_TRADING"


@dataclass(frozen=True)
class EffectiveCaps:
    """Dollar caps for a single pass, derived from live portfolio value."""

    max_order_dollars: float
    max_event_exposure_dollars: float
    max_total_exposure_dollars: float
    daily_loss_stop_dollars: float


class LiveTradingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: TradingMode = "off"
    # Caps are a percentage of live portfolio value (cash + open positions) so
    # they compound as the account grows and stay safe when it is small.
    max_total_exposure_fraction: float = Field(default=0.80, ge=0.0, le=1.0)
    max_order_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    max_event_exposure_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    daily_loss_stop_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    # Optional absolute hard ceilings (dollars). None disables the ceiling; when
    # set, the effective cap is min(fraction * portfolio_value, ceiling).
    max_total_exposure_dollars: float | None = Field(default=None, ge=0.0)
    max_order_dollars: float | None = Field(default=None, ge=0.0)
    max_event_exposure_dollars: float | None = Field(default=None, ge=0.0)
    daily_loss_stop_dollars: float | None = Field(default=None, ge=0.0)
    max_orders_per_day: int = Field(default=20, ge=0)
    max_orders_per_contract_per_day: int = Field(default=1, ge=0)
    event_cutoff_seconds: int = Field(default=7200, ge=0)
    # epsilon_exec: live mid may drift this far from the scored mid and still
    # execute. price_swing_threshold is epsilon_halt: a durable halt boundary.
    execution_drift_tolerance: float = Field(default=0.03, ge=0.0, le=1.0)
    price_swing_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    # Skip (do not halt) when executable depth at the acceptable price exceeds
    # this multiple of the target size: a possible information event/spoof that
    # warrants waiting for the next model re-evaluation.
    depth_anomaly_multiple: float = Field(default=50.0, ge=0.0)
    fee_rate: float = Field(default=0.07, ge=0.0, le=1.0)
    # When a giant +EV block triggers a depth anomaly, request a re-score of that
    # event at most this often, bounded by a global daily cap (OpenAI cost guard).
    rescore_min_interval_seconds: int = Field(default=600, ge=0)
    rescore_max_per_day: int = Field(default=20, ge=0)
    live_confirmation: str | None = None

    def allows_real_orders(self) -> bool:
        return self.mode == "live" and self.live_confirmation == LIVE_CONFIRMATION_TOKEN

    def effective_caps(self, portfolio_value: float) -> EffectiveCaps:
        pv = max(portfolio_value, 0.0)
        return EffectiveCaps(
            max_order_dollars=_cap(self.max_order_fraction, pv, self.max_order_dollars),
            max_event_exposure_dollars=_cap(
                self.max_event_exposure_fraction, pv, self.max_event_exposure_dollars
            ),
            max_total_exposure_dollars=_cap(
                self.max_total_exposure_fraction, pv, self.max_total_exposure_dollars
            ),
            daily_loss_stop_dollars=_cap(
                self.daily_loss_stop_fraction, pv, self.daily_loss_stop_dollars
            ),
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> LiveTradingConfig:
        values: dict[str, object] = {}
        mode = env.get("KALORIE2_TRADING_MODE")
        if mode is not None:
            values["mode"] = mode.strip().lower()
        confirmation = env.get("KALORIE2_LIVE_CONFIRMATION")
        if confirmation is not None:
            values["live_confirmation"] = confirmation.strip()
        _set_float(
            values, "max_total_exposure_fraction", env, "KALORIE2_MAX_TOTAL_EXPOSURE_FRACTION"
        )
        _set_float(values, "max_order_fraction", env, "KALORIE2_MAX_ORDER_FRACTION")
        _set_float(
            values, "max_event_exposure_fraction", env, "KALORIE2_MAX_EVENT_EXPOSURE_FRACTION"
        )
        _set_float(values, "daily_loss_stop_fraction", env, "KALORIE2_DAILY_LOSS_STOP_FRACTION")
        _set_float(values, "max_total_exposure_dollars", env, "KALORIE2_MAX_TOTAL_EXPOSURE_DOLLARS")
        _set_float(values, "max_order_dollars", env, "KALORIE2_MAX_ORDER_DOLLARS")
        _set_float(values, "max_event_exposure_dollars", env, "KALORIE2_MAX_EVENT_EXPOSURE_DOLLARS")
        _set_float(values, "daily_loss_stop_dollars", env, "KALORIE2_DAILY_LOSS_STOP_DOLLARS")
        _set_int(values, "max_orders_per_day", env, "KALORIE2_MAX_ORDERS_PER_DAY")
        _set_int(values, "max_orders_per_contract_per_day", env, "KALORIE2_MAX_ORDERS_PER_CONTRACT")
        _set_int(values, "event_cutoff_seconds", env, "KALORIE2_EVENT_CUTOFF_SECONDS")
        _set_float(values, "execution_drift_tolerance", env, "KALORIE2_EXECUTION_DRIFT_TOLERANCE")
        _set_float(values, "price_swing_threshold", env, "KALORIE2_PRICE_SWING_THRESHOLD")
        _set_float(values, "depth_anomaly_multiple", env, "KALORIE2_DEPTH_ANOMALY_MULTIPLE")
        _set_float(values, "fee_rate", env, "KALORIE2_FEE_RATE")
        _set_int(
            values,
            "rescore_min_interval_seconds",
            env,
            "KALORIE2_RESCORE_MIN_INTERVAL_SECONDS",
        )
        _set_int(values, "rescore_max_per_day", env, "KALORIE2_RESCORE_MAX_PER_DAY")
        return cls(**values)


def _cap(fraction: float, portfolio_value: float, ceiling: float | None) -> float:
    if fraction <= 0.0:
        return round(ceiling, 4) if ceiling is not None else 0.0
    cap = fraction * portfolio_value
    if ceiling is not None:
        cap = min(cap, ceiling)
    return round(cap, 4)


def _set_float(
    values: dict[str, object],
    key: str,
    env: Mapping[str, str],
    env_key: str,
) -> None:
    raw = env.get(env_key)
    if raw is None or not raw.strip():
        return
    values[key] = float(raw)


def _set_int(
    values: dict[str, object],
    key: str,
    env: Mapping[str, str],
    env_key: str,
) -> None:
    raw = env.get(env_key)
    if raw is None or not raw.strip():
        return
    values[key] = int(raw)
