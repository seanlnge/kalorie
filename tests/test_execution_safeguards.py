from datetime import UTC, datetime, timedelta

from kalorie2.execution.config import (
    LIVE_CONFIRMATION_TOKEN,
    LiveTradingConfig,
)
from kalorie2.execution.safeguards import (
    FreshQuote,
    SafeguardContext,
    TradeSignal,
    evaluate_signal,
)

NOW = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)


def _config(**overrides: object) -> LiveTradingConfig:
    base: dict[str, object] = {"mode": "dry_run"}
    base.update(overrides)
    return LiveTradingConfig(**base)


def _signal(**overrides: object) -> TradeSignal:
    base: dict[str, object] = {
        "market_ticker": "KXEARNINGSMENTIONAAPL-26APR30-AI",
        "event_ticker": "KXEARNINGSMENTIONAAPL-26APR30",
        "side": "NO",
        "signal_yes_bid": 0.42,
        "signal_yes_ask": 0.45,
        "recommended_fraction": 0.05,
        "passes_risk_filter": True,
    }
    base.update(overrides)
    return TradeSignal(**base)


def _quote(yes_bid: float = 0.42, yes_ask: float = 0.45) -> FreshQuote:
    return FreshQuote(yes_bid=yes_bid, yes_ask=yes_ask)


def _context(**overrides: object) -> SafeguardContext:
    base: dict[str, object] = {
        "now": NOW,
        "event_start": NOW + timedelta(hours=3),
        "account_available": True,
        "previous_observed_mid": None,
        "daily_orders": 0,
        "daily_orders_for_contract": 0,
        "daily_loss": 0.0,
        "daily_loss_stop": 25.0,
        "is_halted": False,
        "kill_switch_active": False,
    }
    base.update(overrides)
    return SafeguardContext(**base)


def test_gate_approves_when_all_safeguards_pass() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(),
        config=_config(),
    )

    assert decision.action == "APPROVE"
    assert decision.reason == "ok"
    assert decision.side == "NO"


def test_rejects_when_event_within_cutoff_window() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(event_start=NOW + timedelta(hours=1)),
        config=_config(),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "event_cutoff"


def test_rejects_when_event_time_missing() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(event_start=None),
        config=_config(),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "missing_event_time"


def test_halts_contract_on_price_swing_versus_previous_observed_mid() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(yes_bid=0.44, yes_ask=0.46),
        context=_context(previous_observed_mid=0.30),
        config=_config(),
    )

    assert decision.action == "HALT_CONTRACT"
    assert decision.reason == "price_swing"


def test_skips_when_fresh_mid_drifts_into_the_skip_band() -> None:
    # scored mid 0.30, fresh mid 0.35 -> drift 0.05 (between exec 0.03 and halt 0.10)
    decision = evaluate_signal(
        signal=_signal(signal_yes_bid=0.28, signal_yes_ask=0.32),
        quote=_quote(yes_bid=0.34, yes_ask=0.36),
        context=_context(previous_observed_mid=0.35),
        config=_config(),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "stale_model_skip"


def test_halts_when_fresh_mid_drifts_past_halt_threshold() -> None:
    # scored mid 0.30, fresh mid 0.45 -> drift 0.15 (>= halt 0.10)
    decision = evaluate_signal(
        signal=_signal(signal_yes_bid=0.28, signal_yes_ask=0.32),
        quote=_quote(yes_bid=0.44, yes_ask=0.46),
        context=_context(previous_observed_mid=0.45),
        config=_config(),
    )

    assert decision.action == "HALT_CONTRACT"
    assert decision.reason == "stale_model_halt"


def test_executes_within_drift_tolerance() -> None:
    # scored mid 0.30, fresh mid 0.31 -> drift 0.01 (< exec 0.03)
    decision = evaluate_signal(
        signal=_signal(signal_yes_bid=0.28, signal_yes_ask=0.32),
        quote=_quote(yes_bid=0.30, yes_ask=0.32),
        context=_context(previous_observed_mid=0.31),
        config=_config(),
    )

    assert decision.action == "APPROVE"


def test_rejects_when_daily_loss_stop_reached() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(daily_loss=25.0, daily_loss_stop=25.0),
        config=_config(),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "daily_loss_stop"


def test_rejects_when_daily_order_count_exhausted() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(daily_orders=20),
        config=_config(max_orders_per_day=20),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "max_orders_per_day"


def test_rejects_when_contract_already_traded_today() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(daily_orders_for_contract=1),
        config=_config(max_orders_per_contract_per_day=1),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "max_orders_per_contract"


def test_rejects_paper_account() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(account_available=False),
        config=_config(),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "paper_account"


def test_rejects_when_trading_mode_off() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(),
        config=_config(mode="off"),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "trading_disabled"


def test_rejects_when_risk_filter_failed() -> None:
    decision = evaluate_signal(
        signal=_signal(passes_risk_filter=False),
        quote=_quote(),
        context=_context(),
        config=_config(),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "risk_filter"


def test_rejects_when_kill_switch_active() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(kill_switch_active=True),
        config=_config(),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "kill_switch"


def test_rejects_when_contract_is_halted() -> None:
    decision = evaluate_signal(
        signal=_signal(),
        quote=_quote(),
        context=_context(is_halted=True),
        config=_config(),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "contract_halted"


def test_config_allows_real_orders_only_when_live_and_confirmed() -> None:
    assert LiveTradingConfig(mode="dry_run").allows_real_orders() is False
    assert LiveTradingConfig(mode="live").allows_real_orders() is False
    assert (
        LiveTradingConfig(
            mode="live",
            live_confirmation=LIVE_CONFIRMATION_TOKEN,
        ).allows_real_orders()
        is True
    )


def test_config_from_env_reads_mode_and_caps() -> None:
    config = LiveTradingConfig.from_env(
        {
            "KALORIE2_TRADING_MODE": "live",
            "KALORIE2_LIVE_CONFIRMATION": LIVE_CONFIRMATION_TOKEN,
            "KALORIE2_MAX_ORDER_DOLLARS": "7.5",
            "KALORIE2_DAILY_LOSS_STOP_DOLLARS": "30",
            "KALORIE2_EXECUTION_DRIFT_TOLERANCE": "0.04",
            "KALORIE2_FEE_RATE": "0.035",
        }
    )

    assert config.mode == "live"
    assert config.allows_real_orders() is True
    assert config.max_order_dollars == 7.5
    assert config.daily_loss_stop_dollars == 30.0
    assert config.execution_drift_tolerance == 0.04
    assert config.fee_rate == 0.035
