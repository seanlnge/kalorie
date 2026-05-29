from kalorie2.execution.config import LiveTradingConfig


def test_caps_scale_with_portfolio_value_by_default() -> None:
    config = LiveTradingConfig()

    small = config.effective_caps(30.0)
    assert small.max_order_dollars == 3.0  # 0.10 * 30
    assert small.max_event_exposure_dollars == 7.5  # 0.25 * 30
    assert small.max_total_exposure_dollars == 24.0  # 0.80 * 30
    assert small.daily_loss_stop_dollars == 7.5  # 0.25 * 30


def test_caps_compound_as_portfolio_grows() -> None:
    config = LiveTradingConfig()

    big = config.effective_caps(10_000.0)
    assert big.max_order_dollars == 1000.0
    assert big.max_total_exposure_dollars == 8000.0


def test_absolute_ceiling_clamps_fractional_cap_when_set() -> None:
    config = LiveTradingConfig(max_order_dollars=5.0)

    caps = config.effective_caps(1000.0)
    # fractional would be 100, but the absolute hard ceiling clamps it to 5.
    assert caps.max_order_dollars == 5.0


def test_fraction_zero_falls_back_to_absolute_ceiling() -> None:
    config = LiveTradingConfig(max_order_fraction=0.0, max_order_dollars=8.0)

    caps = config.effective_caps(1000.0)
    assert caps.max_order_dollars == 8.0


def test_from_env_reads_fractional_caps() -> None:
    config = LiveTradingConfig.from_env(
        {
            "KALORIE2_MAX_ORDER_FRACTION": "0.05",
            "KALORIE2_MAX_TOTAL_EXPOSURE_FRACTION": "0.5",
            "KALORIE2_DAILY_LOSS_STOP_FRACTION": "0.2",
        }
    )

    assert config.max_order_fraction == 0.05
    assert config.max_total_exposure_fraction == 0.5
    assert config.daily_loss_stop_fraction == 0.2
