from kalorie2.risk_presets import (
    BUILT_IN_RISK_PRESETS,
    RiskPreset,
    apply_risk_preset_to_market,
    get_risk_preset,
)


def test_built_in_risk_presets_are_ordered_from_conservative_to_growth() -> None:
    assert [preset.id for preset in BUILT_IN_RISK_PRESETS] == [
        "capital_preservation",
        "balanced",
        "growth",
    ]
    assert get_risk_preset("balanced").min_margin > 0


def test_risk_preset_selects_no_trade_and_fractional_kelly_size() -> None:
    preset = RiskPreset(
        id="unit",
        label="Unit",
        description="Unit test preset",
        trade_side="no_only",
        min_margin=0.02,
        kelly_fraction=0.5,
        max_position_fraction=0.05,
        max_event_exposure_fraction=0.12,
    )

    decision = apply_risk_preset_to_market(
        preset=preset,
        model_probability=0.35,
        yes_bid=0.42,
        yes_ask=0.45,
    )

    assert decision.side == "NO"
    assert decision.passes_filter is True
    assert decision.cost == 0.58
    assert decision.edge == 0.07
    assert decision.ev_per_contract == 0.07
    assert decision.kelly_fraction_raw == 0.166667
    assert decision.recommended_fraction == 0.05


def test_risk_preset_blocks_trade_when_margin_or_side_policy_fails() -> None:
    preset = RiskPreset(
        id="unit",
        label="Unit",
        description="Unit test preset",
        trade_side="yes_only",
        min_margin=0.04,
        kelly_fraction=0.5,
        max_position_fraction=0.05,
        max_event_exposure_fraction=0.12,
    )

    no_signal = apply_risk_preset_to_market(
        preset=preset,
        model_probability=0.35,
        yes_bid=0.42,
        yes_ask=0.45,
    )
    thin_yes_signal = apply_risk_preset_to_market(
        preset=preset,
        model_probability=0.48,
        yes_bid=0.42,
        yes_ask=0.45,
    )

    assert no_signal.side == "NONE"
    assert no_signal.passes_filter is False
    assert thin_yes_signal.side == "NONE"
    assert thin_yes_signal.passes_filter is False


def test_apply_risk_preset_does_not_enforce_event_exposure_cap() -> None:
    # Event-level exposure is enforced by the live executor's safeguards, not by
    # the per-market preset decision. This guards that boundary so callers never
    # assume the preset already applied an event cap.
    preset = RiskPreset(
        id="unit",
        label="Unit",
        description="Unit test preset",
        trade_side="no_only",
        min_margin=0.02,
        kelly_fraction=0.5,
        max_position_fraction=0.05,
        max_event_exposure_fraction=0.0,
    )

    decision = apply_risk_preset_to_market(
        preset=preset,
        model_probability=0.35,
        yes_bid=0.42,
        yes_ask=0.45,
    )

    assert decision.side == "NO"
    assert decision.passes_filter is True
    # Even with a zero event-exposure cap the per-market size is governed by
    # max_position_fraction; enforcing the event cap is the executor's job.
    assert decision.recommended_fraction == 0.05
