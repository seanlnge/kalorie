from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TradeSide = Literal["YES", "NO", "NONE"]
RiskTradeSide = Literal["all", "no_only", "yes_only"]


class RiskPresetBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RiskPreset(RiskPresetBase):
    id: str
    label: str
    description: str
    trade_side: RiskTradeSide
    min_margin: float = Field(ge=0.0)
    kelly_fraction: float = Field(ge=0.0, le=1.0)
    max_position_fraction: float = Field(ge=0.0, le=1.0)
    max_event_exposure_fraction: float = Field(ge=0.0, le=1.0)
    risk_of_ruin_estimate: float = Field(ge=0.0, le=1.0)
    risk_of_ruin_label: str


class RiskDecision(RiskPresetBase):
    risk_preset_id: str
    side: TradeSide
    edge: float
    cost: float
    ev_per_contract: float
    kelly_fraction_raw: float
    recommended_fraction: float
    passes_filter: bool


BUILT_IN_RISK_PRESETS: tuple[RiskPreset, ...] = (
    RiskPreset(
        id="capital_preservation",
        label="Capital Preservation",
        description="High-confidence NO-only posture with strict margin and low Kelly exposure.",
        trade_side="no_only",
        min_margin=0.04,
        kelly_fraction=0.25,
        max_position_fraction=0.015,
        max_event_exposure_fraction=0.05,
        risk_of_ruin_estimate=0.005,
        risk_of_ruin_label="Very low",
    ),
    RiskPreset(
        id="balanced",
        label="Balanced",
        description="Moderate NO-only preset aligned with current historical edge concentration.",
        trade_side="no_only",
        min_margin=0.02,
        kelly_fraction=0.5,
        max_position_fraction=0.05,
        max_event_exposure_fraction=0.12,
        risk_of_ruin_estimate=0.015,
        risk_of_ruin_label="Low",
    ),
    RiskPreset(
        id="growth",
        label="Growth",
        description="Broader all-side preset with lower margin hurdle and larger Kelly cap.",
        trade_side="all",
        min_margin=0.01,
        kelly_fraction=0.75,
        max_position_fraction=0.1,
        max_event_exposure_fraction=0.25,
        risk_of_ruin_estimate=0.04,
        risk_of_ruin_label="Moderate",
    ),
)


def list_risk_presets() -> list[RiskPreset]:
    return list(BUILT_IN_RISK_PRESETS)


def get_risk_preset(preset_id: str | None) -> RiskPreset:
    normalized_id = preset_id or "balanced"
    for preset in BUILT_IN_RISK_PRESETS:
        if preset.id == normalized_id:
            return preset
    raise KeyError(f"Unknown risk preset: {normalized_id}")


def apply_risk_preset_to_market(
    *,
    preset: RiskPreset,
    model_probability: float,
    yes_bid: float,
    yes_ask: float,
) -> RiskDecision:
    probability = _clip_probability(model_probability)
    yes_bid = _clip_probability(yes_bid)
    yes_ask = _clip_probability(yes_ask)

    if probability > yes_ask + preset.min_margin and preset.trade_side != "no_only":
        side: TradeSide = "YES"
        cost = yes_ask
        win_probability = probability
        edge = win_probability - cost
    elif probability < yes_bid - preset.min_margin and preset.trade_side != "yes_only":
        side = "NO"
        cost = 1.0 - yes_bid
        win_probability = 1.0 - probability
        edge = win_probability - cost
    else:
        return _empty_decision(preset.id)

    kelly_raw = _kelly_fraction(win_probability=win_probability, cost=cost)
    recommended = min(
        max(kelly_raw, 0.0) * preset.kelly_fraction,
        preset.max_position_fraction,
    )
    return RiskDecision(
        risk_preset_id=preset.id,
        side=side,
        edge=round(edge, 6),
        cost=round(cost, 6),
        ev_per_contract=round(edge, 6),
        kelly_fraction_raw=round(kelly_raw, 6),
        recommended_fraction=round(recommended, 6),
        passes_filter=edge > 0,
    )


def _empty_decision(preset_id: str) -> RiskDecision:
    return RiskDecision(
        risk_preset_id=preset_id,
        side="NONE",
        edge=0.0,
        cost=0.0,
        ev_per_contract=0.0,
        kelly_fraction_raw=0.0,
        recommended_fraction=0.0,
        passes_filter=False,
    )


def _kelly_fraction(*, win_probability: float, cost: float) -> float:
    if cost <= 0.0 or cost >= 1.0:
        return 0.0
    return (win_probability - cost) / (1.0 - cost)


def _clip_probability(value: float) -> float:
    return min(0.999999, max(0.000001, float(value)))
