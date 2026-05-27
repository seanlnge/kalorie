from __future__ import annotations

import random
from typing import Literal

from pydantic import Field

from kalorie2.model_cards import EvaluationRow, ModelCardBase
from kalorie2.risk_presets import RiskPreset, apply_risk_preset_to_market


class ReturnPercentileBand(ModelCardBase):
    p10: float
    p25: float
    expected: float
    p75: float
    p90: float


class RiskPresetTrial(ModelCardBase):
    risk_preset_id: str
    label: str
    trade_side: Literal["all", "no_only", "yes_only"]
    min_margin: float = Field(ge=0.0)
    kelly_fraction: float = Field(ge=0.0, le=1.0)
    max_position_fraction: float = Field(ge=0.0, le=1.0)
    max_event_exposure_fraction: float = Field(ge=0.0, le=1.0)
    risk_of_ruin_estimate: float = Field(ge=0.0, le=1.0)
    risk_of_ruin_label: str
    trade_count: int = Field(ge=0)
    market_count: int = Field(ge=0)
    trade_percent: float = Field(ge=0.0)
    ev_per_10_markets: float
    expected_return_per_market: ReturnPercentileBand


def build_risk_preset_trials(
    rows: list[EvaluationRow],
    *,
    presets: list[RiskPreset],
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 531,
) -> list[RiskPresetTrial]:
    return [
        _build_risk_preset_trial(
            rows,
            preset=preset,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for preset in presets
    ]


def _build_risk_preset_trial(
    rows: list[EvaluationRow],
    *,
    preset: RiskPreset,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> RiskPresetTrial:
    trade_rows = _risk_trade_rows(rows, preset=preset)
    total_ev = sum(row["expected_pnl"] for row in trade_rows)
    expected_per_market = total_ev / len(rows) if rows else 0.0
    samples = _risk_return_per_market_samples(
        rows,
        preset=preset,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    band = ReturnPercentileBand(
        p10=round(_percentile(samples, 0.10), 6),
        p25=round(_percentile(samples, 0.25), 6),
        expected=round(expected_per_market, 6),
        p75=round(_percentile(samples, 0.75), 6),
        p90=round(_percentile(samples, 0.90), 6),
    )
    return RiskPresetTrial(
        risk_preset_id=preset.id,
        label=preset.label,
        trade_side=preset.trade_side,
        min_margin=preset.min_margin,
        kelly_fraction=preset.kelly_fraction,
        max_position_fraction=preset.max_position_fraction,
        max_event_exposure_fraction=preset.max_event_exposure_fraction,
        risk_of_ruin_estimate=preset.risk_of_ruin_estimate,
        risk_of_ruin_label=preset.risk_of_ruin_label,
        trade_count=len(trade_rows),
        market_count=len(rows),
        trade_percent=(len(trade_rows) / len(rows)) if rows else 0.0,
        ev_per_10_markets=round(expected_per_market * 10.0, 6),
        expected_return_per_market=band,
    )


def _risk_trade_rows(rows: list[EvaluationRow], *, preset: RiskPreset) -> list[dict[str, float]]:
    trades: list[dict[str, float]] = []
    for row in rows:
        decision = apply_risk_preset_to_market(
            preset=preset,
            model_probability=row.model_probability,
            yes_bid=row.yes_bid,
            yes_ask=row.yes_ask,
        )
        if not decision.passes_filter:
            continue
        if decision.side == "YES":
            realized_pnl = row.outcome_label - decision.cost
        elif decision.side == "NO":
            realized_pnl = (1 - row.outcome_label) - decision.cost
        else:
            continue
        trades.append(
            {
                "expected_pnl": decision.ev_per_contract,
                "realized_pnl": float(realized_pnl),
                "cost": decision.cost,
            }
        )
    return trades


def _risk_return_per_market_samples(
    rows: list[EvaluationRow],
    *,
    preset: RiskPreset,
    samples: int,
    seed: int,
) -> list[float]:
    if not rows:
        return [0.0]
    grouped: dict[str, list[EvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.event_ticker, []).append(row)
    groups = list(grouped.values())
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled_rows: list[EvaluationRow] = []
        for _ in groups:
            sampled_rows.extend(rng.choice(groups))
        trades = _risk_trade_rows(sampled_rows, preset=preset)
        values.append(sum(row["realized_pnl"] for row in trades) / len(sampled_rows))
    return values


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
