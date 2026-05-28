from __future__ import annotations

import json
from pathlib import Path
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
    ),
)


def list_risk_presets() -> list[RiskPreset]:
    return list(BUILT_IN_RISK_PRESETS)


def list_saved_risk_presets(store_path: Path) -> list[RiskPreset]:
    stored = _read_store(store_path)
    deleted_builtin_ids = set(stored["deleted_builtin_ids"])
    by_id = {
        preset.id: preset
        for preset in BUILT_IN_RISK_PRESETS
        if preset.id not in deleted_builtin_ids
    }
    for preset in stored["presets"]:
        by_id[preset.id] = preset
    return list(by_id.values())


def save_risk_preset(preset: RiskPreset, store_path: Path) -> list[RiskPreset]:
    stored = _read_store(store_path)
    presets_by_id = {entry.id: entry for entry in stored["presets"]}
    presets_by_id[preset.id] = preset
    deleted_builtin_ids = set(stored["deleted_builtin_ids"])
    deleted_builtin_ids.discard(preset.id)
    _write_store(
        store_path,
        presets=list(presets_by_id.values()),
        deleted_builtin_ids=sorted(deleted_builtin_ids),
    )
    return list_saved_risk_presets(store_path)


def delete_risk_preset(preset_id: str, store_path: Path) -> list[RiskPreset]:
    current = list_saved_risk_presets(store_path)
    if len(current) <= 1:
        raise ValueError("At least one risk preset must remain")
    stored = _read_store(store_path)
    presets = [preset for preset in stored["presets"] if preset.id != preset_id]
    deleted_builtin_ids = set(stored["deleted_builtin_ids"])
    if preset_id in {preset.id for preset in BUILT_IN_RISK_PRESETS}:
        deleted_builtin_ids.add(preset_id)
    _write_store(
        store_path,
        presets=presets,
        deleted_builtin_ids=sorted(deleted_builtin_ids),
    )
    return list_saved_risk_presets(store_path)


def get_risk_preset(preset_id: str | None, store_path: Path | None = None) -> RiskPreset:
    normalized_id = preset_id or "balanced"
    presets = list_saved_risk_presets(store_path) if store_path else list_risk_presets()
    for preset in presets:
        if preset.id == normalized_id:
            return preset
    raise KeyError(f"Unknown risk preset: {normalized_id}")


def _read_store(store_path: Path) -> dict[str, list]:
    if not store_path.exists():
        return {"presets": [], "deleted_builtin_ids": []}
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    presets = [RiskPreset.model_validate(entry) for entry in payload.get("presets", [])]
    deleted_builtin_ids = [
        str(entry)
        for entry in payload.get("deleted_builtin_ids", [])
        if isinstance(entry, str)
    ]
    return {"presets": presets, "deleted_builtin_ids": deleted_builtin_ids}


def _write_store(
    store_path: Path,
    *,
    presets: list[RiskPreset],
    deleted_builtin_ids: list[str],
) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "presets": [preset.model_dump(mode="json") for preset in presets],
        "deleted_builtin_ids": deleted_builtin_ids,
    }
    temp_path = store_path.with_suffix(f"{store_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(store_path)


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
