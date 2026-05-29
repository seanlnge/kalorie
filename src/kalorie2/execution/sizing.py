from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict


class FillPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contracts: int = 0
    limit_price: float = 0.0
    blended_price: float = 0.0
    gross_cost_dollars: float = 0.0
    fee_dollars: float = 0.0
    expected_value_dollars: float = 0.0
    levels: list[tuple[float, int]] = []


def fee_per_contract(price: float, fee_rate: float) -> float:
    """Kalshi-style trading fee per contract: rate * price * (1 - price)."""

    return fee_rate * price * (1.0 - price)


def plan_fill(
    *,
    win_probability: float,
    ask_levels: list[tuple[float, int]],
    target_dollars: float,
    max_budget_dollars: float,
    min_margin: float,
    fee_rate: float,
) -> FillPlan:
    """Walk the ask ladder, taking each marginal contract only while it clears
    the fee-adjusted minimum margin and stays within the dollar budget.

    ``ask_levels`` must be ascending by price; each is ``(price, available)`` for
    the side being bought. ``win_probability`` is the chance that side wins.
    """

    budget = min(target_dollars, max_budget_dollars)
    if budget <= 0.0:
        return FillPlan()

    contracts = 0
    spent = 0.0
    fee_total = 0.0
    ev_total = 0.0
    worst_price = 0.0
    chosen: list[tuple[float, int]] = []

    for price, available in ask_levels:
        if price <= 0.0 or available <= 0:
            continue
        fee_c = fee_per_contract(price, fee_rate)
        edge_c = win_probability - price - fee_c
        if edge_c < min_margin:
            break
        affordable = math.floor((budget - spent + 1e-9) / price)
        take = min(available, affordable)
        if take <= 0:
            break
        contracts += take
        spent += take * price
        fee_total += take * fee_c
        ev_total += take * edge_c
        worst_price = price
        chosen.append((round(price, 4), take))
        if take < available:
            break

    if contracts < 1:
        return FillPlan()

    return FillPlan(
        contracts=contracts,
        limit_price=round(worst_price, 2),
        blended_price=round(spent / contracts, 4),
        gross_cost_dollars=round(spent, 2),
        fee_dollars=round(math.ceil(fee_total * 100) / 100.0, 2),
        expected_value_dollars=round(ev_total, 4),
        levels=chosen,
    )
