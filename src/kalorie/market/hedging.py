from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

EPSILON = 1e-6


@dataclass(frozen=True)
class HedgeCandidate:
    market_id: str
    phrase: str
    side: Literal["yes", "no"]
    price: float
    win_probability: float
    model_yes_probability: float
    market_yes_bid: float
    market_yes_ask: float
    edge: float
    expected_profit_per_contract: float
    expected_profit_per_dollar: float
    variance_per_dollar: float


def build_best_side_candidates(
    rows: list[dict],
    *,
    model_probability_key: str,
    min_edge: float,
) -> list[HedgeCandidate]:
    candidates: list[HedgeCandidate] = []
    for row in rows:
        model_probability = _clip_probability(float(row[model_probability_key]))
        market_id = str(row["market_id"])
        phrase = str(row["phrase"])
        yes_bid = float(row["kalshi_yes_bid"])
        yes_ask = float(row["kalshi_yes_ask"])
        no_ask = 1.0 - yes_bid
        if yes_ask <= 0 or no_ask <= 0:
            continue

        yes_edge = model_probability - yes_ask
        no_edge = (1.0 - model_probability) - no_ask
        if yes_edge >= no_edge:
            candidate = _candidate(
                market_id=market_id,
                phrase=phrase,
                side="yes",
                price=yes_ask,
                win_probability=model_probability,
                model_yes_probability=model_probability,
                market_yes_bid=yes_bid,
                market_yes_ask=yes_ask,
                edge=yes_edge,
            )
        else:
            candidate = _candidate(
                market_id=market_id,
                phrase=phrase,
                side="no",
                price=no_ask,
                win_probability=1.0 - model_probability,
                model_yes_probability=model_probability,
                market_yes_bid=yes_bid,
                market_yes_ask=yes_ask,
                edge=no_edge,
            )
        if candidate.edge >= min_edge:
            candidates.append(candidate)
    return candidates


def build_hedge_plan(
    rows: list[dict],
    *,
    budget: float,
    model_probability_key: str = "model_company_probability",
    min_edge: float = 0.0,
    risk_aversion: float = 0.50,
    max_fraction_per_market: float = 0.35,
    force_full_deployment: bool = False,
    max_positions: int | None = None,
) -> dict:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0 < max_fraction_per_market <= 1:
        raise ValueError("max_fraction_per_market must be in (0, 1]")
    if min_edge < 0:
        raise ValueError("min_edge must be non-negative")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if max_positions is not None and max_positions < 1:
        raise ValueError("max_positions must be at least 1 when provided")

    candidates = build_best_side_candidates(
        rows,
        model_probability_key=model_probability_key,
        min_edge=min_edge,
    )
    candidates = sorted(candidates, key=lambda row: row.expected_profit_per_dollar, reverse=True)
    if max_positions is not None:
        candidates = candidates[:max_positions]
    stakes = _allocate_stakes(
        candidates,
        budget=budget,
        risk_aversion=risk_aversion,
        max_fraction_per_market=max_fraction_per_market,
        force_full_deployment=force_full_deployment,
    )
    return _portfolio_payload(
        candidates,
        stakes,
        budget=budget,
        assumptions={
            "independent_markets": True,
            "variance_model": "bernoulli_settlement_with_independent_markets",
            "execution_price": "yes_ask_or_implied_no_ask_from_yes_bid",
        },
        controls={
            "risk_aversion": risk_aversion,
            "min_edge": min_edge,
            "max_fraction_per_market": max_fraction_per_market,
            "force_full_deployment": force_full_deployment,
            "max_positions": max_positions,
        },
    )


def _candidate(
    *,
    market_id: str,
    phrase: str,
    side: Literal["yes", "no"],
    price: float,
    win_probability: float,
    model_yes_probability: float,
    market_yes_bid: float,
    market_yes_ask: float,
    edge: float,
) -> HedgeCandidate:
    expected_profit_per_contract = win_probability - price
    expected_profit_per_dollar = (win_probability / price) - 1.0
    variance_per_dollar = (win_probability * (1.0 - win_probability)) / (price**2)
    return HedgeCandidate(
        market_id=market_id,
        phrase=phrase,
        side=side,
        price=price,
        win_probability=win_probability,
        model_yes_probability=model_yes_probability,
        market_yes_bid=market_yes_bid,
        market_yes_ask=market_yes_ask,
        edge=edge,
        expected_profit_per_contract=expected_profit_per_contract,
        expected_profit_per_dollar=expected_profit_per_dollar,
        variance_per_dollar=variance_per_dollar,
    )


def _allocate_stakes(
    candidates: list[HedgeCandidate],
    *,
    budget: float,
    risk_aversion: float,
    max_fraction_per_market: float,
    force_full_deployment: bool,
) -> list[float]:
    if not candidates:
        return []
    cap_per_market = budget * max_fraction_per_market
    caps = [cap_per_market for _ in candidates]
    a_values = [candidate.expected_profit_per_dollar for candidate in candidates]
    b_values = [max(candidate.variance_per_dollar, EPSILON) for candidate in candidates]

    if risk_aversion <= 0:
        return _allocate_linear(
            a_values,
            caps,
            budget=budget,
            force_full_deployment=force_full_deployment,
        )

    unconstrained = [
        min(caps[index], max(a_values[index] / (2.0 * risk_aversion * b_values[index]), 0.0))
        for index in range(len(candidates))
    ]
    unconstrained_total = sum(unconstrained)
    if not force_full_deployment and unconstrained_total <= budget + 1e-9:
        return unconstrained

    target_budget = budget if force_full_deployment or unconstrained_total > budget else unconstrained_total
    max_possible = sum(caps)
    if max_possible + 1e-9 < target_budget:
        raise ValueError(
            "cannot satisfy deployment target with current per-market cap; "
            "increase max_fraction_per_market or reduce budget"
        )

    lower = min(
        a_values[index] - (2.0 * risk_aversion * b_values[index] * caps[index])
        for index in range(len(candidates))
    ) - 1.0
    upper = max(a_values) + 1.0
    allocation = unconstrained
    for _ in range(140):
        midpoint = (lower + upper) / 2.0
        allocation = [
            min(caps[index], max((a_values[index] - midpoint) / (2.0 * risk_aversion * b_values[index]), 0.0))
            for index in range(len(candidates))
        ]
        if sum(allocation) > target_budget:
            lower = midpoint
        else:
            upper = midpoint
    if force_full_deployment and abs(sum(allocation) - target_budget) > 1e-3:
        raise ValueError("failed to reach full deployment target")
    return allocation


def _allocate_linear(
    expected_profit_per_dollar: list[float],
    caps: list[float],
    *,
    budget: float,
    force_full_deployment: bool,
) -> list[float]:
    order = sorted(range(len(expected_profit_per_dollar)), key=lambda index: expected_profit_per_dollar[index], reverse=True)
    stakes = [0.0 for _ in expected_profit_per_dollar]
    remaining = budget
    for index in order:
        if remaining <= 1e-9:
            break
        if expected_profit_per_dollar[index] <= 0 and not force_full_deployment:
            continue
        stake = min(caps[index], remaining)
        stakes[index] = stake
        remaining -= stake
    if force_full_deployment and remaining > 1e-3:
        raise ValueError(
            "cannot satisfy full deployment target with current per-market cap; "
            "increase max_fraction_per_market or reduce budget"
        )
    return stakes


def _portfolio_payload(
    candidates: list[HedgeCandidate],
    stakes: list[float],
    *,
    budget: float,
    assumptions: dict,
    controls: dict,
) -> dict:
    allocations = []
    deployed = 0.0
    expected_profit = 0.0
    variance = 0.0
    for candidate, stake in zip(candidates, stakes, strict=True):
        if stake <= 1e-9:
            continue
        deployed += stake
        contracts = stake / candidate.price
        position_expected_profit = stake * candidate.expected_profit_per_dollar
        position_variance = (stake**2) * candidate.variance_per_dollar
        expected_profit += position_expected_profit
        variance += position_variance
        allocations.append(
            {
                "market_id": candidate.market_id,
                "phrase": candidate.phrase,
                "side": candidate.side,
                "stake_dollars": round(stake, 6),
                "contracts": round(contracts, 6),
                "price_paid": round(candidate.price, 6),
                "market_yes_bid": round(candidate.market_yes_bid, 6),
                "market_yes_ask": round(candidate.market_yes_ask, 6),
                "model_yes_probability": round(candidate.model_yes_probability, 6),
                "side_win_probability": round(candidate.win_probability, 6),
                "edge_vs_execution_price": round(candidate.edge, 6),
                "expected_profit_dollars": round(position_expected_profit, 6),
                "expected_profit_per_dollar": round(candidate.expected_profit_per_dollar, 6),
                "variance_dollars_sq": round(position_variance, 6),
                "stdev_dollars": round(sqrt(max(position_variance, 0.0)), 6),
            }
        )

    expected_roi_on_budget = (expected_profit / budget) if budget > 0 else 0.0
    expected_roi_on_deployed = (expected_profit / deployed) if deployed > 0 else 0.0
    portfolio_stdev = sqrt(max(variance, 0.0))
    return {
        "budget_dollars": round(budget, 6),
        "deployed_dollars": round(deployed, 6),
        "cash_reserve_dollars": round(max(budget - deployed, 0.0), 6),
        "position_count": len(allocations),
        "expected_profit_dollars": round(expected_profit, 6),
        "expected_roi_on_budget": round(expected_roi_on_budget, 6),
        "expected_roi_on_deployed": round(expected_roi_on_deployed, 6),
        "variance_dollars_sq": round(variance, 6),
        "stdev_dollars": round(portfolio_stdev, 6),
        "information_ratio_like": round(expected_profit / portfolio_stdev, 6)
        if portfolio_stdev > 0
        else None,
        "assumptions": assumptions,
        "controls": controls,
        "allocations": allocations,
    }


def _clip_probability(value: float) -> float:
    return min(1.0 - EPSILON, max(EPSILON, value))
