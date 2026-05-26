from kalorie.market.hedging import build_best_side_candidates, build_hedge_plan


def _rows() -> list[dict]:
    return [
        {
            "market_id": "NVDA-AUTO",
            "phrase": "automation",
            "kalshi_yes_bid": 0.48,
            "kalshi_yes_ask": 0.50,
            "model_company_probability": 0.70,
        },
        {
            "market_id": "NVDA-TRUM",
            "phrase": "trump",
            "kalshi_yes_bid": 0.78,
            "kalshi_yes_ask": 0.80,
            "model_company_probability": 0.20,
        },
    ]


def test_build_best_side_candidates_selects_side_with_higher_edge():
    candidates = build_best_side_candidates(
        _rows(),
        model_probability_key="model_company_probability",
        min_edge=0.0,
    )
    by_market = {row.market_id: row for row in candidates}
    assert by_market["NVDA-AUTO"].side == "yes"
    assert by_market["NVDA-TRUM"].side == "no"
    assert by_market["NVDA-AUTO"].edge > 0
    assert by_market["NVDA-TRUM"].edge > 0


def test_build_hedge_plan_respects_budget_and_outputs_risk_stats():
    plan = build_hedge_plan(
        _rows(),
        budget=100.0,
        model_probability_key="model_company_probability",
        risk_aversion=0.75,
        max_fraction_per_market=0.6,
        force_full_deployment=True,
    )
    assert plan["deployed_dollars"] == 100.0
    assert plan["position_count"] == 2
    assert plan["expected_profit_dollars"] > 0
    assert plan["variance_dollars_sq"] > 0
    assert plan["stdev_dollars"] > 0
    assert plan["cash_reserve_dollars"] == 0.0


def test_build_hedge_plan_can_leave_cash_when_edges_are_negative():
    rows = [
        {
            "market_id": "NVDA-BAD",
            "phrase": "bad",
            "kalshi_yes_bid": 0.58,
            "kalshi_yes_ask": 0.62,
            "model_company_probability": 0.60,
        }
    ]
    plan = build_hedge_plan(
        rows,
        budget=100.0,
        model_probability_key="model_company_probability",
        min_edge=0.05,
        risk_aversion=0.50,
        force_full_deployment=False,
    )
    assert plan["position_count"] == 0
    assert plan["deployed_dollars"] == 0.0
    assert plan["cash_reserve_dollars"] == 100.0
