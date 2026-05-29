from kalorie2.execution.sizing import fee_per_contract, plan_fill


def test_fills_across_two_levels_when_both_are_positive_ev() -> None:
    plan = plan_fill(
        win_probability=0.62,
        ask_levels=[(0.53, 4), (0.55, 4)],
        target_dollars=100.0,
        max_budget_dollars=100.0,
        min_margin=0.02,
        fee_rate=0.0,
    )

    assert plan.contracts == 8
    assert plan.limit_price == 0.55
    assert plan.blended_price == 0.54
    assert plan.gross_cost_dollars == 4.32
    assert plan.levels == [(0.53, 4), (0.55, 4)]


def test_stops_walking_ladder_when_marginal_level_is_not_positive_ev() -> None:
    plan = plan_fill(
        win_probability=0.56,
        ask_levels=[(0.53, 4), (0.55, 4)],
        target_dollars=100.0,
        max_budget_dollars=100.0,
        min_margin=0.02,
        fee_rate=0.0,
    )

    assert plan.contracts == 4
    assert plan.limit_price == 0.53
    assert plan.levels == [(0.53, 4)]


def test_budget_caps_contracts_below_available_depth() -> None:
    plan = plan_fill(
        win_probability=0.62,
        ask_levels=[(0.53, 100)],
        target_dollars=2.0,
        max_budget_dollars=100.0,
        min_margin=0.02,
        fee_rate=0.0,
    )

    assert plan.contracts == 3
    assert plan.gross_cost_dollars == 1.59


def test_max_budget_caps_below_target() -> None:
    plan = plan_fill(
        win_probability=0.62,
        ask_levels=[(0.53, 100)],
        target_dollars=100.0,
        max_budget_dollars=1.5,
        min_margin=0.02,
        fee_rate=0.0,
    )

    assert plan.contracts == 2
    assert plan.gross_cost_dollars == 1.06


def test_fees_can_eliminate_a_thin_edge() -> None:
    fee = fee_per_contract(0.53, 0.07)
    assert round(fee, 5) == 0.01744

    plan = plan_fill(
        win_probability=0.56,
        ask_levels=[(0.53, 10)],
        target_dollars=100.0,
        max_budget_dollars=100.0,
        min_margin=0.02,
        fee_rate=0.07,
    )

    assert plan.contracts == 0


def test_empty_ladder_yields_no_fill() -> None:
    plan = plan_fill(
        win_probability=0.62,
        ask_levels=[],
        target_dollars=100.0,
        max_budget_dollars=100.0,
        min_margin=0.02,
        fee_rate=0.0,
    )

    assert plan.contracts == 0
    assert plan.limit_price == 0.0


def test_expected_value_is_net_of_fees() -> None:
    plan = plan_fill(
        win_probability=0.62,
        ask_levels=[(0.50, 2)],
        target_dollars=100.0,
        max_budget_dollars=100.0,
        min_margin=0.0,
        fee_rate=0.07,
    )

    # edge per contract = 0.62 - 0.50 - 0.07*0.50*0.50 = 0.1025; x2 contracts.
    assert plan.contracts == 2
    assert round(plan.expected_value_dollars, 4) == 0.205
