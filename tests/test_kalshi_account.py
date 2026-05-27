from kalorie2.kalshi_account import build_account_summary


def test_build_account_summary_uses_balance_and_position_dollar_fields() -> None:
    summary = build_account_summary(
        balance_payload={
            "balance": {
                "portfolio_value": 125_50,
                "available_balance": 82_25,
            }
        },
        positions_payload={
            "market_positions": [
                {"ticker": "A", "market_exposure": 10_00},
                {"ticker": "B", "market_exposure_dollars": "7.50"},
            ]
        },
    )

    assert summary.available is True
    assert summary.source == "kalshi"
    assert summary.portfolio_value == 125.50
    assert summary.free_cash == 82.25
    assert summary.position_exposure == 17.50
    assert summary.bankroll == 82.25


def test_build_account_summary_falls_back_to_paper_bankroll_without_auth() -> None:
    summary = build_account_summary(balance_payload=None, positions_payload=None)

    assert summary.available is False
    assert summary.source == "paper"
    assert summary.portfolio_value is None
    assert summary.free_cash is None
    assert summary.position_exposure is None
    assert summary.bankroll == 100.0


def test_build_account_summary_accepts_scalar_cent_balance() -> None:
    summary = build_account_summary(balance_payload={"balance": 12_345}, positions_payload=None)

    assert summary.available is True
    assert summary.source == "kalshi"
    assert summary.portfolio_value == 123.45
    assert summary.free_cash == 123.45
    assert summary.bankroll == 123.45
