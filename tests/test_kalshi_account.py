from kalorie2.kalshi_account import build_account_summary, build_open_positions_summary


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


def test_build_open_positions_summary_normalizes_kalshi_positions() -> None:
    summary = build_open_positions_summary(
        {
            "market_positions": [
                {
                    "ticker": "MKT-YES",
                    "position": 5,
                    "fees_paid": 120,
                    "market_exposure": 250,
                    "market_value": 310,
                    "realized_pnl": 40,
                    "average_price": 50,
                },
                {
                    "market_ticker": "MKT-NO",
                    "position": -3,
                    "market_exposure_dollars": "1.80",
                    "market_value_dollars": "2.10",
                    "realized_pnl_dollars": "-0.25",
                    "average_price_dollars": "0.60",
                },
            ]
        }
    )

    assert summary.available is True
    assert summary.open_position_count == 2
    assert summary.total_contracts == 8
    assert summary.total_exposure == 4.3
    assert summary.total_market_value == 5.2
    assert summary.realized_pnl == 0.15
    assert summary.average_price == 0.5375
    assert summary.positions[0].side == "YES"
    assert summary.positions[1].side == "NO"


def test_build_open_positions_summary_supports_doc_shaped_positions() -> None:
    summary = build_open_positions_summary(
        {
            "market_positions": [
                {
                    "ticker": "KX-YES",
                    "position_fp": "5.0000",
                    "total_traded_dollars": "2.50",
                    "market_exposure_dollars": "2.25",
                    "fees_paid_dollars": "0.05",
                }
            ]
        }
    )

    assert summary.open_position_count == 1
    assert summary.total_contracts == 5
    assert summary.average_price == 0.5
    assert summary.total_exposure == 2.25
    assert summary.fees_paid == 0.05
    assert summary.positions[0].average_price == 0.5
