import pytest

from kalorie2.kalshi_orderbook import (
    OrderbookState,
    build_orderbook_subscription,
)


def test_orderbook_snapshot_derives_yes_quote_from_yes_and_no_books() -> None:
    state = OrderbookState("MARKET-1")

    quote = state.apply_message(
        {
            "type": "orderbook_snapshot",
            "msg": {
                "market_ticker": "MARKET-1",
                "yes": [[24, 12], [31, 8]],
                "no": [[58, 4], [63, 2]],
            },
        }
    )

    assert quote is not None
    assert quote.market_ticker == "MARKET-1"
    assert quote.yes_bid == 0.31
    assert quote.yes_ask == 0.37
    assert quote.yes_mid == 0.34


def test_orderbook_delta_updates_and_removes_price_levels() -> None:
    state = OrderbookState("MARKET-1")
    state.apply_message(
        {
            "type": "orderbook_snapshot",
            "msg": {
                "market_ticker": "MARKET-1",
                "yes": [[31, 8]],
                "no": [[63, 2]],
            },
        }
    )

    improved_quote = state.apply_message(
        {
            "type": "orderbook_delta",
            "msg": {
                "market_ticker": "MARKET-1",
                "side": "yes",
                "price": 35,
                "delta": 3,
            },
        }
    )
    removed_quote = state.apply_message(
        {
            "type": "orderbook_delta",
            "msg": {
                "market_ticker": "MARKET-1",
                "side": "yes",
                "price": 35,
                "delta": -3,
            },
        }
    )

    assert improved_quote is not None
    assert improved_quote.yes_bid == 0.35
    assert removed_quote is not None
    assert removed_quote.yes_bid == 0.31


def test_orderbook_ignores_other_markets_and_empty_quotes() -> None:
    state = OrderbookState("MARKET-1")

    ignored = state.apply_message(
        {
            "type": "orderbook_snapshot",
            "msg": {
                "market_ticker": "MARKET-2",
                "yes": [[31, 8]],
                "no": [[63, 2]],
            },
        }
    )
    empty = state.apply_message(
        {
            "type": "orderbook_snapshot",
            "msg": {
                "market_ticker": "MARKET-1",
                "yes": [],
                "no": [[63, 2]],
            },
        }
    )

    assert ignored is None
    assert empty is None


def test_orderbook_subscription_deduplicates_sorted_tickers() -> None:
    payload = build_orderbook_subscription(7, ["MARKET-B", "MARKET-A", "MARKET-B"])

    assert payload == {
        "id": 7,
        "cmd": "subscribe",
        "params": {
            "channels": ["orderbook_delta"],
            "market_tickers": ["MARKET-A", "MARKET-B"],
        },
    }


def test_orderbook_subscription_requires_tickers() -> None:
    with pytest.raises(ValueError, match="market_tickers"):
        build_orderbook_subscription(7, [])
