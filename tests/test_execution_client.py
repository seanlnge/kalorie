import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from kalorie2.execution.client import KalshiExecutionClient, OrderbookDepth

BASE_URL = "https://api.test/trade-api/v2"
_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _client(handler) -> KalshiExecutionClient:
    http_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="")
    return KalshiExecutionClient(
        http_client=http_client,
        api_key_id="key-123",
        private_key=_PRIVATE_KEY,
        base_url=BASE_URL,
        max_retries=2,
        sleep=lambda _seconds: None,
    )


def test_submit_limit_order_builds_yes_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"order": {"order_id": "ord-1"}})

    client = _client(handler)
    order_id = client.submit_limit_order(
        ticker="MKT",
        action="buy",
        side="yes",
        limit_price_cents=45,
        count=8,
        client_order_id="coid-1",
    )

    assert order_id == "ord-1"
    assert captured["method"] == "POST"
    assert captured["path"].endswith("/portfolio/orders")
    body = captured["body"]
    assert body["ticker"] == "MKT"
    assert body["action"] == "buy"
    assert body["type"] == "limit"
    assert body["side"] == "yes"
    assert body["count"] == 8
    assert body["yes_price"] == 45
    assert "no_price" not in body
    assert body["client_order_id"] == "coid-1"
    headers = captured["headers"]
    assert headers["kalshi-access-key"] == "key-123"
    assert "kalshi-access-signature" in headers
    assert "kalshi-access-timestamp" in headers


def test_submit_limit_order_builds_no_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        assert body["side"] == "no"
        assert body["no_price"] == 58
        assert "yes_price" not in body
        return httpx.Response(200, json={"order": {"order_id": "ord-2"}})

    client = _client(handler)
    order_id = client.submit_limit_order(
        ticker="MKT",
        action="buy",
        side="no",
        limit_price_cents=58,
        count=3,
        client_order_id="coid-2",
    )

    assert order_id == "ord-2"


def test_submit_order_does_not_retry_on_server_error() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, json={"error": "boom"})

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.submit_limit_order(
            ticker="MKT",
            action="buy",
            side="yes",
            limit_price_cents=45,
            count=1,
            client_order_id="coid-3",
        )

    assert calls["count"] == 1


def test_get_market_quote_parses_cents_into_probabilities() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith("/markets/MKT")
        return httpx.Response(
            200,
            json={
                "market": {
                    "ticker": "MKT",
                    "yes_bid": 42,
                    "yes_ask": 45,
                    "no_bid": 55,
                    "no_ask": 58,
                }
            },
        )

    client = _client(handler)
    quote = client.get_market_quote("MKT")

    assert quote.market_ticker == "MKT"
    assert quote.yes_bid == 0.42
    assert quote.yes_ask == 0.45
    assert quote.no_bid == 0.55
    assert quote.no_ask == 0.58


def test_cancel_order_issues_delete() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path.endswith("/portfolio/orders/ord-9")
        return httpx.Response(200, json={"reduced_by": 5})

    client = _client(handler)
    assert client.cancel_order("ord-9") is True


def test_list_resting_orders_filters_by_ticker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/portfolio/orders")
        assert request.url.params.get("status") == "resting"
        assert request.url.params.get("ticker") == "MKT"
        return httpx.Response(200, json={"orders": [{"order_id": "ord-1"}]})

    client = _client(handler)
    orders = client.list_resting_orders(ticker="MKT")

    assert orders == [{"order_id": "ord-1"}]


def test_get_orderbook_parses_levels_and_builds_ask_ladders() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith("/markets/MKT/orderbook")
        return httpx.Response(
            200,
            json={
                "orderbook": {
                    "yes": [[40, 100], [38, 200]],
                    "no": [[55, 50], [53, 120]],
                }
            },
        )

    client = _client(handler)
    book = client.get_orderbook("MKT")

    assert book.market_ticker == "MKT"
    # YES asks are derived from resting NO bids (100 - no_price), ascending.
    assert book.ask_levels("YES") == [(0.45, 50), (0.47, 120)]
    # NO asks are derived from resting YES bids, ascending.
    assert book.ask_levels("NO") == [(0.60, 100), (0.62, 200)]
    assert book.best_ask("YES") == 0.45


def test_orderbook_depth_handles_empty_book() -> None:
    book = OrderbookDepth(market_ticker="MKT", yes_bids=[], no_bids=[])

    assert book.ask_levels("YES") == []
    assert book.best_ask("YES") is None


def test_reads_retry_on_transient_server_error() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(200, json={"balance": {"balance": 5000}})

    client = _client(handler)
    balance = client.get_balance()

    assert calls["count"] == 2
    assert balance["balance"]["balance"] == 5000
