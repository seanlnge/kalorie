from decimal import Decimal

import httpx

from kalorie.webapi.kalshi_service import KalshiWebService


def test_list_open_mention_markets_filters_to_earnings_series_and_normalizes_prices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(404, json={"error": "not found"})
        assert request.url.path == "/trade-api/v2/markets"
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL",
                        "event_ticker": "KXEARNINGSMENTIONWMT-26Q2",
                        "title": "Will WMT mention omnichannel during earnings?",
                        "rules_primary": "Resolves yes if phrase is mentioned.",
                        "yes_sub_title": "Omnichannel",
                        "yes_bid_dollars": "0.34",
                        "yes_ask_dollars": "0.53",
                        "volume": 1234,
                    },
                    {
                        "ticker": "KXOTHERMARKET-TEST",
                        "event_ticker": "KXOTHERMARKET",
                        "title": "Not an earnings mention market",
                        "yes_bid_dollars": "0.20",
                        "yes_ask_dollars": "0.80",
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.elections.kalshi.com")
    service = KalshiWebService(http_client=client)

    markets = service.list_open_mention_markets()

    assert len(markets) == 1
    market = markets[0]
    assert market.market_ticker == "KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL"
    assert market.event_ticker == "KXEARNINGSMENTIONWMT-26Q2"
    assert market.company_symbol == "WMT"
    assert market.target_phrase == "Omnichannel"
    assert market.yes_bid == Decimal("0.34")
    assert market.yes_ask == Decimal("0.53")


def test_list_open_mention_markets_derives_yes_ask_from_no_bid_when_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXEARNINGSMENTIONCAVA-26Q1-TRAFFIC",
                        "event_ticker": "KXEARNINGSMENTIONCAVA-26Q1",
                        "title": "Will CAVA mention traffic during earnings?",
                        "rules_primary": "Resolution text",
                        "yes_bid_dollars": "0.45",
                        "no_bid_dollars": "0.41",
                        "volume": 50,
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.elections.kalshi.com")
    service = KalshiWebService(http_client=client)

    markets = service.list_open_mention_markets()

    assert len(markets) == 1
    market = markets[0]
    assert market.yes_ask == Decimal("0.59")


def test_list_open_mention_markets_reads_phrase_from_custom_strike() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXEARNINGSMENTIONWMT-26MAY21-OMNI",
                        "event_ticker": "KXEARNINGSMENTIONWMT-26MAY21",
                        "title": "What will Walmart Inc. say during their next earnings call?",
                        "custom_strike": {"Word": "Omnichannel"},
                        "yes_bid_dollars": "0.31",
                        "yes_ask_dollars": "0.48",
                        "volume": 123,
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.elections.kalshi.com")
    service = KalshiWebService(http_client=client)

    markets = service.list_open_mention_markets()

    assert len(markets) == 1
    assert markets[0].target_phrase == "Omnichannel"


def test_service_uses_absolute_default_kalshi_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(404, json={"error": "not found"})
        assert str(request.url).startswith("https://api.elections.kalshi.com/trade-api/v2/markets")
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXEARNINGSMENTIONNVDA-26Q2-OPENAI",
                        "event_ticker": "KXEARNINGSMENTIONNVDA-26Q2",
                        "title": "Will NVDA mention OpenAI during earnings?",
                        "rules_primary": "Resolution text",
                        "yes_bid_dollars": "0.51",
                        "yes_ask_dollars": "0.62",
                        "volume": 100,
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = KalshiWebService(http_client=client)

    markets = service.list_open_mention_markets()

    assert len(markets) == 1
    assert markets[0].market_ticker == "KXEARNINGSMENTIONNVDA-26Q2-OPENAI"


def test_list_open_mention_markets_returns_partial_results_on_rate_limit() -> None:
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(404, json={"error": "not found"})
        call_count["value"] += 1
        if call_count["value"] == 1:
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL",
                            "event_ticker": "KXEARNINGSMENTIONWMT-26Q2",
                            "title": "Will WMT mention omnichannel during earnings?",
                            "yes_bid_dollars": "0.34",
                            "yes_ask_dollars": "0.53",
                            "volume": 1234,
                        }
                    ],
                    "cursor": "next-page",
                },
            )
        return httpx.Response(429, json={"error": "rate limited"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = KalshiWebService(http_client=client)

    markets = service.list_open_mention_markets()

    assert len(markets) == 1
    assert markets[0].market_ticker == "KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL"


def test_list_open_mention_markets_returns_empty_list_on_first_page_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(429, json={"error": "rate limited"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = KalshiWebService(http_client=client)

    markets = service.list_open_mention_markets()

    assert markets == []


def test_list_open_mention_markets_falls_back_to_series_scan_when_global_scan_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        if path == "/v1/search/series":
            return httpx.Response(404, json={"error": "not found"})
        if path.endswith("/markets") and "series_ticker" not in params:
            return httpx.Response(200, json={"markets": []})
        if path.endswith("/series"):
            return httpx.Response(
                200,
                json={
                    "series": [
                        {"ticker": "KXEARNINGSMENTIONWMT"},
                        {"ticker": "KXEARNINGSMENTIONNVDA"},
                    ]
                },
            )
        if path.endswith("/markets") and params.get("series_ticker") == "KXEARNINGSMENTIONWMT":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONWMT-26MAY21-OMNI",
                            "event_ticker": "KXEARNINGSMENTIONWMT-26MAY21",
                            "title": "Will WMT mention omnichannel during earnings?",
                            "yes_bid_dollars": "0.31",
                            "yes_ask_dollars": "0.48",
                            "volume": 123,
                        }
                    ]
                },
            )
        if path.endswith("/markets") and params.get("series_ticker") == "KXEARNINGSMENTIONNVDA":
            return httpx.Response(200, json={"markets": []})
        raise AssertionError(f"Unexpected request: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = KalshiWebService(http_client=client)

    markets = service.list_open_mention_markets()

    assert len(markets) == 1
    assert markets[0].market_ticker == "KXEARNINGSMENTIONWMT-26MAY21-OMNI"


def test_list_open_mention_markets_uses_ttl_cache_between_calls() -> None:
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(404, json={"error": "not found"})
        call_count["value"] += 1
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXEARNINGSMENTIONWMT-26MAY21-OMNI",
                        "event_ticker": "KXEARNINGSMENTIONWMT-26MAY21",
                        "title": "Will WMT mention omnichannel during earnings?",
                        "yes_bid_dollars": "0.31",
                        "yes_ask_dollars": "0.48",
                        "volume": 123,
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = KalshiWebService(http_client=client, market_cache_ttl_seconds=60.0)

    first = service.list_open_mention_markets()
    second = service.list_open_mention_markets()

    assert len(first) == 1
    assert len(second) == 1
    assert call_count["value"] == 1


def test_list_event_mention_markets_calls_direct_event_filter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trade-api/v2/markets"
        assert request.url.params.get("event_ticker") == "KXEARNINGSMENTIONWMT-26MAY21"
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXEARNINGSMENTIONWMT-26MAY21-OMNI",
                        "event_ticker": "KXEARNINGSMENTIONWMT-26MAY21",
                        "title": "What will Walmart Inc. say during their next earnings call?",
                        "yes_sub_title": "Omnichannel",
                        "yes_bid_dollars": "0.31",
                        "yes_ask_dollars": "0.48",
                        "volume": 123,
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = KalshiWebService(http_client=client)

    markets = service.list_event_mention_markets("KXEARNINGSMENTIONWMT-26MAY21")

    assert len(markets) == 1
    assert markets[0].target_phrase == "Omnichannel"


def test_list_open_mention_markets_uses_v1_series_search_when_available() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search/series":
            return httpx.Response(
                200,
                json={
                    "total_results_count": 1,
                    "current_page": [
                        {
                            "series_ticker": "KXEARNINGSMENTIONWMT",
                            "series_title": "What will Walmart say during their earnings call?",
                            "event_ticker": "KXEARNINGSMENTIONWMT-26MAY21",
                            "event_title": "What will Walmart say during their next earnings call?",
                            "markets": [
                                {
                                    "ticker": "KXEARNINGSMENTIONWMT-26MAY21-OMNI",
                                    "yes_subtitle": "Omnichannel",
                                    "yes_bid_dollars": "0.31",
                                    "yes_ask_dollars": "0.48",
                                    "volume": 123,
                                }
                            ],
                        }
                    ],
                    "next_cursor": "",
                },
            )
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = KalshiWebService(http_client=client)

    markets = service.list_open_mention_markets()

    assert len(markets) == 1
    assert markets[0].market_ticker == "KXEARNINGSMENTIONWMT-26MAY21-OMNI"
    assert markets[0].event_ticker == "KXEARNINGSMENTIONWMT-26MAY21"
    assert markets[0].target_phrase == "Omnichannel"
