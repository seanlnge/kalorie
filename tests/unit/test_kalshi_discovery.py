from datetime import UTC, datetime
from decimal import Decimal

import httpx

from kalorie.clients.kalshi import KalshiPublicClient, parse_mention_market_contracts


def test_parse_mention_market_contracts_extracts_targets_from_rules_and_titles():
    payload = {
        "markets": [
            {
                "ticker": "CAVA-26Q1-TRAFFIC",
                "event_ticker": "CAVA-26Q1",
                "title": "Will CAVA mention traffic during earnings?",
                "rules_primary": (
                    "If traffic is said by any CAVA Group, Inc. representative during the "
                    "next CAVA Group, Inc. earnings call, then the market resolves to Yes."
                ),
                "yes_bid": 38,
                "yes_ask": 45,
                "observed_at": "2026-05-19T14:30:00Z",
            },
            {
                "ticker": "CAVA-26Q1-SAME-RESTAURANT-SALES",
                "event_ticker": "CAVA-26Q1",
                "title": 'Will CAVA mention "same restaurant sales" on its earnings call?',
                "rules_primary": "",
                "yes_bid": "0.50",
                "yes_ask": "0.58",
            },
        ]
    }

    contracts = parse_mention_market_contracts(payload, event_ticker="CAVA-26Q1")

    assert [contract.target_phrase.normalized_phrase for contract in contracts] == [
        "traffic",
        "same restaurant sales",
    ]
    assert contracts[0].market_id == "CAVA-26Q1-TRAFFIC"
    assert contracts[0].rules_text.startswith("If traffic is said")
    assert contracts[0].yes_bid == Decimal("0.38")
    assert contracts[0].observed_at == datetime(2026, 5, 19, 14, 30, tzinfo=UTC)


def test_parse_mention_market_contracts_prefers_dollar_price_fields():
    payload = {
        "markets": [
            {
                "ticker": "KXEARNINGSMENTIONNVDA-26MAY20-H20",
                "event_ticker": "KXEARNINGSMENTIONNVDA-26MAY20",
                "title": "What will NVIDIA Corporation say during their next earnings call?",
                "rules_primary": (
                    "If H20 is said by any NVIDIA Corporation representative during the next "
                    "NVIDIA earnings call, then the market resolves to Yes."
                ),
                "yes_bid_dollars": "0.22",
                "yes_ask_dollars": "0.29",
                "yes_bid": 22,
                "yes_ask": 29,
                "observed_at": "2026-05-20T20:30:00Z",
            }
        ]
    }

    contracts = parse_mention_market_contracts(
        payload,
        event_ticker="KXEARNINGSMENTIONNVDA-26MAY20",
    )

    assert len(contracts) == 1
    assert contracts[0].yes_bid == Decimal("0.22")
    assert contracts[0].yes_ask == Decimal("0.29")


def test_public_client_fetches_event_mention_markets_without_auth_headers():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        assert request.url.path.endswith("/events/CAVA-26Q1/markets")
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "CAVA-26Q1-MARGIN",
                        "event_ticker": "CAVA-26Q1",
                        "title": "Will CAVA mention margin during earnings?",
                        "rules_primary": "If margin is said by any CAVA representative.",
                        "yes_bid": 40,
                        "yes_ask": 44,
                    }
                ]
            },
        )

    client = KalshiPublicClient(httpx.Client(transport=httpx.MockTransport(handler)))

    contracts = client.get_event_mention_markets("CAVA-26Q1")

    assert "kalshi-access-key" not in seen_headers
    assert contracts[0].target_phrase.normalized_phrase == "margin"


def test_public_client_falls_back_to_markets_query_when_event_endpoint_404():
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path.endswith("/events/KXEARNINGSMENTIONNVDA-26MAY20/markets"):
            return httpx.Response(404, json={"error": {"code": "not_found"}})
        if request.url.path.endswith("/markets"):
            assert request.url.params.get("event_ticker") == "KXEARNINGSMENTIONNVDA-26MAY20"
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONNVDA-26MAY20-H20",
                            "event_ticker": "KXEARNINGSMENTIONNVDA-26MAY20",
                            "title": "What will NVIDIA Corporation say during their next earnings call?",
                            "rules_primary": "If H20 is said by any NVIDIA representative.",
                            "yes_bid_dollars": "0.20",
                            "yes_ask_dollars": "0.30",
                        }
                    ]
                },
            )
        return httpx.Response(500, json={"error": "unexpected path"})

    client = KalshiPublicClient(httpx.Client(transport=httpx.MockTransport(handler)))

    contracts = client.get_event_mention_markets("KXEARNINGSMENTIONNVDA-26MAY20")

    assert contracts[0].target_phrase.normalized_phrase == "h20"
    assert contracts[0].yes_bid == Decimal("0.20")
    assert any(path.endswith("/events/KXEARNINGSMENTIONNVDA-26MAY20/markets") for path in seen_paths)
    assert any(path.endswith("/markets") for path in seen_paths)


def test_public_client_fetches_historical_market_page_with_query_params():
    seen_url = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = request.url
        return httpx.Response(
            200,
            json={
                "cursor": "next-cursor",
                "markets": [
                    {
                        "ticker": "CAVA-26Q1-TRAFFIC",
                        "event_ticker": "CAVA-26Q1",
                        "title": "Will CAVA mention traffic during earnings?",
                        "rules_primary": "If traffic is said by any CAVA representative.",
                        "yes_bid": 38,
                        "yes_ask": 45,
                    }
                ],
            },
        )

    client = KalshiPublicClient(httpx.Client(transport=httpx.MockTransport(handler)))

    payload = client.get_historical_markets(status="closed", search="CAVA", limit=10)

    assert seen_url is not None
    assert seen_url.params["status"] == "closed"
    assert seen_url.params["search"] == "CAVA"
    assert seen_url.params["limit"] == "10"
    assert payload["cursor"] == "next-cursor"
