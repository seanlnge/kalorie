from datetime import UTC, datetime

import httpx

from kalorie.clients.kalshi import KalshiEarningsMarketsClient


def test_list_company_mention_markets_filters_to_requested_symbol():
    seen_search_values: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/markets")
        seen_search_values.append(str(request.url.params.get("search")))
        cursor = request.url.params.get("cursor")
        if cursor == "next-page":
            return httpx.Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONWMT-26MAY15-OMNI",
                            "event_ticker": "KXEARNINGSMENTIONWMT-26MAY15",
                            "title": "What will Walmart say during their next earnings call?",
                            "rules_primary": "If omnichannel is said by any Walmart representative.",
                            "yes_bid": 41,
                            "yes_ask": 46,
                            "observed_at": "2026-05-15T13:30:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "cursor": "next-page",
                "markets": [
                    {
                        "ticker": "KXEARNINGSMENTIONWMT-26MAY15-GROC",
                        "event_ticker": "KXEARNINGSMENTIONWMT-26MAY15",
                        "title": "What will Walmart say during their next earnings call?",
                        "rules_primary": "If grocery is said by any Walmart representative.",
                        "yes_bid": 40,
                        "yes_ask": 44,
                        "observed_at": "2026-05-15T13:00:00Z",
                    },
                    {
                        "ticker": "KXEARNINGSMENTIONNVDA-26MAY20-H20",
                        "event_ticker": "KXEARNINGSMENTIONNVDA-26MAY20",
                        "title": "What will NVIDIA say during their next earnings call?",
                        "rules_primary": "If h20 is said by any NVIDIA representative.",
                        "yes_bid": 50,
                        "yes_ask": 56,
                        "observed_at": "2026-05-20T13:00:00Z",
                    },
                ],
            },
        )

    client = KalshiEarningsMarketsClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    contracts = client.list_company_mention_markets("WMT", status="closed")

    assert len(contracts) == 2
    assert {contract.target_phrase.normalized_phrase for contract in contracts} == {
        "grocery",
        "omnichannel",
    }
    assert all(contract.event_ticker.startswith("KXEARNINGSMENTIONWMT") for contract in contracts)
    assert seen_search_values[0] == "KXEARNINGSMENTIONWMT"


def test_get_latest_company_event_ticker_uses_most_recent_market_timestamp():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXEARNINGSMENTIONWMT-26MAY15-GROC",
                        "event_ticker": "KXEARNINGSMENTIONWMT-26MAY15",
                        "title": "What will Walmart say during their next earnings call?",
                        "rules_primary": "If grocery is said by any Walmart representative.",
                        "yes_bid": 40,
                        "yes_ask": 44,
                        "observed_at": "2026-05-15T13:00:00Z",
                    },
                    {
                        "ticker": "KXEARNINGSMENTIONWMT-26AUG21-OMNI",
                        "event_ticker": "KXEARNINGSMENTIONWMT-26AUG21",
                        "title": "What will Walmart say during their next earnings call?",
                        "rules_primary": "If omnichannel is said by any Walmart representative.",
                        "yes_bid": 45,
                        "yes_ask": 49,
                        "observed_at": "2026-08-21T13:00:00Z",
                    },
                ]
            },
        )

    client = KalshiEarningsMarketsClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    latest = client.get_latest_company_event_ticker("WMT", status="closed")
    assert latest == "KXEARNINGSMENTIONWMT-26AUG21"

    events = client.list_company_event_tickers("WMT", status="closed")
    assert events == ["KXEARNINGSMENTIONWMT-26AUG21", "KXEARNINGSMENTIONWMT-26MAY15"]
    assert datetime.fromisoformat("2026-08-21T13:00:00+00:00").tzinfo == UTC
