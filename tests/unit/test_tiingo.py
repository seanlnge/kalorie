import httpx
import pytest

from kalorie.clients.tiingo import (
    TiingoAuthError,
    TiingoNewsClient,
    TiingoRateLimitError,
)


def test_search_news_sends_token_header_and_parses_rows():
    seen_header = None
    seen_params = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_header, seen_params
        seen_header = request.headers.get("Authorization")
        seen_params = request.url.params
        return httpx.Response(
            200,
            json=[
                {
                    "id": 123,
                    "title": "Walmart earnings outlook rises",
                    "url": "https://example.com/story",
                    "description": "Description",
                    "publishedDate": "2024-05-15T12:00:00+00:00",
                    "crawlDate": "2024-05-15T13:00:00+00:00",
                    "source": "Reuters",
                    "tickers": ["WMT"],
                    "tags": ["earnings"],
                }
            ],
        )

    client = TiingoNewsClient(
        api_key="tiingo-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    rows = client.search_news(
        ticker="WMT",
        start_date="2024-01-01",
        end_date="2024-12-31",
        limit=10,
    )

    assert seen_header == "Token tiingo-secret"
    assert seen_params is not None
    assert seen_params["tickers"] == "WMT"
    assert seen_params["startDate"] == "2024-01-01"
    assert rows[0].source_name == "Reuters"
    assert rows[0].tickers == ["WMT"]


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, TiingoAuthError), (403, TiingoAuthError), (429, TiingoRateLimitError)],
)
def test_search_news_raises_typed_vendor_errors(status_code: int, error_type: type[Exception]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "error"})

    client = TiingoNewsClient(
        api_key="tiingo-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(error_type):
        client.search_news(
            ticker="WMT",
            start_date="2024-01-01",
            end_date="2024-12-31",
            limit=10,
        )
