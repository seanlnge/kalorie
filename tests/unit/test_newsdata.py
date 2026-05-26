from datetime import date

import httpx
import pytest

from kalorie.clients.newsdata import (
    NewsDataAuthError,
    NewsDataClient,
    NewsDataRateLimitError,
    NewsDataSubscriptionError,
)


def test_search_archive_maps_articles_and_next_page():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.params["apikey"] == "news-secret"
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "results": [
                        {
                            "article_id": "a1",
                            "title": "NVIDIA earnings analysis",
                            "pubDate": "2026-05-20 08:00:00",
                            "source_name": "Example",
                            "source_priority": 2,
                            "datatype": "analysis",
                        }
                    ],
                    "nextPage": "next",
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "success",
                "results": [
                    {
                        "article_id": "a2",
                        "title": "NVIDIA relevant update",
                        "pubDate": "2026-05-19 08:00:00",
                    }
                ],
            },
        )

    client = NewsDataClient(
        api_key="news-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    rows = client.search_archive(
        query="NVDA",
        from_date=date(2026, 5, 1).isoformat(),
        to_date=date(2026, 5, 20).isoformat(),
        max_articles=5,
    )

    assert len(rows) == 2
    assert rows[0].source_priority == 2
    assert rows[0].datatype == "analysis"


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, NewsDataAuthError), (403, NewsDataAuthError), (429, NewsDataRateLimitError)],
)
def test_search_archive_raises_typed_errors(status_code: int, error_type: type[Exception]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"status": "error"})

    client = NewsDataClient(
        api_key="news-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(error_type):
        client.search_archive(
            query="NVDA",
            from_date="2026-05-01",
            to_date="2026-05-20",
        )


def test_search_archive_raises_subscription_error_on_access_denied():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "status": "error",
                "results": {"code": "AccessDenied", "message": "upgrade your plan"},
            },
        )

    client = NewsDataClient(
        api_key="news-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(NewsDataSubscriptionError):
        client.search_archive(
            query="NVDA",
            from_date="2026-05-01",
            to_date="2026-05-20",
        )


def test_search_latest_reads_articles():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "success",
                "results": [
                    {
                        "article_id": "x1",
                        "title": "NVIDIA latest earnings analysis",
                        "pubDate": "2026-05-20 09:00:00",
                    }
                ],
            },
        )

    client = NewsDataClient(
        api_key="news-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    rows = client.search_latest(query="NVDA", max_articles=5)
    assert len(rows) == 1
    assert rows[0].article_id == "x1"
