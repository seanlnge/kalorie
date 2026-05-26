from datetime import date

import httpx
import pytest

from kalorie.clients.financial_modeling_prep import (
    FinancialModelingPrepClient,
    FmpApiAuthError,
    FmpApiRateLimitError,
    FmpApiSubscriptionError,
)


def test_fmp_api_key_is_sent_as_query_param():
    seen_url = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = str(request.url)
        return httpx.Response(200, json=[])

    client = FinancialModelingPrepClient(
        api_key="fmp-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    client.get_transcript_dates("CAVA")

    assert "apikey=fmp-secret" in seen_url
    assert "x-api-key" not in seen_url.lower()


def test_get_transcript_dates_maps_rows():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"symbol": "CAVA", "year": 2026, "quarter": 1, "date": "2026-05-28"},
                {"symbol": "CAVA", "fiscalYear": 2025, "fiscalQuarter": 4},
            ],
        )

    client = FinancialModelingPrepClient(
        "key",
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    refs = client.get_transcript_dates("CAVA")

    assert len(refs) == 2
    assert refs[0].symbol == "CAVA"
    assert refs[0].fiscal_year == 2026
    assert refs[0].fiscal_quarter == 1
    assert refs[0].published_at.date() == date(2026, 5, 28)
    assert refs[1].fiscal_year == 2025
    assert refs[1].fiscal_quarter == 4


def test_get_transcript_text_accepts_list_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"content": "Operator: Hello\nTraffic improved.", "date": "2026-05-28T21:00:00Z"}],
        )

    client = FinancialModelingPrepClient(
        "key",
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    transcript, published_at = client.get_transcript_text(
        symbol="CAVA", fiscal_year=2026, fiscal_quarter=1
    )

    assert "Traffic improved" in transcript
    assert published_at.isoformat().startswith("2026-05-28T21:00:00")


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, FmpApiAuthError), (402, FmpApiSubscriptionError), (429, FmpApiRateLimitError)],
)
def test_fmp_vendor_errors_are_typed(status_code: int, error_type: type[Exception]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "nope"})

    client = FinancialModelingPrepClient(
        "key",
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(error_type):
        client.get_transcript_dates("CAVA")
