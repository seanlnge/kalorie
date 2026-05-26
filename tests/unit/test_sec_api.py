import httpx
import pytest

from kalorie.clients.sec_api import (
    SecApiAuthError,
    SecApiClient,
    SecApiRateLimitError,
    select_best_company_mapping,
)


def test_sec_api_query_sends_token_as_query_param_and_parses_filings():
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json={
                "filings": [
                    {
                        "ticker": "WMT",
                        "cik": "104169",
                        "filedAt": "2025-08-21T12:00:00-04:00",
                        "documentFormatFiles": [
                            {
                                "type": "EX-99.1",
                                "documentUrl": "https://www.sec.gov/exhibit.htm",
                            },
                            {
                                "type": "EX-99.2",
                                "description": "EARNINGS PRESENTATION",
                                "documentUrl": "https://www.sec.gov/presentation.htm",
                            }
                        ],
                    }
                ]
            },
        )

    client = SecApiClient(
        api_key="sec-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    filings = client.query_ex99_1_filings(
        query='formType:"8-K" AND cik:104169 AND documentFormatFiles.type:"EX-99.1"',
        size=10,
    )

    assert seen_request is not None
    assert seen_request.url.params["token"] == "sec-secret"
    assert "sec-secret" not in seen_request.content.decode("utf-8")
    assert filings[0].ticker == "WMT"
    assert filings[0].exhibit_url == "https://www.sec.gov/exhibit.htm"
    assert [exhibit.document_type for exhibit in filings[0].exhibits] == [
        "EX-99.1",
        "EX-99.2",
    ]


def test_sec_api_auth_and_rate_limit_errors():
    auth_client = SecApiClient(
        api_key="bad",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(401))
        ),
    )
    with pytest.raises(SecApiAuthError):
        auth_client.query_ex99_1_filings(query="formType:\"8-K\"")

    rate_client = SecApiClient(
        api_key="rate",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(429))
        ),
    )
    with pytest.raises(SecApiRateLimitError):
        rate_client.query_ex99_1_filings(query="formType:\"8-K\"")


def test_sec_api_skips_rows_without_ex99_exhibits():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "filings": [
                    {
                        "ticker": "WMT",
                        "cik": "104169",
                        "filedAt": "2025-01-01T00:00:00-05:00",
                        "documentFormatFiles": [{"type": "8-K", "documentUrl": "x"}],
                    },
                    {
                        "ticker": "WMT",
                        "cik": "104169",
                        "filedAt": "2025-02-01T00:00:00-05:00",
                        "documentFormatFiles": [
                            {"type": "EX-99", "documentUrl": "https://www.sec.gov/ex99.htm"}
                        ],
                    },
                ]
            },
        )

    client = SecApiClient(
        api_key="sec-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    filings = client.query_ex99_1_filings(query="formType:\"8-K\"")

    assert len(filings) == 1
    assert filings[0].exhibit_url == "https://www.sec.gov/ex99.htm"


def test_sec_api_mapping_resolves_best_operating_company_without_secret_body():
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json=[
                {
                    "name": "MICROSOFT CORP",
                    "ticker": "MSFT",
                    "cik": "789019",
                    "isDelisted": False,
                    "category": "Domestic Common Stock",
                },
                {
                    "name": "MICROSOFT ETF",
                    "ticker": "MSFX",
                    "cik": "1771146",
                    "isDelisted": False,
                    "category": "ETF",
                },
            ],
        )

    client = SecApiClient(
        api_key="sec-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    mappings = client.resolve_mapping(resolve_by="name", value="Microsoft")

    assert seen_request is not None
    assert seen_request.url.path == "/mapping/name/Microsoft"
    assert seen_request.url.params["token"] == "sec-secret"
    assert seen_request.content == b""
    assert "sec-secret" not in str(mappings)
    assert select_best_company_mapping(mappings).ticker == "MSFT"
    assert select_best_company_mapping(mappings).cik == "789019"
