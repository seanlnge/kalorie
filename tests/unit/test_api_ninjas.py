from datetime import date
from pathlib import Path

import httpx
import pytest

from kalorie.clients.api_ninjas import ApiNinjasClient, VendorAuthError, VendorRateLimitError


def test_api_key_is_sent_only_as_required_header():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        assert "api-secret" not in str(request.url)
        return httpx.Response(200, json=[])

    client = ApiNinjasClient(
        api_key="api-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    client.get_earnings_calendar("CAVA", date(2026, 5, 1), date(2026, 5, 31))

    assert seen_headers["x-api-key"] == "api-secret"


def test_earnings_calendar_response_maps_to_events():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "ticker": "CAVA",
                    "fiscal_year": 2026,
                    "fiscal_quarter": 1,
                    "date": "2026-05-19",
                }
            ],
        )

    client = ApiNinjasClient("key", httpx.Client(transport=httpx.MockTransport(handler)))

    events = client.get_earnings_calendar("CAVA", date(2026, 5, 1), date(2026, 5, 31))

    assert events[0].company_symbol == "CAVA"
    assert events[0].fiscal_year == 2026
    assert events[0].fiscal_quarter == 1
    assert events[0].event_date == date(2026, 5, 19)


def test_transcript_response_maps_to_document_and_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ticker": "CAVA",
                "date": "2026-05-19T20:00:00Z",
                "transcript": "Operator: Welcome.\nTraffic improved in Q1.",
            },
        )

    client = ApiNinjasClient("key", httpx.Client(transport=httpx.MockTransport(handler)))

    document, chunks = client.get_transcript("CAVA", 2026, 1)

    assert document.document_type == "earnings_call_transcript"
    assert document.company_symbol == "CAVA"
    assert document.content_hash
    assert chunks[0].document_id == document.source_id
    assert "Traffic improved" in chunks[0].text


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, VendorAuthError), (429, VendorRateLimitError)],
)
def test_vendor_errors_are_typed(status_code: int, error_type: type[Exception]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "nope"})

    client = ApiNinjasClient("key", httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(error_type):
        client.get_earnings_calendar("CAVA", date(2026, 5, 1), date(2026, 5, 31))


def test_client_does_not_load_env(monkeypatch: pytest.MonkeyPatch):
    def fail_on_env_read(*args, **kwargs):
        raise AssertionError("unit client must not load .env files")

    monkeypatch.setattr(Path, "read_text", fail_on_env_read)
    ApiNinjasClient(
        "key",
        httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200))),
    )
