from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from kalorie.clients.kalshi import (
    KalshiAuthorizedClient,
    KalshiParseError,
    KalshiPublicClient,
)


class FakeSigner:
    def __init__(self):
        self.calls = []

    def sign_headers(self, method: str, path: str) -> dict[str, str]:
        self.calls.append((method, path))
        return {"KALSHI-ACCESS-SIGNATURE": "fake-signature"}


def test_public_market_fetch_maps_snapshot_without_credentials():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(
            200,
            json={
                "market": {
                    "ticker": "CAVA-TRAFFIC",
                    "title": "Will CAVA mention traffic during earnings?",
                    "yes_bid": 38,
                    "yes_ask": 45,
                    "observed_at": "2026-05-19T14:30:00Z",
                }
            },
        )

    client = KalshiPublicClient(httpx.Client(transport=httpx.MockTransport(handler)))

    snapshot = client.get_market("CAVA-TRAFFIC")

    assert "kalshi-access-key" not in seen_headers
    assert snapshot.market_id == "CAVA-TRAFFIC"
    assert snapshot.yes_bid == Decimal("0.38")
    assert snapshot.yes_ask == Decimal("0.45")
    assert snapshot.observed_at == datetime(2026, 5, 19, 14, 30, tzinfo=UTC)


def test_public_market_parse_errors_include_market_id():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"market": {"ticker": "CAVA-TRAFFIC", "title": "bad", "yes_bid": 38}},
        )

    client = KalshiPublicClient(httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(KalshiParseError, match="CAVA-TRAFFIC"):
        client.get_market("CAVA-TRAFFIC")


def test_public_client_fetches_market_candlesticks_for_filing_plus_five_window():
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json={
                "ticker": "KXEARNINGSMENTIONCAVA-26MAY19-TARI",
                "candlesticks": [
                    {
                        "end_period_ts": 1779222000,
                        "yes_bid": {"close_dollars": "0.6400"},
                        "yes_ask": {"close_dollars": "0.6900"},
                        "price": {"close_dollars": "0.6600"},
                        "volume_fp": "10.00",
                        "open_interest_fp": "126.74",
                    }
                ],
            },
        )

    client = KalshiPublicClient(httpx.Client(transport=httpx.MockTransport(handler)))

    payload = client.get_market_candlesticks(
        series_ticker="KXEARNINGSMENTION",
        market_id="KXEARNINGSMENTIONCAVA-26MAY19-TARI",
        start_ts=1779221700,
        end_ts=1779222000,
        period_interval=1,
    )

    assert seen_request is not None
    assert (
        seen_request.url.path
        == "/trade-api/v2/series/KXEARNINGSMENTION/markets/"
        "KXEARNINGSMENTIONCAVA-26MAY19-TARI/candlesticks"
    )
    assert seen_request.url.params["period_interval"] == "1"
    assert payload["candlesticks"][0]["price"]["close_dollars"] == "0.6600"


def test_public_client_retries_429_with_bounded_backoff():
    attempts = 0
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"code": "rate_limited"}})
        return httpx.Response(200, json={"markets": []})

    client = KalshiPublicClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        retry_sleep=sleep_calls.append,
    )

    payload = client.get_historical_markets(status="closed", search="KXEARNINGSMENTION", limit=10)

    assert payload == {"markets": []}
    assert attempts == 2
    assert sleep_calls
    assert max(sleep_calls) <= 8.0


def test_event_markets_fallback_paginates_by_event_ticker():
    seen_cursors: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events/KXEARNINGSMENTIONNVDA-26MAY20/markets"):
            return httpx.Response(404, json={"error": {"code": "not_found"}})
        if request.url.path.endswith("/markets"):
            seen_cursors.append(request.url.params.get("cursor"))
            assert request.url.params["event_ticker"] == "KXEARNINGSMENTIONNVDA-26MAY20"
            if request.url.params.get("cursor") == "page-2":
                return httpx.Response(
                    200,
                    json={
                        "markets": [
                            {
                                "ticker": "KXEARNINGSMENTIONNVDA-26MAY20-COSM",
                                "event_ticker": "KXEARNINGSMENTIONNVDA-26MAY20",
                                "title": (
                                    "What will NVIDIA Corporation say during their next "
                                    "earnings call?"
                                ),
                                "rules_primary": "If Cosmos is said by any NVIDIA representative.",
                                "yes_bid": 86,
                                "yes_ask": 87,
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "cursor": "page-2",
                    "markets": [
                        {
                            "ticker": "KXEARNINGSMENTIONNVDA-26MAY20-H20",
                            "event_ticker": "KXEARNINGSMENTIONNVDA-26MAY20",
                            "title": (
                                "What will NVIDIA Corporation say during their next "
                                "earnings call?"
                            ),
                            "rules_primary": "If H20 is said by any NVIDIA representative.",
                            "yes_bid": 63,
                            "yes_ask": 64,
                        }
                    ],
                },
            )
        return httpx.Response(500, json={"error": "unexpected path"})

    client = KalshiPublicClient(httpx.Client(transport=httpx.MockTransport(handler)))

    contracts = client.get_event_mention_markets("KXEARNINGSMENTIONNVDA-26MAY20")

    assert [contract.target_phrase.normalized_phrase for contract in contracts] == ["h20", "cosmos"]
    assert seen_cursors == [None, "page-2"]


def test_authorized_client_requires_key_id_and_existing_private_key_path(tmp_path: Path):
    key_path = tmp_path / "kalshi.key"
    key_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="KALSHI_API_KEY_ID"):
        KalshiAuthorizedClient(
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda req: httpx.Response(200))
            ),
            key_id=None,
            private_key_path=key_path,
            signer=FakeSigner(),
        )

    with pytest.raises(ValueError, match="does not exist"):
        KalshiAuthorizedClient(
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda req: httpx.Response(200))
            ),
            key_id="key-id",
            private_key_path=tmp_path / "missing.key",
            signer=FakeSigner(),
        )


def test_authorized_client_delegates_signing_without_reading_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    key_path = tmp_path / "kalshi.key"
    key_path.write_text("", encoding="utf-8")
    signer = FakeSigner()
    seen_headers = {}

    def fail_on_read(*args, **kwargs):
        raise AssertionError("private key content must not be read")

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(
            200,
            json={
                "market": {
                    "ticker": "CAVA-TRAFFIC",
                    "title": "Will CAVA mention traffic during earnings?",
                    "yes_bid": "0.38",
                    "yes_ask": "0.45",
                    "observed_at": "2026-05-19T14:30:00Z",
                }
            },
        )

    monkeypatch.setattr(Path, "read_text", fail_on_read)
    monkeypatch.setattr(Path, "read_bytes", fail_on_read)
    client = KalshiAuthorizedClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        key_id="key-id",
        private_key_path=key_path,
        signer=signer,
    )

    snapshot = client.get_market("CAVA-TRAFFIC")

    assert snapshot.yes_ask == Decimal("0.45")
    assert signer.calls == [("GET", "/markets/CAVA-TRAFFIC")]
    assert seen_headers["kalshi-access-key"] == "key-id"
    assert seen_headers["kalshi-access-signature"] == "fake-signature"
