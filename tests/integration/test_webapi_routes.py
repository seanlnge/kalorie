from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from kalorie.webapi.kalshi_service import WebMentionMarket
from kalorie.webapi.main import create_app
from kalorie.webapi.run_store import EventScope, RunStore


class StubKalshiService:
    def __init__(self) -> None:
        self._markets = [
            WebMentionMarket(
                market_ticker="KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL",
                event_ticker="KXEARNINGSMENTIONWMT-26Q2",
                title="Will WMT mention omnichannel during earnings?",
                target_phrase="Omnichannel",
                company_symbol="WMT",
                yes_bid=Decimal("0.34"),
                yes_ask=Decimal("0.53"),
                volume=999,
            )
        ]

    def list_open_mention_markets(self) -> list[WebMentionMarket]:
        return self._markets

    def list_event_mention_markets(self, event_ticker: str) -> list[WebMentionMarket]:
        return [market for market in self._markets if market.event_ticker == event_ticker]


def _build_client(tmp_path: Path) -> TestClient:
    app = create_app(run_root=tmp_path / "runs" / "web", kalshi_service=StubKalshiService())
    return TestClient(app)


def test_get_open_markets_returns_stub_data(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/api/markets/open")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["markets"]) == 1
    assert payload["markets"][0]["market_ticker"] == "KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL"
    assert len(payload["events"]) == 1
    assert payload["events"][0]["event_ticker"] == "KXEARNINGSMENTIONWMT-26Q2"
    assert payload["events"][0]["representative_market_ticker"] == "KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL"
    assert payload["events"][0]["representative_phrase"] == "Omnichannel"


def test_post_job_requires_idempotency_key(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    response = client.post("/api/markets/KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL/jobs")

    assert response.status_code == 400


def test_get_event_markets_returns_contracts_for_event(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/api/events/KXEARNINGSMENTIONWMT-26Q2/markets")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["markets"]) == 1
    assert payload["markets"][0]["event_ticker"] == "KXEARNINGSMENTIONWMT-26Q2"
    assert payload["markets"][0]["target_phrase"] == "Omnichannel"


def test_latest_run_endpoint_returns_newest_completed(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    store = RunStore(root=tmp_path / "runs" / "web")
    scope = EventScope(company_symbol="WMT", event_key="KXEARNINGSMENTIONWMT-26Q2")
    run = store.create_run(
        scope=scope,
        market_ticker="KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL",
        created_at=datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC),
        options={},
    )
    store.update_status(scope=scope, run_id=run.run_id, status="completed")

    response = client.get("/api/markets/KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL/runs/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["run_id"] == run.run_id


def test_get_run_endpoint_returns_result_payload(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    store = RunStore(root=tmp_path / "runs" / "web")
    scope = EventScope(company_symbol="WMT", event_key="KXEARNINGSMENTIONWMT-26Q2")
    run = store.create_run(
        scope=scope,
        market_ticker="KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL",
        created_at=datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC),
        options={},
    )
    (run.run_dir / "result.json").write_text(
        '{"rows":[{"target_phrase":"omnichannel","probability":"0.57"}]}',
        encoding="utf-8",
    )
    store.update_status(scope=scope, run_id=run.run_id, status="completed")

    response = client.get(
        f"/api/markets/KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL/runs/{run.run_id}"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["run_id"] == run.run_id
    assert payload["result"]["rows"][0]["target_phrase"] == "omnichannel"
