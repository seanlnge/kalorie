import json
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from kalorie.webapi.kalshi_service import WebMentionMarket
from kalorie.webapi.main import create_app


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


def _build_client(tmp_path: Path) -> TestClient:
    app = create_app(run_root=tmp_path / "runs" / "web", kalshi_service=StubKalshiService())
    return TestClient(app)


def test_job_run_writes_cutoff_and_cache_manifest(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        "/api/markets/KXEARNINGSMENTIONWMT-26Q2-OMNICHANNEL/jobs",
        data={"decision_cutoff_ts": "2026-05-20T22:00:00+00:00"},
        headers={"Idempotency-Key": "k1"},
    )
    assert response.status_code == 200

    run = response.json()["run"]
    run_dir = Path(run["run_dir"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["effective_decision_cutoff_ts"] == "2026-05-20T22:00:00+00:00"
    assert isinstance(summary["cache_reused"], bool)

    cache_manifest = (
        tmp_path
        / "runs"
        / "web"
        / "events"
        / "WMT"
        / "KXEARNINGSMENTIONWMT-26Q2"
        / "data"
        / "cache_manifest.json"
    )
    assert cache_manifest.exists()
    payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    assert payload["pipeline_version"]
    assert payload["cutoff_ts"] == "2026-05-20T22:00:00+00:00"
