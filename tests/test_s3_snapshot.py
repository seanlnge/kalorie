from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pytest
from moto import mock_aws

from kalorie2.market_poller import ActiveMarketRow, PollPredictionRow
from kalorie2.s3_snapshot import (
    build_snapshot_payload,
    filter_non_past_markets,
    handler_from_event,
    load_openai_api_key_from_secrets_manager,
    put_snapshot_json,
    run_snapshot,
    snapshot_id_for,
)


def test_snapshot_id_for_uses_utc_hour() -> None:
    now = datetime(2026, 8, 12, 0, 5, 12, tzinfo=UTC)
    assert snapshot_id_for(now) == "2026081200"


def test_filter_non_past_markets_drops_prior_event_days() -> None:
    past = ActiveMarketRow(
        market_ticker="KXEARNINGSMENTIONBULL-26JUL24-AI",
        event_ticker="KXEARNINGSMENTIONBULL-26JUL24",
        series_ticker="KXEARNINGSMENTIONBULL",
        event_datetime="2026-07-24T00:00:00+00:00",
        event_title="Bull",
        market_title="AI",
        target_phrase="AI",
        yes_bid=0.3,
        yes_ask=0.4,
        yes_mid=0.35,
    )
    today = ActiveMarketRow(
        market_ticker="KXEARNINGSMENTIONAAPL-26AUG12-AI",
        event_ticker="KXEARNINGSMENTIONAAPL-26AUG12",
        series_ticker="KXEARNINGSMENTIONAAPL",
        event_datetime="2026-08-12T15:00:00+00:00",
        event_title="Apple",
        market_title="AI",
        target_phrase="AI",
        yes_bid=0.3,
        yes_ask=0.4,
        yes_mid=0.35,
    )
    kept = filter_non_past_markets(
        [past, today],
        now=datetime(2026, 8, 12, 2, 0, tzinfo=UTC),
    )
    assert [row.market_ticker for row in kept] == [today.market_ticker]


def test_build_snapshot_payload_exposes_delta_not_trades() -> None:
    markets = [
        ActiveMarketRow(
            market_ticker="KXEARNINGSMENTIONAAPL-26APR30-AI",
            event_ticker="KXEARNINGSMENTIONAAPL-26APR30",
            series_ticker="KXEARNINGSMENTIONAAPL",
            event_title="What will Apple say during their next earnings call?",
            market_title="AI",
            target_phrase="AI",
            yes_bid=0.37,
            yes_ask=0.4,
            yes_mid=0.385,
            volume=10,
        )
    ]
    predictions = [
        PollPredictionRow(
            market_ticker=markets[0].market_ticker,
            event_ticker=markets[0].event_ticker,
            event_title=markets[0].event_title,
            target_phrase="AI",
            model_name="kalorie-v6",
            model_probability=0.3,
            market_probability=0.385,
            yes_bid=0.37,
            yes_ask=0.4,
            residual_delta=-0.1,
            side="NO",
            edge=0.05,
            cost=0.63,
            volume=10,
        ),
        PollPredictionRow(
            market_ticker=markets[0].market_ticker,
            event_ticker=markets[0].event_ticker,
            event_title=markets[0].event_title,
            target_phrase="AI",
            model_name="kalorie-v6",
            model_probability=0.39,
            market_probability=0.385,
            yes_bid=0.37,
            yes_ask=0.4,
            residual_delta=0.01,
            side="NONE",
            edge=0.0,
            cost=0.0,
            volume=10,
        ),
    ]
    payload = build_snapshot_payload(
        snapshot_id="2026081200",
        generated_at=datetime(2026, 8, 12, 0, 5, 12, tzinfo=UTC),
        model_name="kalorie-v6",
        markets=markets,
        predictions=predictions,
    )
    assert payload["snapshot_id"] == "2026081200"
    assert payload["market_count"] == 1
    assert payload["prediction_count"] == 2
    assert "trade_count" not in payload
    assert payload["markets"][0]["market_ticker"] == markets[0].market_ticker
    assert payload["predictions"][0]["delta"] == -0.1
    assert payload["predictions"][0]["residual_delta"] == -0.1
    assert payload["predictions"][0]["abs_delta"] == 0.1
    assert "side" not in payload["predictions"][0]
    assert "edge" not in payload["predictions"][0]
    assert "cost" not in payload["predictions"][0]


@mock_aws
def test_put_snapshot_json_writes_hour_key() -> None:
    bucket = "kalorie-snapshots-test"
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=bucket)
    key = put_snapshot_json(
        bucket=bucket,
        snapshot_id="2026081200",
        payload={"snapshot_id": "2026081200", "markets": []},
        s3_client=client,
    )
    assert key == "2026081200.json"
    obj = client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    assert '"snapshot_id": "2026081200"' in body


@mock_aws
def test_run_snapshot_writes_s3_without_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bucket = "kalorie-snapshots-test"
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=bucket)

    market = ActiveMarketRow(
        market_ticker="KXEARNINGSMENTIONAAPL-26APR30-AI",
        event_ticker="KXEARNINGSMENTIONAAPL-26APR30",
        series_ticker="KXEARNINGSMENTIONAAPL",
        event_title="What will Apple say during their next earnings call?",
        market_title="AI",
        target_phrase="AI",
        yes_bid=0.37,
        yes_ask=0.4,
        yes_mid=0.385,
        volume=10,
    )
    prediction = PollPredictionRow(
        market_ticker=market.market_ticker,
        event_ticker=market.event_ticker,
        target_phrase="AI",
        model_name="kalorie-v6",
        model_probability=0.3,
        market_probability=0.385,
        yes_bid=0.37,
        yes_ask=0.4,
        residual_delta=-0.1,
        side="NO",
        edge=0.05,
        cost=0.63,
        volume=10,
    )

    class StubSource:
        def list_active_markets(self) -> list[ActiveMarketRow]:
            return [market]

    class StubScorer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def score_active_markets(
            self,
            markets: list[ActiveMarketRow],
            *,
            model_name: str,
        ) -> list[PollPredictionRow]:
            assert model_name == "kalorie-v6"
            assert markets == [market]
            return [prediction]

    monkeypatch.setattr(
        "kalorie2.s3_snapshot.KalshiActiveMarketSource",
        lambda **kwargs: StubSource(),
    )
    monkeypatch.setattr("kalorie2.s3_snapshot.CachedSavedModelMarketScorer", StubScorer)
    monkeypatch.setattr(
        "kalorie2.s3_snapshot.OpenAIWebEvidenceSource",
        lambda **kwargs: object(),
    )

    result = run_snapshot(
        model_name="kalorie-v6",
        models_root=tmp_path,
        bucket=bucket,
        live_web_evidence=True,
        now=datetime(2026, 8, 12, 6, 1, 0, tzinfo=UTC),
        s3_client=client,
        http_client=object(),  # type: ignore[arg-type]
    )
    assert result["key"] == "2026081206.json"
    assert result["prediction_count"] == 1
    stored = client.get_object(Bucket=bucket, Key="2026081206.json")
    body = stored["Body"].read()
    assert b'"delta": -0.1' in body
    assert b'"trade_count"' not in body
    latest = client.get_object(Bucket=bucket, Key="latest.json")
    assert b'"snapshot_id": "2026081206"' in latest["Body"].read()


@mock_aws
def test_load_openai_api_key_from_secrets_manager() -> None:
    client = boto3.client("secretsmanager", region_name="us-east-1")
    created = client.create_secret(
        Name="kalorie-openai",
        SecretString=json.dumps({"OPENAI_API_KEY": "sk-test-key"}),
    )
    assert load_openai_api_key_from_secrets_manager(created["ARN"]) == "sk-test-key"


@mock_aws
def test_handler_from_event_requires_bucket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SNAPSHOT_BUCKET", raising=False)
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="SNAPSHOT_BUCKET"):
        handler_from_event({})


@mock_aws
def test_handler_from_event_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bucket = "kalorie-snapshots-test"
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=bucket)
    monkeypatch.setenv("SNAPSHOT_BUCKET", bucket)
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path))
    monkeypatch.setenv("MODEL_NAME", "kalorie-v6")
    monkeypatch.setenv("LIVE_WEB_EVIDENCE", "false")

    class StubSource:
        def list_active_markets(self) -> list[ActiveMarketRow]:
            return []

    class StubScorer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert kwargs.get("web_evidence_source") is None

        def score_active_markets(
            self,
            markets: list[ActiveMarketRow],
            *,
            model_name: str,
        ) -> list[PollPredictionRow]:
            return []

    monkeypatch.setattr(
        "kalorie2.s3_snapshot.KalshiActiveMarketSource",
        lambda **kwargs: StubSource(),
    )
    monkeypatch.setattr("kalorie2.s3_snapshot.CachedSavedModelMarketScorer", StubScorer)
    monkeypatch.setattr("kalorie2.s3_snapshot.boto3.client", lambda *a, **k: client)

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return datetime(2026, 8, 12, 12, 0, tzinfo=UTC)

    monkeypatch.setattr("kalorie2.s3_snapshot.datetime", FakeDateTime)

    result = handler_from_event({})
    assert result["key"] == "2026081212.json"
    assert result["market_count"] == 0
