import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from kalorie2.market_poller import (
    ActiveMarketRow,
    MarketPollCacheStore,
    MarketPollSnapshot,
    PollPredictionRow,
)
from kalorie2.webapi.main import create_app


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_api_bundle(root: Path, *, side: str = "NO") -> Path:
    model_dir = root / "unit-model"
    (model_dir / "runtime").mkdir(parents=True)
    (model_dir / "training").mkdir()
    (model_dir / "README.md").write_text("# Unit Model\n\nAPI summary.", encoding="utf-8")
    _write_json(
        model_dir / "artifacts" / "model.json",
        {
            "model_name": "unit-model",
            "model_type": "market_anchored_linear_residual",
            "training_summary": {"row_count": 1, "event_count": 1, "feature_count": 2},
            "model": {"weights": {"alpha": 0.1}},
        },
    )
    _write_json(
        model_dir / "artifacts" / "feature-schema.json",
        {"feature_names": ["alpha", "beta"]},
    )
    _write_json(
        model_dir / "artifacts" / "training-manifest.json",
        {"training_corpus": {"saved_csv": "training/rows.csv"}},
    )
    _write_json(model_dir / "artifacts" / "evaluation-reports.json", {})
    with (model_dir / "training" / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["market_ticker", "event_ticker"])
        writer.writeheader()
        writer.writerow({"market_ticker": "MARKET-1", "event_ticker": "EVENT-1"})
    (model_dir / "runtime" / "model_runtime.py").write_text(
        "\n".join(
            [
                "import argparse, csv, json",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model-dir')",
                "parser.add_argument('--csv')",
                "parser.add_argument('--row-index', type=int)",
                "args = parser.parse_args()",
                "with open(args.csv, newline='', encoding='utf-8') as handle:",
                "    row = list(csv.DictReader(handle))[args.row_index]",
                "print(json.dumps({",
                "    'market_ticker': row['market_ticker'],",
                "    'event_ticker': row['event_ticker'],",
                "    'probability': 0.42,",
                "    'market_probability': 0.50,",
                "    'residual_delta': -0.32,",
                f"    'trade_decision': {{'side': '{side}', 'cost': 0.48, 'edge': 0.07}},",
                "}))",
            ]
        ),
        encoding="utf-8",
    )
    return model_dir


def test_model_list_and_detail_endpoints_return_saved_model_metadata(tmp_path: Path) -> None:
    _write_api_bundle(tmp_path)
    client = TestClient(create_app(models_root=tmp_path))

    list_response = client.get("/api/models")
    detail_response = client.get("/api/models/unit-model")

    assert list_response.status_code == 200
    assert list_response.json()["models"][0]["name"] == "unit-model"
    assert list_response.json()["models"][0]["training"]["row_count"] == 1
    assert detail_response.status_code == 200
    assert detail_response.json()["model"]["readme_summary"] == "API summary."
    assert (
        detail_response.json()["model"]["artifact_paths"]["runtime"]
        == "runtime/model_runtime.py"
    )


def test_risk_presets_endpoint_returns_available_policy_overlays(tmp_path: Path) -> None:
    client = TestClient(create_app(models_root=tmp_path))

    response = client.get("/api/risk-presets")

    assert response.status_code == 200
    presets = response.json()["risk_presets"]
    assert [preset["id"] for preset in presets] == [
        "capital_preservation",
        "balanced",
        "growth",
    ]
    assert presets[1]["min_margin"] > 0


def test_sample_rows_endpoint_returns_training_csv_preview(tmp_path: Path) -> None:
    _write_api_bundle(tmp_path)
    client = TestClient(create_app(models_root=tmp_path))

    response = client.get("/api/models/unit-model/sample-rows")

    assert response.status_code == 200
    assert response.json()["rows"] == [
        {"row_index": "0", "market_ticker": "MARKET-1", "event_ticker": "EVENT-1"}
    ]


def test_score_endpoint_scores_uploaded_csv_and_filters_no_only(tmp_path: Path) -> None:
    _write_api_bundle(tmp_path, side="NO")
    client = TestClient(create_app(models_root=tmp_path))
    csv_body = "market_ticker,event_ticker\nMARKET-1,EVENT-1\n"

    response = client.post(
        "/api/models/unit-model/score",
        data={"row_index": "0", "execution_mode": "no_only"},
        files={"csv_file": ("rows.csv", csv_body, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_mode"] == "no_only"
    assert payload["rows"][0]["market_ticker"] == "MARKET-1"
    assert payload["rows"][0]["model_probability"] == 0.42
    assert payload["rows"][0]["side"] == "NO"
    assert payload["rows"][0]["edge"] == 0.07


def test_score_endpoint_no_only_filters_non_no_trade(tmp_path: Path) -> None:
    _write_api_bundle(tmp_path, side="YES")
    client = TestClient(create_app(models_root=tmp_path))
    csv_body = "market_ticker,event_ticker\nMARKET-1,EVENT-1\n"

    response = client.post(
        "/api/models/unit-model/score",
        data={"row_index": "0", "execution_mode": "no_only"},
        files={"csv_file": ("rows.csv", csv_body, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["rows"] == []


def test_live_poll_endpoint_returns_latest_cached_snapshot(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    _write_poll_snapshot(cache_root)
    client = TestClient(create_app(models_root=tmp_path / "models", poll_cache_root=cache_root))

    response = client.get("/api/polls/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot"]["poll_id"] == "20260526-040000"
    assert payload["snapshot"]["prediction_rows"][0]["market_ticker"] == "MARKET-1"


def test_live_trades_endpoint_returns_latest_cached_trade_rows(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    _write_poll_snapshot(cache_root)
    client = TestClient(create_app(models_root=tmp_path / "models", poll_cache_root=cache_root))

    response = client.get("/api/trades/latest")

    assert response.status_code == 200
    assert response.json()["trades"][0]["side"] == "NO"


def test_poll_history_endpoint_returns_recent_cached_snapshots_newest_first(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    _write_poll_snapshot(cache_root, poll_id="20260526-040000", edge=0.06)
    _write_poll_snapshot(cache_root, poll_id="20260526-041000", edge=0.09)
    client = TestClient(create_app(models_root=tmp_path / "models", poll_cache_root=cache_root))

    response = client.get("/api/polls/history")

    assert response.status_code == 200
    payload = response.json()
    assert [snapshot["poll_id"] for snapshot in payload["snapshots"]] == [
        "20260526-041000",
        "20260526-040000",
    ]
    assert payload["snapshots"][0]["trade_rows"][0]["edge"] == 0.09


def test_live_poll_endpoint_returns_404_before_first_poll(tmp_path: Path) -> None:
    client = TestClient(
        create_app(models_root=tmp_path / "models", poll_cache_root=tmp_path / "cache")
    )

    response = client.get("/api/polls/latest")

    assert response.status_code == 404


def test_current_markets_endpoint_scores_latest_active_market_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_api_bundle(tmp_path, side="NO")

    class FakeActiveMarketSource:
        def __init__(self, **_: object) -> None:
            pass

        def list_active_markets(self) -> list[ActiveMarketRow]:
            return [
                ActiveMarketRow(
                    market_ticker="MARKET-1",
                    event_ticker="EVENT-1",
                    series_ticker="KXEARNINGSMENTIONTEST",
                    event_datetime="2026-05-26T04:00:00Z",
                    event_title="Event",
                    market_title="Market",
                    target_phrase="AI",
                    yes_bid=0.31,
                    yes_ask=0.39,
                    yes_mid=0.35,
                    volume=100,
                )
            ]

    class FakeScorer:
        def __init__(self, *, models_root: Path) -> None:
            self.models_root = models_root

        def score_active_markets(
            self, markets: list[ActiveMarketRow], *, model_name: str
        ) -> list[PollPredictionRow]:
            market = markets[0]
            return [
                PollPredictionRow(
                    market_ticker=market.market_ticker,
                    event_ticker=market.event_ticker,
                    event_datetime=market.event_datetime,
                    target_phrase=market.target_phrase,
                    model_name=model_name,
                    model_probability=0.25,
                    market_probability=market.yes_mid,
                    yes_bid=market.yes_bid,
                    yes_ask=market.yes_ask,
                    residual_delta=-0.1,
                    side="NO",
                    edge=0.03,
                    cost=0.58,
                    volume=market.volume,
                )
            ]

    monkeypatch.setattr("kalorie2.webapi.main.KalshiActiveMarketSource", FakeActiveMarketSource)
    monkeypatch.setattr("kalorie2.webapi.main.CachedSavedModelMarketScorer", FakeScorer)
    client = TestClient(create_app(models_root=tmp_path))

    response = client.post("/api/models/unit-model/current-markets?risk_preset_id=balanced")

    assert response.status_code == 200
    payload = response.json()["snapshot"]
    assert payload["model_name"] == "unit-model"
    assert payload["risk_preset_id"] == "balanced"
    assert payload["market_count"] == 1
    assert payload["prediction_rows"][0]["market_ticker"] == "MARKET-1"
    assert payload["prediction_rows"][0]["event_datetime"] == "2026-05-26T04:00:00Z"
    assert payload["prediction_rows"][0]["risk_preset_id"] == "balanced"
    assert payload["prediction_rows"][0]["recommended_fraction"] > 0


def test_current_markets_endpoint_accepts_custom_risk_preset_body(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_api_bundle(tmp_path, side="NO")

    class FakeActiveMarketSource:
        def __init__(self, **_: object) -> None:
            pass

        def list_active_markets(self) -> list[ActiveMarketRow]:
            return [
                ActiveMarketRow(
                    market_ticker="MARKET-1",
                    event_ticker="EVENT-1",
                    series_ticker="KXEARNINGSMENTIONTEST",
                    event_title="Event",
                    market_title="Market",
                    target_phrase="AI",
                    yes_bid=0.42,
                    yes_ask=0.45,
                    yes_mid=0.435,
                    volume=100,
                )
            ]

    class FakeScorer:
        def __init__(self, *, models_root: Path) -> None:
            self.models_root = models_root

        def score_active_markets(
            self, markets: list[ActiveMarketRow], *, model_name: str
        ) -> list[PollPredictionRow]:
            market = markets[0]
            return [
                PollPredictionRow(
                    market_ticker=market.market_ticker,
                    event_ticker=market.event_ticker,
                    target_phrase=market.target_phrase,
                    model_name=model_name,
                    model_probability=0.35,
                    market_probability=market.yes_mid,
                    yes_bid=market.yes_bid,
                    yes_ask=market.yes_ask,
                    residual_delta=-0.08,
                    side="NONE",
                    edge=0,
                    cost=0,
                    volume=market.volume,
                )
            ]

    monkeypatch.setattr("kalorie2.webapi.main.KalshiActiveMarketSource", FakeActiveMarketSource)
    monkeypatch.setattr("kalorie2.webapi.main.CachedSavedModelMarketScorer", FakeScorer)
    client = TestClient(create_app(models_root=tmp_path))

    response = client.post(
        "/api/models/unit-model/current-markets",
        json={
            "risk_preset": {
                "id": "custom-tight",
                "label": "Custom Tight",
                "description": "Custom preset",
                "trade_side": "no_only",
                "min_margin": 0.02,
                "kelly_fraction": 0.25,
                "max_position_fraction": 0.02,
                "max_event_exposure_fraction": 0.08,
                "risk_of_ruin_estimate": 0.01,
                "risk_of_ruin_label": "Low",
            }
        },
    )

    assert response.status_code == 200
    row = response.json()["snapshot"]["prediction_rows"][0]
    assert row["risk_preset_id"] == "custom-tight"
    assert row["recommended_fraction"] == 0.02


def _write_poll_snapshot(
    cache_root: Path,
    *,
    poll_id: str = "20260526-040000",
    edge: float = 0.06,
) -> None:
    row = PollPredictionRow(
        market_ticker="MARKET-1",
        event_ticker="EVENT-1",
        target_phrase="AI",
        model_name="unit-model",
        model_probability=0.31,
        market_probability=0.39,
        yes_bid=0.37,
        yes_ask=0.4,
        residual_delta=-0.29,
        side="NO",
        edge=edge,
        cost=0.63,
        volume=123,
    )
    MarketPollCacheStore(root=cache_root).write_snapshot(
        MarketPollSnapshot(
            poll_id=poll_id,
            model_name="unit-model",
            started_at=datetime(2026, 5, 26, 4, 0, tzinfo=UTC),
            completed_at=datetime(2026, 5, 26, 4, 0, 1, tzinfo=UTC),
            market_count=1,
            prediction_count=1,
            trade_count=1,
            prediction_rows=[row],
            trade_rows=[row],
        )
    )
