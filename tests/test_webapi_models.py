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


def _write_risk_trial_bundle(root: Path) -> Path:
    model_dir = _write_api_bundle(root, side="NONE")
    with (model_dir / "training" / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "market_ticker",
            "event_ticker",
            "close_time",
            "final_outcome",
            "preclose_yes_bid",
            "preclose_yes_ask",
            "preclose_yes_mid",
            "probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "market_ticker": "MKT-YES-WIN",
                    "event_ticker": "EVENT-1",
                    "close_time": "2026-05-20T12:00:00Z",
                    "final_outcome": "yes",
                    "preclose_yes_bid": "0.40",
                    "preclose_yes_ask": "0.45",
                    "preclose_yes_mid": "0.425",
                    "probability": "0.70",
                },
                {
                    "market_ticker": "MKT-NO-WIN",
                    "event_ticker": "EVENT-2",
                    "close_time": "2026-05-21T12:00:00Z",
                    "final_outcome": "no",
                    "preclose_yes_bid": "0.55",
                    "preclose_yes_ask": "0.60",
                    "preclose_yes_mid": "0.575",
                    "probability": "0.25",
                },
                {
                    "market_ticker": "MKT-NONE",
                    "event_ticker": "EVENT-3",
                    "close_time": "2026-05-22T12:00:00Z",
                    "final_outcome": "yes",
                    "preclose_yes_bid": "0.48",
                    "preclose_yes_ask": "0.52",
                    "preclose_yes_mid": "0.50",
                    "probability": "0.51",
                },
            ]
        )
    (model_dir / "runtime" / "model_runtime.py").write_text(
        "\n".join(
            [
                "def load_model(model_dir):",
                "    return {}",
                "",
                "def load_web_evidence(model_dir):",
                "    return {}",
                "",
                "def score_row(row, model, web_evidence_by_event):",
                "    probability = float(row['probability'])",
                "    market_probability = float(row['preclose_yes_mid'])",
                "    return {",
                "        'market_ticker': row['market_ticker'],",
                "        'event_ticker': row['event_ticker'],",
                "        'probability': probability,",
                "        'market_probability': market_probability,",
                "        'residual_delta': probability - market_probability,",
                "        'trade_decision': {'side': 'NONE', 'cost': 0, 'edge': 0},",
                "    }",
            ]
        ),
        encoding="utf-8",
    )
    return model_dir


def test_model_list_and_detail_endpoints_return_saved_model_metadata(tmp_path: Path) -> None:
    _write_api_bundle(tmp_path)
    client = TestClient(create_app(models_root=tmp_path, env_path=tmp_path / ".env.missing"))

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
    client = TestClient(create_app(models_root=tmp_path, env_path=tmp_path / ".env.missing"))

    response = client.get("/api/risk-presets")

    assert response.status_code == 200
    presets = response.json()["risk_presets"]
    assert [preset["id"] for preset in presets] == [
        "capital_preservation",
        "balanced",
        "growth",
    ]
    assert presets[1]["min_margin"] > 0


def test_custom_risk_trial_endpoint_computes_metrics_from_saved_rows(tmp_path: Path) -> None:
    _write_risk_trial_bundle(tmp_path)
    client = TestClient(create_app(models_root=tmp_path))

    response = client.post(
        "/api/models/unit-model/risk-trial",
        json={
            "risk_preset": {
                "id": "custom-open",
                "label": "Custom Open",
                "description": "custom",
                "trade_side": "all",
                "min_margin": 0.05,
                "kelly_fraction": 0.5,
                "max_position_fraction": 0.05,
                "max_event_exposure_fraction": 0.1,
            }
        },
    )

    assert response.status_code == 200
    trial = response.json()["trial"]
    assert trial["risk_preset_id"] == "custom-open"
    assert trial["trade_count"] == 2
    assert trial["market_count"] == 3
    assert trial["trade_percent"] == 2 / 3
    assert trial["ev_per_10_markets"] > 0
    assert trial["risk_of_ruin_estimate"] >= 0
    assert trial["expected_return_per_market"]["expected"] > 0


def test_account_summary_endpoint_uses_paper_bankroll_without_auth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_API_KEY", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    client = TestClient(create_app(models_root=tmp_path, env_path=tmp_path / ".env.missing"))

    response = client.get("/api/account/summary")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["available"] is False
    assert summary["source"] == "paper"
    assert summary["bankroll"] == 100.0


def test_account_summary_endpoint_uses_authenticated_balance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAccountClient:
        @classmethod
        def from_env(cls, *, http_client: object) -> "FakeAccountClient":
            return cls()

        def get_balance(self) -> dict[str, object]:
            return {"balance": {"portfolio_value": 240_00, "available_balance": 180_00}}

        def list_positions(self) -> dict[str, object]:
            return {"market_positions": [{"ticker": "A", "market_exposure": 12_00}]}

    monkeypatch.setattr("kalorie2.webapi.main.KalshiAccountClient", FakeAccountClient)
    client = TestClient(create_app(models_root=tmp_path))

    response = client.get("/api/account/summary")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["available"] is True
    assert summary["source"] == "kalshi"
    assert summary["portfolio_value"] == 240.0
    assert summary["free_cash"] == 180.0
    assert summary["position_exposure"] == 12.0
    assert summary["bankroll"] == 180.0


def test_account_positions_endpoint_returns_normalized_open_positions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAccountClient:
        @classmethod
        def from_env(cls, *, http_client: object) -> "FakeAccountClient":
            return cls()

        def list_positions(self) -> dict[str, object]:
            return {
                "market_positions": [
                    {
                        "ticker": "MKT-YES",
                        "position": 5,
                        "market_exposure": 250,
                        "market_value": 310,
                        "average_price": 50,
                    }
                ]
            }

    monkeypatch.setattr("kalorie2.webapi.main.KalshiAccountClient", FakeAccountClient)
    client = TestClient(create_app(models_root=tmp_path))

    response = client.get("/api/account/positions")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["available"] is True
    assert summary["open_position_count"] == 1
    assert summary["total_contracts"] == 5
    assert summary["total_exposure"] == 2.5
    assert summary["positions"][0]["market_ticker"] == "MKT-YES"


def test_account_positions_endpoint_surfaces_position_fetch_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAccountClient:
        @classmethod
        def from_env(cls, *, http_client: object) -> "FakeAccountClient":
            return cls()

        def list_positions(self) -> dict[str, object]:
            raise RuntimeError("positions unavailable")

    monkeypatch.setattr("kalorie2.webapi.main.KalshiAccountClient", FakeAccountClient)
    client = TestClient(create_app(models_root=tmp_path))

    response = client.get("/api/account/positions")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["available"] is False
    assert summary["source"] == "kalshi"
    assert "positions unavailable" in summary["error"]


def test_account_summary_keeps_balance_when_positions_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeAccountClient:
        @classmethod
        def from_env(cls, *, http_client: object) -> "FakeAccountClient":
            return cls()

        def get_balance(self) -> dict[str, object]:
            return {"balance": {"portfolio_value": 240_00, "available_balance": 180_00}}

        def list_positions(self) -> dict[str, object]:
            raise RuntimeError("positions unavailable")

    monkeypatch.setattr("kalorie2.webapi.main.KalshiAccountClient", FakeAccountClient)
    client = TestClient(create_app(models_root=tmp_path))

    response = client.get("/api/account/summary")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["available"] is True
    assert summary["source"] == "kalshi"
    assert summary["free_cash"] == 180.0
    assert summary["bankroll"] == 180.0
    assert summary["position_exposure"] is None


def test_create_app_loads_env_file_for_account_auth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("KALSHI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("KALSHI_API_KEY", raising=False)

    class FakeAccountClient:
        @classmethod
        def from_env(cls, *, http_client: object) -> "FakeAccountClient | None":
            if "KALSHI_API_KEY" not in __import__("os").environ:
                return None
            return cls()

        def get_balance(self) -> dict[str, object]:
            return {"balance": {"available_balance": 120_00}}

        def list_positions(self) -> dict[str, object]:
            return {"market_positions": []}

    monkeypatch.setattr("kalorie2.webapi.main.KalshiAccountClient", FakeAccountClient)
    client = TestClient(create_app(models_root=tmp_path, env_path=env_path))

    response = client.get("/api/account/summary")

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["available"] is True
    assert summary["free_cash"] == 120.0


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
            }
        },
    )

    assert response.status_code == 200
    row = response.json()["snapshot"]["prediction_rows"][0]
    assert row["risk_preset_id"] == "custom-tight"
    assert row["recommended_fraction"] == 0.02


def test_current_markets_reuses_cached_predictions_when_only_risk_changes(
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
        calls = 0

        def __init__(self, *, models_root: Path) -> None:
            self.models_root = models_root

        def score_active_markets(
            self, markets: list[ActiveMarketRow], *, model_name: str
        ) -> list[PollPredictionRow]:
            type(self).calls += 1
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

    first_response = client.post("/api/models/unit-model/current-markets?risk_preset_id=balanced")
    second_response = client.post(
        "/api/models/unit-model/current-markets",
        json={
            "risk_preset": {
                "id": "custom-wide",
                "label": "Custom Wide",
                "description": "Custom preset",
                "trade_side": "no_only",
                "min_margin": 0.2,
                "kelly_fraction": 0.25,
                "max_position_fraction": 0.02,
                "max_event_exposure_fraction": 0.08,
            }
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert FakeScorer.calls == 1
    second_snapshot = second_response.json()["snapshot"]
    assert second_snapshot["risk_preset_id"] == "custom-wide"
    assert second_snapshot["prediction_rows"][0]["risk_preset_id"] == "custom-wide"


def test_current_markets_can_reapply_risk_without_refetching_kalshi_markets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_api_bundle(tmp_path, side="NO")

    class FakeActiveMarketSource:
        calls = 0

        def __init__(self, **_: object) -> None:
            pass

        def list_active_markets(self) -> list[ActiveMarketRow]:
            type(self).calls += 1
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
        calls = 0

        def __init__(self, *, models_root: Path) -> None:
            self.models_root = models_root

        def score_active_markets(
            self, markets: list[ActiveMarketRow], *, model_name: str
        ) -> list[PollPredictionRow]:
            type(self).calls += 1
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

    first_response = client.post("/api/models/unit-model/current-markets?risk_preset_id=balanced")
    second_response = client.post(
        "/api/models/unit-model/current-markets?refresh_markets=false",
        json={
            "risk_preset": {
                "id": "custom-wide",
                "label": "Custom Wide",
                "description": "Custom preset",
                "trade_side": "no_only",
                "min_margin": 0.2,
                "kelly_fraction": 0.25,
                "max_position_fraction": 0.02,
                "max_event_exposure_fraction": 0.08,
            }
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert FakeActiveMarketSource.calls == 1
    assert FakeScorer.calls == 1


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
