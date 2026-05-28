import csv
import json
import math
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from typer.testing import CliRunner

from kalorie2 import prediction_cli
from kalorie2.prediction_cli import app
from kalorie2.prediction_types import PredictionInputRow, prediction_row_key
from kalorie2.residual_engine import ResidualPrediction


def _write_market_csv(path: Path) -> None:
    fieldnames = [
        "market_ticker",
        "event_ticker",
        "series_ticker",
        "market_category",
        "event_phrase",
        "market_name",
        "word_said",
        "normalized_word_said",
        "final_outcome",
        "status",
        "close_time",
        "snapshot_target_time",
        "preclose_yes_bid",
        "preclose_yes_ask",
        "preclose_yes_mid",
        "candle_end_ts",
        "snapshot_staleness_seconds",
        "settlement_ts",
        "source",
    ]
    rows = [
        {
            "market_ticker": "EVENT1-TARI",
            "event_ticker": "EVENT1",
            "series_ticker": "KXEARNINGSMENTIONDE",
            "market_category": "earnings",
            "event_phrase": "What will John Deere say during their next earnings call?",
            "market_name": "What will John Deere say during their next earnings call? - Tariff",
            "word_said": "Tariff",
            "normalized_word_said": "tariff",
            "final_outcome": "yes",
            "status": "",
            "close_time": "2026-01-01T20:00:00Z",
            "snapshot_target_time": "2026-01-01T12:00:00Z",
            "preclose_yes_bid": "0.49",
            "preclose_yes_ask": "0.51",
            "preclose_yes_mid": "0.50",
            "candle_end_ts": "1767279600",
            "snapshot_staleness_seconds": "0",
            "settlement_ts": "",
            "source": "unit_test",
        },
        {
            "market_ticker": "EVENT2-TARI",
            "event_ticker": "EVENT2",
            "series_ticker": "KXEARNINGSMENTIONDE",
            "market_category": "earnings",
            "event_phrase": "What will John Deere say during their next earnings call?",
            "market_name": "What will John Deere say during their next earnings call? - Tariff",
            "word_said": "Tariff",
            "normalized_word_said": "tariff",
            "final_outcome": "no",
            "status": "",
            "close_time": "2026-01-02T20:00:00Z",
            "snapshot_target_time": "2026-01-02T12:00:00Z",
            "preclose_yes_bid": "0.49",
            "preclose_yes_ask": "0.51",
            "preclose_yes_mid": "0.50",
            "candle_end_ts": "1767366000",
            "snapshot_staleness_seconds": "0",
            "settlement_ts": "",
            "source": "unit_test",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_row(**overrides) -> PredictionInputRow:
    payload = {
        "market_ticker": "EVENT2-TARI",
        "event_ticker": "EVENT2",
        "series_ticker": "KXEARNINGSMENTIONDE",
        "market_category": "earnings",
        "event_phrase": "What will John Deere say during their next earnings call?",
        "market_name": "What will John Deere say during their next earnings call? - Tariff",
        "word_said": "Tariff",
        "normalized_word_said": "tariff",
        "final_outcome": "yes",
        "status": None,
        "close_time": datetime(2026, 1, 2, 20, tzinfo=UTC),
        "snapshot_target_time": datetime(2026, 1, 2, 12, tzinfo=UTC),
        "preclose_yes_bid": Decimal("0.20"),
        "preclose_yes_ask": Decimal("0.30"),
        "preclose_yes_mid": Decimal("0.25"),
        "candle_end_ts": 1767369600,
        "snapshot_staleness_seconds": 0,
        "settlement_ts": None,
        "source": "unit_test",
    }
    payload.update(overrides)
    return PredictionInputRow.model_validate(payload)


def test_build_trades_uses_row_key_for_duplicate_market_ticker_snapshots():
    early = _prediction_row(
        snapshot_target_time=datetime(2026, 1, 1, 20, tzinfo=UTC),
        preclose_yes_ask=Decimal("0.30"),
        preclose_yes_mid=Decimal("0.25"),
    )
    late = _prediction_row(
        snapshot_target_time=datetime(2026, 1, 2, 18, tzinfo=UTC),
        preclose_yes_ask=Decimal("0.70"),
        preclose_yes_mid=Decimal("0.65"),
    )
    predictions = [
        ResidualPrediction(
            row_key=prediction_row_key(early),
            market_ticker=early.market_ticker,
            event_ticker=early.event_ticker,
            probability=Decimal("0.80"),
            market_probability=early.preclose_yes_mid,
            residual_delta=0.1,
        ),
        ResidualPrediction(
            row_key=prediction_row_key(late),
            market_ticker=late.market_ticker,
            event_ticker=late.event_ticker,
            probability=Decimal("0.90"),
            market_probability=late.preclose_yes_mid,
            residual_delta=0.1,
        ),
    ]

    trades = prediction_cli._build_trades([early, late], predictions, margin=0.0)

    assert [trade["cost"] for trade in trades] == [0.3, 0.7]


def test_summarize_predictions_reports_log_loss_and_ece():
    rows = [
        _prediction_row(
            final_outcome="yes",
            preclose_yes_bid=Decimal("0.70"),
            preclose_yes_ask=Decimal("0.80"),
            preclose_yes_mid=Decimal("0.75"),
        ),
        _prediction_row(
            market_ticker="EVENT2-COST",
            word_said="Cost",
            normalized_word_said="cost",
            final_outcome="no",
        ),
    ]
    predictions = [
        ResidualPrediction(
            row_key=prediction_row_key(rows[0]),
            market_ticker=rows[0].market_ticker,
            event_ticker=rows[0].event_ticker,
            probability=Decimal("0.80"),
            market_probability=Decimal("0.75"),
            residual_delta=0.1,
        ),
        ResidualPrediction(
            row_key=prediction_row_key(rows[1]),
            market_ticker=rows[1].market_ticker,
            event_ticker=rows[1].event_ticker,
            probability=Decimal("0.20"),
            market_probability=Decimal("0.25"),
            residual_delta=-0.1,
        ),
    ]

    summary = prediction_cli._summarize_predictions(rows, predictions)

    assert summary["log_loss"] == round(-math.log(0.8), 6)
    assert summary["market_log_loss"] == round(-math.log(0.75), 6)
    assert summary["ece"] == 0.2
    assert summary["market_ece"] == 0.25


def test_summarize_predictions_reports_event_weighted_and_collapsed_edges():
    good_event = _prediction_row(
        market_ticker="EVENT1-TARI",
        event_ticker="EVENT1",
        final_outcome="yes",
        preclose_yes_mid=Decimal("0.50"),
        snapshot_target_time=datetime(2026, 1, 1, 12, tzinfo=UTC),
    )
    bad_snapshot_1 = _prediction_row(
        market_ticker="EVENT2-TARI",
        event_ticker="EVENT2",
        final_outcome="no",
        preclose_yes_mid=Decimal("0.50"),
        snapshot_target_time=datetime(2026, 1, 2, 12, tzinfo=UTC),
    )
    bad_snapshot_2 = _prediction_row(
        market_ticker="EVENT2-TARI",
        event_ticker="EVENT2",
        final_outcome="no",
        preclose_yes_mid=Decimal("0.50"),
        snapshot_target_time=datetime(2026, 1, 2, 13, tzinfo=UTC),
    )
    bad_snapshot_3 = _prediction_row(
        market_ticker="EVENT2-TARI",
        event_ticker="EVENT2",
        final_outcome="no",
        preclose_yes_mid=Decimal("0.50"),
        snapshot_target_time=datetime(2026, 1, 2, 14, tzinfo=UTC),
    )
    rows = [good_event, bad_snapshot_1, bad_snapshot_2, bad_snapshot_3]
    predictions = [
        ResidualPrediction(
            row_key=prediction_row_key(row),
            market_ticker=row.market_ticker,
            event_ticker=row.event_ticker,
            probability=Decimal("0.90"),
            market_probability=row.preclose_yes_mid,
            residual_delta=0.1,
        )
        for row in rows
    ]

    summary = prediction_cli._summarize_predictions(rows, predictions)

    assert summary["brier_score"] == 0.61
    assert summary["event_weighted_brier_score"] == 0.41
    assert summary["snapshot_collapsed_brier_score"] == 0.41
    assert summary["brier_edge_vs_market"] == -0.36
    assert summary["event_weighted_brier_edge_vs_market"] == -0.16


def test_summarize_predictions_reports_row_quality_without_filtering_zero_microstructure():
    fresh_tight = _prediction_row(
        snapshot_staleness_seconds=30 * 60,
        preclose_yes_bid=Decimal("0.49"),
        preclose_yes_ask=Decimal("0.51"),
        preclose_yes_mid=Decimal("0.50"),
        preclose_volume=0,
        preclose_open_interest=0,
        preclose_yes_bid_size=0,
        preclose_yes_ask_size=0,
    )
    stale_wide = _prediction_row(
        market_ticker="EVENT2-COST",
        word_said="Cost",
        normalized_word_said="cost",
        snapshot_staleness_seconds=5 * 60 * 60,
        preclose_yes_bid=Decimal("0.20"),
        preclose_yes_ask=Decimal("0.35"),
        preclose_yes_mid=Decimal("0.275"),
        preclose_volume=0,
        preclose_open_interest=0,
        preclose_yes_bid_size=0,
        preclose_yes_ask_size=0,
    )
    predictions = [
        ResidualPrediction(
            row_key=prediction_row_key(row),
            market_ticker=row.market_ticker,
            event_ticker=row.event_ticker,
            probability=Decimal("0.50"),
            market_probability=row.preclose_yes_mid,
            residual_delta=0.0,
        )
        for row in [fresh_tight, stale_wide]
    ]

    summary = prediction_cli._summarize_predictions([fresh_tight, stale_wide], predictions)

    assert summary["prediction_count"] == 2
    assert summary["row_quality"]["row_count"] == 2
    assert summary["row_quality"]["stale_over_4h_count"] == 1
    assert summary["row_quality"]["wide_spread_over_10c_count"] == 1
    assert summary["row_quality"]["microstructure_volume_present_count"] == 0
    assert summary["row_quality"]["buckets"]["fresh_tight"]["count"] == 1
    assert summary["row_quality"]["buckets"]["stale_wide"]["count"] == 1


def test_sweep_ranking_prefers_probability_metrics_over_roi():
    bad_probability_high_roi = {
        "evaluation": {"log_loss": 0.9, "brier_score": 0.30, "ece": 0.20},
        "backtest": {"roi_on_cost": 1.0},
    }
    good_probability_low_roi = {
        "evaluation": {"log_loss": 0.4, "brier_score": 0.16, "ece": 0.05},
        "backtest": {"roi_on_cost": -0.1},
    }

    ranked = prediction_cli._rank_sweep_results(
        [bad_probability_high_roi, good_probability_low_roi]
    )

    assert ranked[0] is good_probability_low_roi


def test_evaluate_cli_writes_walk_forward_predictions_outside_full(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--min-training-events",
            "1",
            "--epochs",
            "5",
        ],
    )

    assert result.exit_code == 0
    report = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    model_summary = json.loads((out_dir / "model-summary.json").read_text(encoding="utf-8"))
    run_config = json.loads((out_dir / "run-config.json").read_text(encoding="utf-8"))
    predictions = list(csv.DictReader((out_dir / "predictions.csv").open(encoding="utf-8")))

    assert report["summary"]["prediction_count"] == 1
    assert model_summary["model_type"] == "linear_residual_walk_forward"
    assert run_config["run_id"] == "unit"
    assert predictions[0]["event_ticker"] == "EVENT2"
    assert predictions[0]["training_event_tickers"] == "EVENT1"


def test_evaluate_cli_loads_web_evidence_packets(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    web_evidence_dir = tmp_path / "web-evidence"
    web_evidence_dir.mkdir()
    _write_market_csv(input_csv)
    (web_evidence_dir / "EVENT2.json").write_text(
        json.dumps(
            {
                "event_ticker": "EVENT2",
                "company_name": "John Deere",
                "cutoff_time": "2026-01-02T12:00:00Z",
                "items": [
                    {
                        "title": "Tariff pressure before earnings",
                        "url": "https://example.com/tariff",
                        "source": "Example News",
                        "published_at": "2026-01-02T10:00:00Z",
                        "snippet": "Deere tariff cost pressure before earnings.",
                        "target_phrases": ["tariff"],
                        "evidence_strength": 0.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--web-evidence-dir",
            str(web_evidence_dir),
            "--min-training-events",
            "1",
            "--epochs",
            "5",
        ],
    )

    feature_matrix = json.loads((out_dir / "feature-matrix.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert feature_matrix["web_evidence_count"] == 1
    assert feature_matrix["feature_rows"][1]["web_evidence_available"] == 1.0


def test_evaluate_cli_can_run_no_side_residual_training(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--min-training-events",
            "1",
            "--epochs",
            "5",
            "--target-side",
            "no",
            "--positive-label-weight",
            "2.0",
        ],
    )

    model_summary = json.loads((out_dir / "model-summary.json").read_text(encoding="utf-8"))
    predictions = list(csv.DictReader((out_dir / "predictions.csv").open(encoding="utf-8")))

    assert result.exit_code == 0
    assert model_summary["target_side"] == "no"
    assert model_summary["positive_label_weight"] == 2.0
    assert "target_side:no" in predictions[0]["reasons"]


def test_evaluate_cli_can_ablate_named_feature_groups(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--min-training-events",
            "1",
            "--epochs",
            "5",
            "--feature-ablation-group",
            "semantic",
        ],
    )

    model_summary = json.loads((out_dir / "model-summary.json").read_text(encoding="utf-8"))
    feature_matrix = json.loads((out_dir / "feature-matrix.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert model_summary["feature_ablation_group"] == "semantic"
    assert not any(
        key.startswith("phrase_semantic_")
        for key in feature_matrix["feature_rows"][0]
    )


def test_backtest_cli_can_run_no_side_residual_training(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit-backtest"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "backtest",
            str(input_csv),
            "--run-id",
            "unit-backtest",
            "--out-dir",
            str(out_dir),
            "--min-training-events",
            "1",
            "--epochs",
            "5",
            "--target-side",
            "no",
            "--positive-label-weight",
            "2.0",
            "--trade-side",
            "no_only",
        ],
    )

    report = json.loads((out_dir / "backtest.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "summary" in report
    assert "trades" in report


def test_evaluate_cli_rejects_generated_outputs_in_full_directory(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(input_csv),
            "--run-id",
            "bad",
            "--out-dir",
            str(tmp_path / "artifacts" / "full"),
        ],
    )

    assert result.exit_code != 0
    assert "artifacts/full" in str(result.exception)


def test_evaluate_cli_can_exclude_events_from_manifest(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    manifest = tmp_path / "exclusion-manifest.json"
    _write_market_csv(input_csv)
    manifest.write_text(json.dumps({"event_tickers": ["EVENT2"]}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--exclude-events-manifest",
            str(manifest),
            "--min-training-events",
            "1",
        ],
    )

    report = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert report["summary"]["prediction_count"] == 0


def test_collect_web_evidence_dry_run_writes_payloads_outside_full(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "collect-web-evidence",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--model",
            "gpt-5.5",
            "--max-events",
            "1",
            "--dry-run",
        ],
    )

    request_payload = json.loads(
        (out_dir / "web-evidence" / "requests" / "EVENT1.json").read_text(encoding="utf-8")
    )

    assert result.exit_code == 0
    assert request_payload["tools"] == [{"type": "web_search"}]
    assert request_payload["text"]["format"]["strict"] is True
    assert not (out_dir / "web-evidence" / "packets" / "EVENT1.json").exists()


def test_collect_web_evidence_dry_run_defaults_to_mini_model(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "collect-web-evidence",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--max-events",
            "1",
            "--dry-run",
        ],
    )

    request_payload = json.loads(
        (out_dir / "web-evidence" / "requests" / "EVENT1.json").read_text(encoding="utf-8")
    )

    assert result.exit_code == 0
    assert request_payload["model"] == "gpt-5.4-mini"


def test_collect_web_evidence_live_uses_configurable_timeout(
    tmp_path: Path,
    monkeypatch,
):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)
    captured: dict[str, float] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "id": "resp_test",
                "model": "gpt-5.5",
                "output_text": json.dumps(
                    {
                        "event_ticker": "EVENT1",
                        "company_name": "John Deere",
                        "cutoff_time": "2026-01-01T12:00:00Z",
                        "items": [],
                    }
                ),
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            }

    class FakeClient:
        def __init__(self, *, timeout: float):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(prediction_cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(
        app,
        [
            "collect-web-evidence",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--model",
            "gpt-5.5",
            "--max-events",
            "1",
            "--no-dry-run",
            "--request-timeout-seconds",
            "900",
        ],
    )

    packet = json.loads(
        (out_dir / "web-evidence" / "packets" / "EVENT1.json").read_text(encoding="utf-8")
    )
    usage = json.loads(
        (out_dir / "web-evidence" / "usage" / "EVENT1.json").read_text(encoding="utf-8")
    )

    assert result.exit_code == 0
    assert captured["timeout"] == 900.0
    assert packet["event_ticker"] == "EVENT1"
    assert usage["event_ticker"] == "EVENT1"
    assert usage["response_id"] == "resp_test"
    assert usage["usage"]["total_tokens"] == 150


def test_collect_web_evidence_live_skips_existing_packets(tmp_path: Path, monkeypatch):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    packet_dir = out_dir / "web-evidence" / "packets"
    packet_dir.mkdir(parents=True)
    _write_market_csv(input_csv)
    (packet_dir / "EVENT1.json").write_text(
        json.dumps(
            {
                "event_ticker": "EVENT1",
                "company_name": "John Deere",
                "cutoff_time": "2026-01-01T12:00:00Z",
                "items": [],
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OpenAI should not be called for existing packets")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(prediction_cli, "_fetch_web_evidence_packet", fail_if_called)

    result = CliRunner().invoke(
        app,
        [
            "collect-web-evidence",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--model",
            "gpt-5.5",
            "--max-events",
            "1",
            "--no-dry-run",
            "--skip-existing",
        ],
    )

    assert result.exit_code == 0
    assert "Skipped 1 existing web evidence packets" in result.output


def test_collect_web_evidence_live_accepts_parallel_requests(tmp_path: Path, monkeypatch):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)
    calls: list[str] = []

    def fake_fetch(payload: dict, *, timeout_seconds: float):
        event_ticker = "EVENT1" if '"event_ticker": "EVENT1"' in payload["input"] else "EVENT2"
        calls.append(event_ticker)
        return prediction_cli.WebEvidencePacket.model_validate(
            {
                "event_ticker": event_ticker,
                "company_name": "John Deere",
                "cutoff_time": "2026-01-01T12:00:00Z",
                "items": [],
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(prediction_cli, "_fetch_web_evidence_packet", fake_fetch)

    result = CliRunner().invoke(
        app,
        [
            "collect-web-evidence",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--model",
            "gpt-5.5",
            "--max-events",
            "2",
            "--no-dry-run",
            "--parallel-requests",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert sorted(calls) == ["EVENT1", "EVENT2"]
    assert (out_dir / "web-evidence" / "packets" / "EVENT1.json").exists()
    assert (out_dir / "web-evidence" / "packets" / "EVENT2.json").exists()
    assert "Parallel requests: 2" in result.output


def test_collect_web_evidence_rejects_runs_over_max_paid_calls(
    tmp_path: Path,
    monkeypatch,
):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OpenAI should not be called over the paid-call cap")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(prediction_cli, "_fetch_web_evidence_packet", fail_if_called)

    result = CliRunner().invoke(
        app,
        [
            "collect-web-evidence",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--model",
            "gpt-5.5",
            "--max-events",
            "1",
            "--no-dry-run",
            "--max-paid-calls",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "max paid calls" in str(result.exception)


def test_collect_web_evidence_rejects_runs_over_estimated_cost(
    tmp_path: Path,
    monkeypatch,
):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OpenAI should not be called over the cost cap")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(prediction_cli, "_fetch_web_evidence_packet", fail_if_called)

    result = CliRunner().invoke(
        app,
        [
            "collect-web-evidence",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--model",
            "gpt-5.5",
            "--max-events",
            "2",
            "--no-dry-run",
            "--max-estimated-cost-dollars",
            "1.00",
            "--estimated-cost-per-call-dollars",
            "0.75",
        ],
    )

    assert result.exit_code != 0
    assert "estimated cost" in str(result.exception)


def test_build_prompts_backtest_and_cleanup_commands(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    prompt_result = CliRunner().invoke(
        app,
        ["build-prompts", str(input_csv), "--run-id", "unit", "--out-dir", str(out_dir)],
    )
    backtest_result = CliRunner().invoke(
        app,
        [
            "backtest",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--min-training-events",
            "1",
            "--epochs",
            "5",
        ],
    )
    scratch_file = out_dir / "scratch.tmp"
    scratch_file.write_text("delete me", encoding="utf-8")
    cleanup_result = CliRunner().invoke(app, ["cleanup", str(out_dir)])

    assert prompt_result.exit_code == 0
    assert (out_dir / "prompts" / "EVENT1.txt").exists()
    assert backtest_result.exit_code == 0
    assert (out_dir / "backtest.json").exists()
    assert (out_dir / "trades.csv").exists()
    assert cleanup_result.exit_code == 0
    assert not scratch_file.exists()
    assert (out_dir / "backtest.json").exists()


def test_collect_mixmcp_dry_run_defaults_to_mini_model(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "unit"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "collect-mixmcp",
            str(input_csv),
            "--run-id",
            "unit",
            "--out-dir",
            str(out_dir),
            "--max-events",
            "1",
            "--dry-run",
        ],
    )

    request_payload = json.loads(
        (out_dir / "mixmcp" / "requests" / "EVENT1.json").read_text(encoding="utf-8")
    )

    assert result.exit_code == 0
    assert request_payload["model"] == "gpt-5.4-mini"
    assert request_payload["tools"] == [{"type": "web_search"}]
    assert "market_probability" in request_payload["input"]


def test_audit_web_evidence_cli_writes_reports(tmp_path: Path):
    web_dir = tmp_path / "web-evidence" / "packets"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "audit"
    web_dir.mkdir(parents=True)
    (web_dir / "EVENT1.json").write_text(
        json.dumps(
            {
                "event_ticker": "EVENT1",
                "company_name": "Example Co",
                "cutoff_time": "2026-01-01T12:00:00Z",
                "items": [
                    {
                        "title": "Example earnings call transcript",
                        "url": "https://example.com/transcript",
                        "source": "Example",
                        "published_at": "2026-01-01T10:00:00Z",
                        "snippet": "Transcript text",
                        "target_phrases": ["tariff"],
                        "evidence_strength": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["audit-web-evidence", str(tmp_path / "web-evidence"), "--out-dir", str(out_dir)],
    )

    report = json.loads((out_dir / "web-evidence-audit.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert report["summary"]["issue_count"] == 1
    assert (out_dir / "web-evidence-audit.csv").exists()
    assert (out_dir / "exclusion-manifest.json").exists()


def test_audit_web_evidence_cli_rejects_artifacts_full_output(tmp_path: Path):
    web_dir = tmp_path / "web-evidence"
    web_dir.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "audit-web-evidence",
            str(web_dir),
            "--out-dir",
            str(tmp_path / "artifacts" / "full"),
        ],
    )

    assert result.exit_code != 0
    assert "artifacts/full" in str(result.exception)


def test_sweep_cli_writes_ranked_local_results(tmp_path: Path):
    input_csv = tmp_path / "markets.csv"
    out_dir = tmp_path / "artifacts" / "prediction-engine" / "sweep"
    _write_market_csv(input_csv)

    result = CliRunner().invoke(
        app,
        [
            "sweep",
            str(input_csv),
            "--run-id",
            "unit-sweep",
            "--out-dir",
            str(out_dir),
            "--min-training-events",
            "1",
            "--epochs-grid",
            "1,2",
            "--learning-rate-grid",
            "0.05",
            "--l2-grid",
            "0.001",
            "--residual-clip-grid",
            "1.0",
            "--margin-grid",
            "0,0.05",
            "--trade-side-grid",
            "no_only",
            "--target-side-grid",
            "no",
            "--positive-label-weight-grid",
            "1.0,2.0",
            "--feature-ablation-group-grid",
            "none,semantic",
        ],
    )

    summary = json.loads((out_dir / "sweep-summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    rows = list(csv.DictReader((out_dir / "sweep-results.csv").open(encoding="utf-8")))

    assert len(summary["results"]) == 16
    assert summary["results"][0]["config"]["trade_side"] == "no_only"
    assert {result["config"]["target_side"] for result in summary["results"]} == {"no"}
    assert {row["target_side"] for row in rows} == {"no"}
    assert {row["positive_label_weight"] for row in rows} == {"1.0", "2.0"}
    assert {row["margin"] for row in rows} == {"0.0", "0.05"}
    assert {row["feature_ablation_group"] for row in rows} == {"none", "semantic"}
    assert (out_dir / "sweep-results.csv").exists()
