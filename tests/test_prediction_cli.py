import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from kalorie2 import prediction_cli
from kalorie2.prediction_cli import app


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
            "0",
            "--trade-side-grid",
            "no_only",
        ],
    )

    summary = json.loads((out_dir / "sweep-summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert len(summary["results"]) == 2
    assert summary["results"][0]["config"]["trade_side"] == "no_only"
    assert (out_dir / "sweep-results.csv").exists()
