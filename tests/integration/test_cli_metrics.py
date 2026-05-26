import json
from pathlib import Path

from typer.testing import CliRunner

from kalorie.app.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_evaluate_run_writes_brier_for_transcript_smoke_run(tmp_path):
    run_dir = tmp_path / "cava-q1-2026-transcript"
    metrics_path = tmp_path / "metrics.json"
    runner = CliRunner()

    run_result = runner.invoke(
        app,
        [
            "run-local-transcript",
            "--transcript",
            str(PROJECT_ROOT / "data" / "raw" / "cava-q1-2026-transcript.txt"),
            "--market-title",
            "Will CAVA mention traffic during earnings?",
            "--yes-bid",
            "0.38",
            "--yes-ask",
            "0.45",
            "--out",
            str(run_dir),
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    metrics_result = runner.invoke(
        app,
        ["evaluate-run", "--run", str(run_dir), "--out", str(metrics_path)],
    )

    assert metrics_result.exit_code == 0, metrics_result.output
    assert "Brier score:" in metrics_result.output
    assert "MSE:" not in metrics_result.output

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["evaluation_kind"] == "smoke"
    assert metrics["sample_count"] == 6
    assert "expected_calibration_error" in metrics
    assert "mean_squared_error" not in metrics
    assert not metrics["trained_model"]
