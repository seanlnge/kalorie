import json

from tests.conftest import PROJECT_ROOT
from typer.testing import CliRunner

from kalorie.app.cli import app


def test_run_local_cava_writes_expected_artifacts(tmp_path):
    out_dir = tmp_path / "cava-q1-2026"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-local-cava",
            "--pdf",
            str(PROJECT_ROOT / "Earnings-Release-2026-Q1.pdf"),
            "--market-title",
            "Will CAVA mention traffic during earnings?",
            "--yes-bid",
            "0.38",
            "--yes-ask",
            "0.45",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    expected_files = {
        "document.json",
        "chunks.jsonl",
        "labels.json",
        "features.json",
        "prediction.json",
        "paper_comparison.json",
    }
    assert expected_files == {path.name for path in out_dir.iterdir()}

    prediction = json.loads((out_dir / "prediction.json").read_text(encoding="utf-8"))
    comparison = json.loads((out_dir / "paper_comparison.json").read_text(encoding="utf-8"))
    labels = json.loads((out_dir / "labels.json").read_text(encoding="utf-8"))

    assert prediction["target_phrase"] == "traffic"
    assert 0 <= prediction["probability"] <= 1
    assert comparison["side"] in {"yes", "no", "skip"}
    assert any(label["target_phrase"] == "traffic" and label["exact_mentioned"] for label in labels)
