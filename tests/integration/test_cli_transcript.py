import json
from pathlib import Path

from typer.testing import CliRunner

from kalorie.app.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_local_transcript_writes_expected_artifacts(tmp_path):
    out_dir = tmp_path / "cava-q1-2026-transcript"
    runner = CliRunner()

    result = runner.invoke(
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

    document = json.loads((out_dir / "document.json").read_text(encoding="utf-8"))
    labels = json.loads((out_dir / "labels.json").read_text(encoding="utf-8"))
    prediction = json.loads((out_dir / "prediction.json").read_text(encoding="utf-8"))
    comparison = json.loads((out_dir / "paper_comparison.json").read_text(encoding="utf-8"))

    labels_by_phrase = {label["target_phrase"]: label for label in labels}
    assert document["document_type"] == "earnings_call_transcript"
    assert document["company_symbol"] == "CAVA"
    assert labels_by_phrase["traffic"]["exact_mentioned"]
    assert labels_by_phrase["same restaurant sales"]["exact_mentioned"]
    assert not labels_by_phrase["digital revenue"]["exact_mentioned"]
    assert labels_by_phrase["geopolitical uncertainty"]["exact_mentioned"]
    assert labels_by_phrase["value proposition"]["exact_mentioned"]
    assert labels_by_phrase["margin"]["exact_mentioned"]
    assert prediction["target_phrase"] == "traffic"
    assert prediction["probability"] > 0.5
    assert comparison["side"] in {"yes", "no", "skip"}
