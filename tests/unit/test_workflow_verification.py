import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kalorie.workflows.verification import verify_event_pack_artifacts


def test_verify_event_pack_artifacts_flags_post_cutoff_snapshots(tmp_path: Path):
    event_dir = tmp_path / "KXEARNINGSMENTIONWMT-26AUG"
    event_dir.mkdir()
    cutoff = datetime(2026, 8, 21, 15, 50, tzinfo=UTC)
    (event_dir / "contracts.json").write_text("[]", encoding="utf-8")
    (event_dir / "snapshots.json").write_text(
        json.dumps(
            [
                {
                    "market_id": "bad",
                    "snapshot_target_time": cutoff.isoformat(),
                    "candle_end_ts": int((cutoff + timedelta(minutes=1)).timestamp()),
                }
            ]
        ),
        encoding="utf-8",
    )

    report = verify_event_pack_artifacts(tmp_path)

    assert report.ok is False
    assert any(error.startswith("post_cutoff_snapshot") for error in report.errors)


def test_verify_event_pack_artifacts_requires_transcript_evidence_and_snapshot_parity(
    tmp_path: Path,
):
    event_dir = tmp_path / "KXEARNINGSMENTIONWMT-26AUG"
    event_dir.mkdir()
    (event_dir / "contracts.json").write_text(json.dumps([{"market_id": "one"}]), encoding="utf-8")
    (event_dir / "snapshots.json").write_text("[]", encoding="utf-8")
    (event_dir / "evidence-manifests.json").write_text("[]", encoding="utf-8")

    report = verify_event_pack_artifacts(tmp_path)

    assert report.ok is False
    assert "missing_transcript:KXEARNINGSMENTIONWMT-26AUG" in report.errors
    assert "missing_evidence:KXEARNINGSMENTIONWMT-26AUG" in report.errors
    assert "snapshot_contract_count_mismatch:KXEARNINGSMENTIONWMT-26AUG" in report.errors
