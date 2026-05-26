from pathlib import Path

from kalorie.workflows.historical_synthetic import build_transcript_inventory
from kalorie.workflows.models import (
    HistoricalSyntheticWorkflowConfig,
    TranscriptInventoryRow,
)


def test_workflow_config_defaults_are_json_serializable():
    config = HistoricalSyntheticWorkflowConfig()

    payload = config.model_dump(mode="json")

    assert payload["sec_request_budget"] == 80
    assert payload["phrase_target_max"] == 12
    assert payload["openai_enabled"] is True
    assert Path(payload["output_root"]).as_posix().endswith("artifacts/model1/workflows")


def test_transcript_inventory_row_normalizes_symbol_and_serializes(tmp_path: Path):
    transcript = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript.write_text("Traffic improved.", encoding="utf-8")

    row = TranscriptInventoryRow(
        company_symbol="wmt",
        company_name="Walmart",
        fiscal_year=2025,
        fiscal_quarter=2,
        transcript_path=transcript,
    )

    assert row.company_symbol == "WMT"
    assert row.model_dump(mode="json")["transcript_path"].endswith("2025_Q2_wmt_processed.txt")


def test_build_transcript_inventory_skips_bad_filenames(tmp_path: Path):
    walmart = tmp_path / "Walmart"
    walmart.mkdir()
    good = walmart / "2025_Q2_wmt_processed.txt"
    good.write_text("Traffic improved.", encoding="utf-8")
    bad = walmart / "notes.txt"
    bad.write_text("ignore me", encoding="utf-8")

    inventory = build_transcript_inventory(tmp_path)

    assert len(inventory.rows) == 1
    assert inventory.rows[0].company_symbol == "WMT"
    assert inventory.rows[0].company_name == "Walmart"
    assert inventory.skipped_count == 1
    assert inventory.skipped_records[0].reason == "unsupported_filename"


def test_build_transcript_inventory_sets_estimated_call_time(tmp_path: Path):
    folder = tmp_path / "Walmart"
    folder.mkdir()
    transcript = folder / "2025_Q2_wmt_processed.txt"
    transcript.write_text("Traffic improved.", encoding="utf-8")

    inventory = build_transcript_inventory(tmp_path)

    assert inventory.rows[0].estimated_call_time is not None
    assert inventory.rows[0].call_time_source == "estimated_fiscal_period_plus_50d"
