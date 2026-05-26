import json
from pathlib import Path

from kalorie.workflows.models import TranscriptInventoryRow
from kalorie.workflows.phrase_generation import (
    build_phrase_catalog,
    generate_validated_phrase_entries,
    is_simple_kalshi_phrase,
    parse_openai_phrase_response,
)


def _row(tmp_path: Path) -> TranscriptInventoryRow:
    transcript = tmp_path / "2025_Q2_wmt_processed.txt"
    transcript.write_text("", encoding="utf-8")
    return TranscriptInventoryRow(
        company_symbol="WMT",
        company_name="Walmart",
        fiscal_year=2025,
        fiscal_quarter=2,
        transcript_path=transcript,
    )


def test_simple_kalshi_phrase_filter_accepts_market_like_terms():
    assert is_simple_kalshi_phrase("OpenAI") is True
    assert is_simple_kalshi_phrase("brick and mortar") is True
    assert is_simple_kalshi_phrase("omnichannel") is True
    assert is_simple_kalshi_phrase("a very long conceptual business transformation phrase") is False
    assert is_simple_kalshi_phrase("the") is False


def test_parse_openai_phrase_response_requires_strict_lists():
    payload = parse_openai_phrase_response(
        json.dumps(
            {
                "present_phrases": ["OpenAI", "omnichannel"],
                "absent_phrases": ["robotaxi"],
            }
        )
    )

    assert payload.present_phrases == ["OpenAI", "omnichannel"]
    assert payload.absent_phrases == ["robotaxi"]


def test_generate_validated_phrase_entries_rejects_invalid_and_conflicting_openai_phrases(
    tmp_path: Path,
):
    transcript_text = (
        "Walmart discussed OpenAI, omnichannel investments, automotive products, "
        "and brick and mortar stores."
    )

    entries = generate_validated_phrase_entries(
        _row(tmp_path),
        transcript_text=transcript_text,
        openai_response=json.dumps(
            {
                "present_phrases": [
                    "OpenAI",
                    "brick and mortar",
                    "a very long conceptual business transformation phrase",
                    "robotaxi",
                ],
                "absent_phrases": ["robotaxi", "omnichannel", "the"],
            }
        ),
        max_per_label=12,
    )

    by_label = {
        label: {entry.phrase for entry in entries if entry.label == label}
        for label in ["present", "absent"]
    }
    assert {"OpenAI", "brick and mortar"}.issubset(by_label["present"])
    assert "robotaxi" in by_label["absent"]
    assert "omnichannel" not in by_label["absent"]
    assert "a very long conceptual business transformation phrase" not in by_label["present"]


def test_build_phrase_catalog_checkpoints_each_completed_transcript(tmp_path: Path):
    row = _row(tmp_path)
    row.transcript_path.write_text("OpenAI and omnichannel were discussed.", encoding="utf-8")
    checkpoints = []

    catalog = build_phrase_catalog(
        rows=[row],
        openai_response_provider=lambda row, text: (
            '{"present_phrases":["OpenAI"],"absent_phrases":["robotaxi"]}'
        ),
        checkpoint_writer=lambda catalog: checkpoints.append(len(catalog.entries)),
        max_workers=2,
    )

    assert {entry.phrase for entry in catalog.entries} >= {"OpenAI", "robotaxi"}
    assert checkpoints == [len(catalog.entries)]


def test_build_phrase_catalog_uses_absent_fallback_when_openai_fails(tmp_path: Path):
    row = _row(tmp_path)
    row.transcript_path.write_text("Omnichannel traffic improved.", encoding="utf-8")

    catalog = build_phrase_catalog(
        rows=[row],
        openai_response_provider=lambda row, text: (_ for _ in ()).throw(
            RuntimeError("rate limited")
        ),
    )

    assert any(entry.label == "present" for entry in catalog.entries)
    assert any(entry.label == "absent" for entry in catalog.entries)
    assert catalog.skipped_records[0].reason == "openai_phrase_generation_failed"
