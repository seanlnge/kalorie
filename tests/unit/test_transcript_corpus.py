from pathlib import Path

from kalorie.io.transcript_corpus import scan_transcript_corpus


def test_scan_transcript_corpus_parses_processed_transcript_paths(tmp_path: Path):
    folder = tmp_path / "Walmart"
    folder.mkdir()
    transcript = folder / "2025_Q2_wmt_processed.txt"
    transcript.write_text("Traffic and automation were discussed.", encoding="utf-8")

    records = scan_transcript_corpus(tmp_path)

    assert len(records) == 1
    assert records[0].company_name == "Walmart"
    assert records[0].company_symbol == "WMT"
    assert records[0].fiscal_year == 2025
    assert records[0].fiscal_quarter == 2
    assert records[0].path == transcript
