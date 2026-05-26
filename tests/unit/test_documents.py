from datetime import UTC, datetime
from pathlib import Path

import pytest

from kalorie.io.documents import (
    chunk_text,
    content_hash,
    extract_text_from_pdf,
    ingest_local_press_release,
    normalize_text,
)


def test_normalize_text_collapses_repeated_whitespace():
    assert normalize_text("CAVA   reports\n\nfirst\tquarter") == "CAVA reports first quarter"


def test_content_hash_is_stable_for_text_and_bytes():
    assert content_hash("same restaurant sales") == content_hash(b"same restaurant sales")
    assert content_hash("same restaurant sales") != content_hash("traffic")


def test_chunk_text_preserves_order_and_detects_section_labels():
    text = "HIGHLIGHTS\n" + "A" * 80 + "\n\nOUTLOOK\n" + "B" * 80

    chunks = chunk_text(text, max_chars=100, overlap_chars=10)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].section == "HIGHLIGHTS"
    assert chunks[-1].text
    assert chunks[-1].token_end >= chunks[-1].token_start


@pytest.mark.parametrize(
    "name",
    [".env", "kalshi.rsa", "private.pem", "private.key", "contains-key-material.txt"],
)
def test_document_extraction_rejects_secret_looking_paths(tmp_path: Path, name: str):
    secret_like_path = tmp_path / name
    secret_like_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="secret-looking"):
        extract_text_from_pdf(secret_like_path)


def test_ingest_local_press_release_uses_normalized_document_models(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "release.pdf"
    pdf_path.write_bytes(b"%PDF-placeholder")
    monkeypatch.setattr(
        "kalorie.io.documents.extract_text_from_pdf",
        lambda path: "HEADLINE\ntraffic grew",
    )

    document, chunks = ingest_local_press_release(
        pdf_path,
        company_symbol="cava",
        fiscal_year=2026,
        fiscal_quarter=1,
        published_at=datetime(2026, 5, 19, tzinfo=UTC),
    )

    assert document.company_symbol == "CAVA"
    assert document.document_type == "earnings_press_release"
    assert document.source_path == str(pdf_path)
    assert document.content_hash
    assert chunks[0].document_id == document.source_id
