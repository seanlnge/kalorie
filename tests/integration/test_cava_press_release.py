from datetime import UTC, datetime

from tests.conftest import PROJECT_ROOT

from kalorie.io.documents import extract_text_from_pdf, ingest_local_press_release


def test_cava_press_release_pdf_ingests_into_document_and_chunks():
    pdf_path = PROJECT_ROOT / "Earnings-Release-2026-Q1.pdf"

    text = extract_text_from_pdf(pdf_path)
    document, chunks = ingest_local_press_release(
        pdf_path,
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        published_at=datetime(2026, 5, 19, tzinfo=UTC),
    )

    assert document.company_symbol == "CAVA"
    assert document.document_type == "earnings_press_release"
    assert "CAVA GROUP REPORTS FIRST QUARTER 2026 RESULTS" in text
    assert "same restaurant sales" in text.lower()
    assert chunks
    assert all(chunk.text for chunk in chunks)
    assert document.content_hash
