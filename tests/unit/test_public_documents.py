from datetime import UTC, datetime
from pathlib import Path

import httpx

from kalorie.io.public_documents import collect_public_document, normalize_html_text


def test_normalize_html_text_extracts_visible_text():
    html = """
    <html><head><title>CAVA Q1</title><script>ignore()</script></head>
    <body><h1>CAVA Group Reports Results</h1><p>Traffic improved.</p></body></html>
    """

    assert normalize_html_text(html) == "CAVA Group Reports Results Traffic improved."


def test_collect_public_document_writes_raw_artifact_and_manifest(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><h1>CAVA Group Reports Results</h1></body></html>",
        )

    manifest = collect_public_document(
        url="https://investor.cava.com/q1-2026-results",
        company_symbol="CAVA",
        fiscal_year=2026,
        fiscal_quarter=1,
        source_type="press_release",
        published_at=datetime(2026, 5, 19, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 20, tzinfo=UTC),
        raw_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert manifest.company_symbol == "CAVA"
    assert manifest.source_url == "https://investor.cava.com/q1-2026-results"
    assert manifest.source_type == "press_release"
    assert manifest.extraction_method == "html_text"
    assert manifest.content_hash
    assert Path(manifest.raw_path).exists()
    assert "CAVA Group Reports Results" in Path(manifest.raw_path).read_text(encoding="utf-8")
    assert manifest.raw_path == manifest.extracted_text_path
    assert manifest.raw_original_path is not None
    assert manifest.raw_original_content_hash is not None
    original_path = Path(manifest.raw_original_path)
    assert original_path.exists()
    assert original_path.suffix == ".htm"
    assert "<h1>CAVA Group Reports Results</h1>" in original_path.read_text(
        encoding="utf-8"
    )
    assert Path(manifest.extracted_text_path).suffix == ".txt"


def test_collect_public_document_preserves_sec_ix_original_suffix(tmp_path: Path):
    seen_url = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = str(request.url)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><h1>SEC Exhibit</h1></body></html>",
        )

    manifest = collect_public_document(
        url="https://www.sec.gov/ix?doc=/Archives/edgar/data/1/example-ex991.htm",
        company_symbol="TEST",
        fiscal_year=2026,
        fiscal_quarter=1,
        source_type="sec_ex_99_1_supplemental",
        published_at=datetime(2026, 5, 19, tzinfo=UTC),
        fetched_at=datetime(2026, 5, 20, tzinfo=UTC),
        raw_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert seen_url == "https://www.sec.gov/Archives/edgar/data/1/example-ex991.htm"
    assert manifest.source_url == "https://www.sec.gov/Archives/edgar/data/1/example-ex991.htm"
    assert manifest.raw_original_path is not None
    assert Path(manifest.raw_original_path).suffix == ".htm"
    assert Path(manifest.extracted_text_path).suffix == ".txt"
