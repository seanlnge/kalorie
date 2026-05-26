import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx
from pydantic import AnyHttpUrl, Field, field_validator

from kalorie.domain.models import KalorieModel
from kalorie.io.documents import content_hash


class PublicDocumentManifest(KalorieModel):
    source_url: str
    company_symbol: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    source_type: str
    published_at: datetime
    fetched_at: datetime
    raw_path: str
    raw_original_path: str | None = None
    raw_original_content_hash: str | None = None
    extracted_text_path: str | None = None
    content_hash: str
    extraction_method: str

    @field_validator("company_symbol")
    @classmethod
    def normalize_company_symbol(cls, value: str) -> str:
        return value.upper()


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"head", "script", "style", "noscript", "title"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"head", "script", "style", "noscript", "title"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def normalize_html_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def collect_public_document(
    *,
    url: str,
    company_symbol: str,
    fiscal_year: int,
    fiscal_quarter: int,
    source_type: str,
    published_at: datetime,
    fetched_at: datetime,
    raw_dir: Path,
    http_client: httpx.Client,
) -> PublicDocumentManifest:
    fetch_url = _normalize_sec_ix_url(url)
    AnyHttpUrl(fetch_url)
    response = http_client.get(fetch_url)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    raw_dir.mkdir(parents=True, exist_ok=True)
    original_payload = response.content
    original_digest = content_hash(original_payload)
    original_suffix = _source_suffix(fetch_url, content_type)
    original_filename = (
        f"{company_symbol.upper()}-{fiscal_year}-Q{fiscal_quarter}-"
        f"{original_digest[:12]}-original{original_suffix}"
    )
    original_path = (
        raw_dir
        / original_filename
    )
    original_path.write_bytes(original_payload)

    if "html" in content_type:
        text = normalize_html_text(response.text)
        extraction_method = "html_text"
        extracted_digest = content_hash(text)
        extracted_filename = (
            f"{company_symbol.upper()}-{fiscal_year}-Q{fiscal_quarter}-"
            f"{extracted_digest[:12]}.txt"
        )
        raw_path = (
            raw_dir
            / extracted_filename
        )
        raw_path.write_text(text, encoding="utf-8")
        raw_payload = text
        extracted_text_path = raw_path
    else:
        extraction_method = "raw_bytes"
        raw_payload = response.content
        raw_path = original_path
        extracted_text_path = None

    digest = content_hash(raw_payload)
    return PublicDocumentManifest(
        source_url=fetch_url,
        company_symbol=company_symbol,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        source_type=source_type,
        published_at=published_at,
        fetched_at=fetched_at,
        raw_path=str(raw_path),
        raw_original_path=str(original_path),
        raw_original_content_hash=original_digest,
        extracted_text_path=str(extracted_text_path) if extracted_text_path else None,
        content_hash=digest,
        extraction_method=extraction_method,
    )


def _source_suffix(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if not suffix and parsed.query:
        doc_values = parse_qs(parsed.query).get("doc", [])
        if doc_values:
            suffix = Path(doc_values[0]).suffix
    if suffix:
        return suffix
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".htm"
    if "text" in content_type:
        return ".txt"
    return ".bin"


def _normalize_sec_ix_url(url: str) -> str:
    parsed = urlparse(url)
    doc_values = parse_qs(parsed.query).get("doc", [])
    if parsed.netloc.lower().endswith("sec.gov") and parsed.path == "/ix" and doc_values:
        return urlunparse((parsed.scheme, parsed.netloc, doc_values[0], "", "", ""))
    return url
