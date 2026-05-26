import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pdfplumber

from kalorie.domain.models import DocumentChunk, SourceDocument

SECRET_SUFFIXES = {".env", ".rsa", ".pem", ".key", ".p8", ".p12"}


def _reject_secret_looking_path(path: Path) -> None:
    lowered_name = path.name.lower()
    lowered_suffix = path.suffix.lower()
    if lowered_name == ".env" or lowered_suffix in SECRET_SUFFIXES or "key" in lowered_name:
        raise ValueError(f"Refusing to open secret-looking path: {path.name}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_hash(bytes_or_text: bytes | str) -> str:
    payload = bytes_or_text.encode("utf-8") if isinstance(bytes_or_text, str) else bytes_or_text
    return sha256(payload).hexdigest()


def extract_text_from_pdf(path: Path) -> str:
    _reject_secret_looking_path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    page_texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts.append(page_text)
    return normalize_text("\n\n".join(page_texts))


def _detect_section(chunk_text_value: str) -> str | None:
    for raw_line in chunk_text_value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.isupper() and len(line) <= 80:
            return line
        break
    return None


def chunk_text(text: str, max_chars: int = 1800, overlap_chars: int = 200) -> list[DocumentChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    chunks: list[DocumentChunk] = []
    cursor = 0
    token_cursor = 0
    document_id = content_hash(normalize_text(text))
    while cursor < len(text):
        end = min(cursor + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", cursor, end)
            if boundary > cursor:
                end = boundary
        raw_chunk = text[cursor:end].strip()
        if raw_chunk:
            normalized_chunk = normalize_text(raw_chunk)
            token_count = len(normalized_chunk.split())
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    text=normalized_chunk,
                    section=_detect_section(raw_chunk),
                    token_start=token_cursor,
                    token_end=token_cursor + token_count,
                )
            )
            token_cursor += token_count
        if end == len(text):
            break
        cursor = max(end - overlap_chars, cursor + 1)
    return chunks


def ingest_local_press_release(
    path: Path,
    company_symbol: str,
    fiscal_year: int,
    fiscal_quarter: int,
    published_at: datetime,
) -> tuple[SourceDocument, list[DocumentChunk]]:
    _reject_secret_looking_path(path)
    text = extract_text_from_pdf(path)
    normalized = normalize_text(text)
    source_hash = content_hash(normalized)
    source_id = f"{company_symbol.upper()}-{fiscal_year}-Q{fiscal_quarter}-{source_hash[:12]}"
    document = SourceDocument(
        source_id=source_id,
        company_symbol=company_symbol,
        document_type="earnings_press_release",
        source_path=str(path),
        published_at=published_at,
        content_hash=source_hash,
    )
    chunks = [
        chunk.model_copy(update={"document_id": source_id})
        for chunk in chunk_text(normalized)
    ]
    return document, chunks


def ingest_local_transcript(
    path: Path,
    company_symbol: str,
    fiscal_year: int,
    fiscal_quarter: int,
    published_at: datetime,
) -> tuple[SourceDocument, list[DocumentChunk]]:
    _reject_secret_looking_path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    normalized = normalize_text(text)
    source_hash = content_hash(normalized)
    source_id = (
        f"{company_symbol.upper()}-{fiscal_year}-Q{fiscal_quarter}-"
        f"TRANSCRIPT-{source_hash[:12]}"
    )
    document = SourceDocument(
        source_id=source_id,
        company_symbol=company_symbol,
        document_type="earnings_call_transcript",
        source_path=str(path),
        published_at=published_at,
        content_hash=source_hash,
    )
    chunks = [
        chunk.model_copy(update={"document_id": source_id})
        for chunk in chunk_text(normalized)
    ]
    return document, chunks
