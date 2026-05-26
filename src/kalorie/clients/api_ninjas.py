from datetime import UTC, date, datetime
from typing import Any

import httpx

from kalorie.domain.models import DocumentChunk, EarningsEvent, SourceDocument
from kalorie.io.documents import chunk_text, content_hash, normalize_text


class VendorError(RuntimeError):
    pass


class VendorAuthError(VendorError):
    pass


class VendorRateLimitError(VendorError):
    pass


class VendorParseError(VendorError):
    pass


class ApiNinjasClient:
    def __init__(
        self,
        api_key: str,
        http_client: httpx.Client,
        base_url: str = "https://api.api-ninjas.com/v1",
    ) -> None:
        self._api_key = api_key
        self._http = http_client
        self._base_url = base_url.rstrip("/")

    def get_earnings_calendar(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[EarningsEvent]:
        response = self._request(
            "GET",
            "/earningscalendar",
            params={
                "ticker": ticker.upper(),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )
        rows = response.json()
        if not isinstance(rows, list):
            raise VendorParseError("earnings calendar response must be a list")
        return [self._parse_event(row) for row in rows]

    def get_transcript(
        self,
        ticker: str,
        fiscal_year: int,
        fiscal_quarter: int,
    ) -> tuple[SourceDocument, list[DocumentChunk]]:
        response = self._request(
            "GET",
            "/earningstranscript",
            params={
                "ticker": ticker.upper(),
                "year": fiscal_year,
                "quarter": fiscal_quarter,
            },
        )
        payload = response.json()
        transcript = normalize_text(str(payload.get("transcript", "")))
        if not transcript:
            raise VendorParseError("transcript response missing transcript text")
        published_at = _parse_datetime(payload.get("date"))
        source_hash = content_hash(transcript)
        source_id = (
            f"{ticker.upper()}-{fiscal_year}-Q{fiscal_quarter}-TRANSCRIPT-{source_hash[:12]}"
        )
        document = SourceDocument(
            source_id=source_id,
            company_symbol=ticker,
            document_type="earnings_call_transcript",
            source_path=f"api-ninjas://earningstranscript/{ticker.upper()}/{fiscal_year}/Q{fiscal_quarter}",
            published_at=published_at,
            content_hash=source_hash,
        )
        chunks = [
            chunk.model_copy(update={"document_id": source_id})
            for chunk in chunk_text(transcript)
        ]
        return document, chunks

    def _request(self, method: str, path: str, params: dict[str, Any]) -> httpx.Response:
        response = self._http.request(
            method,
            f"{self._base_url}{path}",
            params=params,
            headers={"X-Api-Key": self._api_key},
        )
        if response.status_code == 401:
            raise VendorAuthError("API Ninjas authentication failed")
        if response.status_code == 429:
            raise VendorRateLimitError("API Ninjas rate limit exceeded")
        response.raise_for_status()
        return response

    @staticmethod
    def _parse_event(row: dict[str, Any]) -> EarningsEvent:
        event_date_value = row.get("date") or row.get("pricedate")
        if event_date_value is None:
            raise VendorParseError("earnings calendar row missing date")
        return EarningsEvent(
            company_symbol=str(row.get("ticker") or row.get("symbol")),
            fiscal_year=int(row.get("fiscal_year") or row.get("year")),
            fiscal_quarter=int(row.get("fiscal_quarter") or row.get("quarter")),
            event_date=date.fromisoformat(str(event_date_value)[:10]),
        )


def _parse_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
