from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from kalorie.io.documents import normalize_text


class FmpApiError(RuntimeError):
    pass


class FmpApiAuthError(FmpApiError):
    pass


class FmpApiRateLimitError(FmpApiError):
    pass


class FmpApiSubscriptionError(FmpApiError):
    pass


class FmpApiParseError(FmpApiError):
    pass


class FmpTranscriptReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    fiscal_year: int
    fiscal_quarter: int = Field(ge=1, le=4)
    published_at: datetime


class FinancialModelingPrepClient:
    def __init__(
        self,
        api_key: str,
        http_client: httpx.Client,
        base_url: str = "https://financialmodelingprep.com",
    ) -> None:
        self._api_key = api_key
        self._http = http_client
        self._base_url = base_url.rstrip("/")

    def get_transcript_dates(self, symbol: str) -> list[FmpTranscriptReference]:
        response = self._request(
            "GET",
            "/stable/earning-call-transcript-dates",
            params={"symbol": symbol.upper()},
        )
        payload = response.json()
        if not isinstance(payload, list):
            raise FmpApiParseError("FMP transcript dates response must be a list")
        rows: list[FmpTranscriptReference] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            parsed = self._parse_transcript_reference(symbol=symbol, row=row)
            if parsed is not None:
                rows.append(parsed)
        return rows

    def get_transcript_text(
        self,
        *,
        symbol: str,
        fiscal_year: int,
        fiscal_quarter: int,
    ) -> tuple[str, datetime]:
        response = self._request(
            "GET",
            "/stable/earning-call-transcript",
            params={
                "symbol": symbol.upper(),
                "year": fiscal_year,
                "quarter": fiscal_quarter,
            },
        )
        payload = response.json()
        row = self._extract_transcript_row(payload)
        transcript = normalize_text(str(row.get("content") or row.get("transcript") or ""))
        if not transcript:
            raise FmpApiParseError("FMP transcript payload missing transcript content")
        published_at = _parse_datetime(row.get("date") or row.get("publishedDate"))
        return transcript, published_at

    def _request(self, method: str, path: str, params: dict[str, Any]) -> httpx.Response:
        response = self._http.request(
            method,
            f"{self._base_url}{path}",
            params={**params, "apikey": self._api_key},
        )
        if response.status_code == 401:
            raise FmpApiAuthError("Financial Modeling Prep authentication failed")
        if response.status_code == 402:
            raise FmpApiSubscriptionError(
                "Financial Modeling Prep transcript endpoint is restricted for this subscription tier"
            )
        if response.status_code == 429:
            raise FmpApiRateLimitError("Financial Modeling Prep rate limit exceeded")
        response.raise_for_status()
        return response

    def _parse_transcript_reference(
        self,
        *,
        symbol: str,
        row: dict[str, Any],
    ) -> FmpTranscriptReference | None:
        year_value = row.get("year") or row.get("fiscalYear")
        quarter_value = row.get("quarter") or row.get("fiscalQuarter")
        if year_value is None or quarter_value is None:
            return None
        try:
            fiscal_year = int(str(year_value))
            fiscal_quarter = int(str(quarter_value))
        except ValueError:
            return None
        date_value = row.get("date") or row.get("publishedDate")
        return FmpTranscriptReference(
            symbol=symbol.upper(),
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            published_at=_parse_datetime(date_value),
        )

    @staticmethod
    def _extract_transcript_row(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    return row
        raise FmpApiParseError("FMP transcript response missing transcript row")


def _parse_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
