from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field


class TiingoApiError(RuntimeError):
    pass


class TiingoAuthError(TiingoApiError):
    pass


class TiingoRateLimitError(TiingoApiError):
    pass


class TiingoParseError(TiingoApiError):
    pass


class TiingoArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_id: str
    title: str
    link: str
    source_name: str | None = None
    source_priority: int | None = None
    description: str | None = None
    content: str | None = None
    datatype: str | None = None
    published_at: datetime
    tickers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    crawled_at: datetime | None = None


class TiingoNewsClient:
    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.Client,
        base_url: str = "https://api.tiingo.com",
    ) -> None:
        self._api_key = api_key
        self._http = http_client
        self._base_url = base_url.rstrip("/")

    def search_news(
        self,
        *,
        ticker: str,
        start_date: str,
        end_date: str,
        limit: int = 200,
    ) -> list[TiingoArticle]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        params: dict[str, str | int] = {
            "tickers": ticker.upper(),
            "startDate": start_date,
            "endDate": end_date,
            "limit": limit,
        }
        response = self._http.get(
            f"{self._base_url}/tiingo/news",
            params=params,
            headers={"Authorization": f"Token {self._api_key}"},
        )
        if response.status_code in {401, 403}:
            raise TiingoAuthError("Tiingo authentication failed")
        if response.status_code == 429:
            raise TiingoRateLimitError("Tiingo rate limit exceeded")
        if response.status_code >= 400:
            raise TiingoApiError(f"Tiingo request failed ({response.status_code})")
        payload = response.json()
        if not isinstance(payload, list):
            raise TiingoParseError("Tiingo news response must be a list")
        rows: list[TiingoArticle] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                rows.append(self._parse_article(item))
            except TiingoParseError:
                continue
        return rows[:limit]

    @staticmethod
    def _parse_article(item: dict[str, Any]) -> TiingoArticle:
        article_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()
        link = str(item.get("url") or "").strip()
        if not article_id or not title or not link:
            raise TiingoParseError("Tiingo article missing id, title, or url")
        source = item.get("source")
        return TiingoArticle(
            article_id=article_id,
            title=title,
            link=link,
            source_name=str(source).strip() if source else None,
            description=_optional_str(item.get("description")),
            datatype=str(item.get("type")).strip().lower() if item.get("type") else "tiingo",
            published_at=_parse_datetime(item.get("publishedDate"), field_name="publishedDate"),
            crawled_at=_parse_datetime(item.get("crawlDate"), field_name="crawlDate", required=False),
            tickers=_optional_str_list(item.get("tickers")),
            tags=_optional_str_list(item.get("tags")),
        )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().upper() for item in value if str(item).strip()]


def _parse_datetime(value: Any, *, field_name: str, required: bool = True) -> datetime | None:
    if value is None:
        if required:
            raise TiingoParseError(f"Tiingo article missing {field_name}")
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TiingoParseError(f"Invalid Tiingo {field_name}: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
