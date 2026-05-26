from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict


class NewsDataApiError(RuntimeError):
    pass


class NewsDataAuthError(NewsDataApiError):
    pass


class NewsDataSubscriptionError(NewsDataApiError):
    pass


class NewsDataRateLimitError(NewsDataApiError):
    pass


class NewsDataParseError(NewsDataApiError):
    pass


class NewsDataArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_id: str
    title: str
    link: str | None = None
    source_name: str | None = None
    source_priority: int | None = None
    description: str | None = None
    content: str | None = None
    datatype: str | None = None
    published_at: datetime


class NewsDataClient:
    def __init__(
        self,
        api_key: str,
        http_client: httpx.Client,
        base_url: str = "https://newsdata.io/api/1",
    ) -> None:
        self._api_key = api_key
        self._http = http_client
        self._base_url = base_url.rstrip("/")

    def search_archive(
        self,
        *,
        query: str,
        from_date: str,
        to_date: str,
        language: str = "en",
        size: int = 50,
        max_articles: int = 100,
    ) -> list[NewsDataArticle]:
        if max_articles < 1:
            raise ValueError("max_articles must be at least 1")
        if size < 1:
            raise ValueError("size must be at least 1")
        articles: list[NewsDataArticle] = []
        next_page: str | None = None

        while len(articles) < max_articles:
            params: dict[str, str | int] = {
                "apikey": self._api_key,
                "q": query,
                "from_date": from_date,
                "to_date": to_date,
                "language": language,
                "size": size,
            }
            if next_page:
                params["page"] = next_page
            payload = self._request_json("/archive", params)
            next_page = _extend_articles_from_payload(
                payload=payload,
                articles=articles,
                max_articles=max_articles,
                parse_article=self._parse_article,
                parse_error_cls=NewsDataParseError,
            )
            if not next_page:
                break
        return articles[:max_articles]

    def search_latest(
        self,
        *,
        query: str,
        language: str = "en",
        size: int = 50,
        max_articles: int = 100,
    ) -> list[NewsDataArticle]:
        if max_articles < 1:
            raise ValueError("max_articles must be at least 1")
        if size < 1:
            raise ValueError("size must be at least 1")
        articles: list[NewsDataArticle] = []
        next_page: str | None = None
        while len(articles) < max_articles:
            params: dict[str, str | int] = {
                "apikey": self._api_key,
                "q": query,
                "language": language,
                "size": size,
            }
            if next_page:
                params["page"] = next_page
            payload = self._request_json("/latest", params)
            next_page = _extend_articles_from_payload(
                payload=payload,
                articles=articles,
                max_articles=max_articles,
                parse_article=self._parse_article,
                parse_error_cls=NewsDataParseError,
            )
            if not next_page:
                break
        return articles[:max_articles]

    def _request_json(self, path: str, params: dict[str, str | int]) -> dict[str, Any]:
        response = self._http.get(f"{self._base_url}{path}", params=params)
        if response.status_code == 401:
            raise NewsDataAuthError("NewsData authentication failed")
        if response.status_code == 403:
            payload = _safe_json(response)
            results = payload.get("results", {}) if isinstance(payload, dict) else {}
            code = str(results.get("code", "")).lower()
            message = str(results.get("message", "")).lower()
            if code == "accessdenied" or "upgrade your plan" in message:
                raise NewsDataSubscriptionError("NewsData endpoint is restricted for this plan")
            raise NewsDataAuthError("NewsData authentication failed")
        if response.status_code == 429:
            raise NewsDataRateLimitError("NewsData rate limit exceeded")
        if response.status_code >= 400:
            payload = _safe_json(response)
            results = payload.get("results", {}) if isinstance(payload, dict) else {}
            message = results.get("message") if isinstance(results, dict) else None
            if not message:
                message = f"NewsData request failed ({response.status_code})"
            raise NewsDataApiError(str(message))
        payload = response.json()
        if payload.get("status") == "error":
            message = payload.get("results", {}).get("message") or "NewsData request failed"
            raise NewsDataApiError(str(message))
        return payload

    @staticmethod
    def _parse_article(row: dict[str, Any]) -> NewsDataArticle:
        title = str(row.get("title") or "").strip()
        article_id = str(row.get("article_id") or "").strip()
        if not title or not article_id:
            raise NewsDataParseError("NewsData article missing title or article_id")
        return NewsDataArticle(
            article_id=article_id,
            title=title,
            link=_optional_str(row.get("link")),
            source_name=_optional_str(row.get("source_name")),
            source_priority=_optional_int(row.get("source_priority")),
            description=_optional_str(row.get("description")),
            content=_optional_str(row.get("content")),
            datatype=_optional_str(row.get("datatype")),
            published_at=_parse_datetime(row.get("pubDate")),
        )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise NewsDataParseError(f"Invalid NewsData pubDate: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _extend_articles_from_payload(
    *,
    payload: dict[str, Any],
    articles: list[NewsDataArticle],
    max_articles: int,
    parse_article,
    parse_error_cls: type[Exception],
) -> str | None:
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise NewsDataParseError("NewsData response results must be a list")
    if not rows:
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            article = parse_article(row)
        except parse_error_cls:
            continue
        articles.append(article)
        if len(articles) >= max_articles:
            break
    next_page = payload.get("nextPage")
    if isinstance(next_page, str) and next_page:
        return next_page
    return None
