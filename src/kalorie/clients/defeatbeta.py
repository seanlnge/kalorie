from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DEFEATBETA_STOCK_NEWS_URL = (
    "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/"
    "resolve/main/data/stock_news.parquet"
)


class DefeatBetaApiError(RuntimeError):
    pass


class DefeatBetaParseError(DefeatBetaApiError):
    pass


class DefeatBetaArticle(BaseModel):
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


class DefeatBetaNewsClient:
    def __init__(
        self,
        *,
        dataset_url: str | None = None,
    ) -> None:
        self._dataset_url = dataset_url

    def search_stock_news(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
        max_rows: int = 200,
    ) -> list[DefeatBetaArticle]:
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        ticker = symbol.upper()
        dataset_url = self._resolve_dataset_url()
        filters: list[tuple[str, str, str]] = [
            ("related_symbols", "==", ticker),
            ("report_date", ">=", start_date),
            ("report_date", "<=", end_date),
        ]
        try:
            frame = pd.read_parquet(
                dataset_url,
                columns=["uuid", "related_symbols", "title", "publisher", "report_date", "type", "link", "news"],
                filters=filters,
            )
        except ImportError as exc:
            raise DefeatBetaApiError(
                "DefeatBeta parquet access requires pyarrow (install pyarrow>=24)"
            ) from exc
        except Exception as exc:
            raise DefeatBetaApiError(f"DefeatBeta parquet query failed: {exc}") from exc

        rows = frame.to_dict(orient="records")
        articles: list[DefeatBetaArticle] = []
        for row in sorted(rows, key=lambda item: str(item.get("report_date") or "")):
            if not isinstance(row, dict):
                continue
            try:
                articles.append(self._parse_article(row))
            except DefeatBetaParseError:
                continue
        return articles[:max_rows]

    def _resolve_dataset_url(self) -> str:
        if self._dataset_url:
            return self._dataset_url
        try:
            # Prefer library-resolved path when available; fallback handles import side effects.
            from defeatbeta_api.client.hugging_face_client import HuggingFaceClient
            from defeatbeta_api.utils.const import stock_news

            return HuggingFaceClient().get_url_path(stock_news)
        except Exception:
            return DEFAULT_DEFEATBETA_STOCK_NEWS_URL

    @staticmethod
    def _parse_article(row: dict[str, Any]) -> DefeatBetaArticle:
        article_id = str(row.get("uuid") or row.get("id") or "").strip()
        title = str(row.get("title") or "").strip()
        link = str(row.get("link") or row.get("url") or "").strip()
        if not article_id or not title or not link:
            raise DefeatBetaParseError("DefeatBeta article missing uuid/title/link")
        published_at = _parse_datetime(
            row.get("report_date") or row.get("publishedDate"),
            field_name="report_date",
        )
        paragraphs = row.get("paragraphs")
        if paragraphs is None:
            paragraphs = row.get("news")
        content = None
        if hasattr(paragraphs, "tolist"):
            paragraphs = paragraphs.tolist()
        if isinstance(paragraphs, list):
            text_parts = []
            for item in paragraphs:
                if not isinstance(item, dict):
                    continue
                paragraph_text = str(item.get("paragraph") or "").strip()
                if paragraph_text:
                    text_parts.append(paragraph_text)
            if text_parts:
                content = "\n\n".join(text_parts)
        return DefeatBetaArticle(
            article_id=article_id,
            title=title,
            link=link,
            source_name=_optional_str(row.get("publisher")),
            description=_optional_str(row.get("summary")),
            content=content,
            datatype=_optional_str(row.get("type")),
            published_at=published_at,
            tickers=_normalize_tickers(row.get("related_symbols")),
            tags=[],
        )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_tickers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().upper() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text.upper()]


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    if value is None:
        raise DefeatBetaParseError(f"DefeatBeta article missing {field_name}")
    text = str(value).strip().replace("Z", "+00:00")
    # API docs show plain date values for report_date.
    if len(text) == 10 and text.count("-") == 2:
        parsed = datetime.fromisoformat(f"{text}T00:00:00+00:00")
    else:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise DefeatBetaParseError(f"Invalid DefeatBeta {field_name}: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
