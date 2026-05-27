import re
import time
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx

from kalorie2.models import (
    CollectionResult,
    HistoricalMentionMarketRow,
    MarketCategory,
    PrecloseSnapshot,
    SkippedMarket,
)

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
EARNINGS_MENTION_PREFIX = "KXEARNINGSMENTION"

_POLITICS_TERMS = {
    "biden",
    "campaign",
    "congress",
    "debate",
    "democrat",
    "election",
    "federal reserve",
    "harris",
    "house",
    "mayor",
    "minister",
    "parliament",
    "politic",
    "president",
    "presidential",
    "prime minister",
    "republican",
    "senate",
    "senator",
    "speech",
    "state of the union",
    "trump",
    "vance",
    "vp",
    "white house",
}
_SPORTS_TERMS = {
    "athlete",
    "baseball",
    "basketball",
    "coach",
    "college football",
    "fifa",
    "game",
    "golf",
    "league",
    "match",
    "mlb",
    "nba",
    "nfl",
    "nhl",
    "olympic",
    "player",
    "soccer",
    "sports",
    "super bowl",
    "team",
    "tennis",
    "tournament",
    "ufc",
    "world cup",
}
_MENTION_TEXT_PATTERNS = (
    re.compile(r"\bwhat will .+ say during\b", re.IGNORECASE),
    re.compile(r"\bwhat will .+ say on\b", re.IGNORECASE),
    re.compile(r"\bwill .+ mention\b", re.IGNORECASE),
    re.compile(r"\bif .+ is said by\b", re.IGNORECASE | re.DOTALL),
)


class KalshiClientError(RuntimeError):
    pass


class KalshiMentionClient:
    def __init__(
        self,
        *,
        http_client: httpx.Client,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 3,
        retry_sleep: Any = time.sleep,
        fallback_to_live_candlesticks: bool = True,
    ) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._search_series_url = f"{self._base_url.split('/trade-api/', 1)[0]}/v1/search/series"
        self._max_retries = max_retries
        self._retry_sleep = retry_sleep
        self._fallback_to_live_candlesticks = fallback_to_live_candlesticks

    def iter_mention_series(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        query: str | None = None,
        page_size: int = 100,
        max_pages: int | None = None,
    ) -> Iterator[dict]:
        cursor: str | None = None
        pages_seen = 0
        while True:
            if max_pages is not None and pages_seen >= max_pages:
                return
            params: dict[str, str | int] = {
                "page_size": page_size,
                "hydrate": "milestones,structured_targets",
                "with_milestones": "true",
            }
            if category:
                params["category"] = category
            if query:
                params["query"] = query
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            response = self._get_url(self._search_series_url, params=params)
            payload = response.json()
            current_page = payload.get("current_page", [])
            if not isinstance(current_page, list):
                raise KalshiClientError("Kalshi search series response missing current_page list")
            yield from (row for row in current_page if isinstance(row, dict))
            pages_seen += 1
            next_cursor = payload.get("next_cursor")
            if not next_cursor:
                return
            cursor = str(next_cursor)

    def iter_event_markets(
        self,
        *,
        event_ticker: str,
        status: str | None = None,
        historical: bool = True,
        limit: int = 200,
    ) -> Iterator[dict]:
        yield from self.iter_markets(
            status=status,
            historical=historical,
            limit=limit,
            event_ticker=event_ticker,
        )

    def iter_markets(
        self,
        *,
        status: str | None = None,
        historical: bool = False,
        limit: int = 200,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict]:
        path = "/historical/markets" if historical else "/markets"
        cursor: str | None = None
        pages_seen = 0
        while True:
            if max_pages is not None and pages_seen >= max_pages:
                return
            params: dict[str, str | int] = {"limit": limit}
            if event_ticker:
                params["event_ticker"] = event_ticker
            if series_ticker:
                params["series_ticker"] = series_ticker
            if not historical and status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            response = self._get_path(path, params=params)
            payload = response.json()
            markets = payload.get("markets", [])
            if not isinstance(markets, list):
                raise KalshiClientError("Kalshi markets response missing markets list")
            yield from (market for market in markets if isinstance(market, dict))
            pages_seen += 1
            next_cursor = payload.get("cursor")
            if not next_cursor:
                return
            cursor = str(next_cursor)

    def get_historical_market(self, market_ticker: str) -> dict:
        response = self._get_path(f"/historical/markets/{market_ticker}")
        payload = response.json()
        market = payload.get("market", payload)
        if not isinstance(market, dict):
            raise KalshiClientError("Kalshi historical market response missing market object")
        return market

    def get_event(self, event_ticker: str) -> dict:
        response = self._get_path(f"/events/{event_ticker}", allowed_statuses={404})
        if response.status_code == 404:
            return {}
        payload = response.json()
        event = payload.get("event", payload)
        if not isinstance(event, dict):
            raise KalshiClientError("Kalshi event response missing event object")
        return event

    def get_market_candlesticks(
        self,
        *,
        market_ticker: str,
        series_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> dict:
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        historical_path = f"/historical/markets/{market_ticker}/candlesticks"
        response = self._get_path(historical_path, params=params, allowed_statuses={404})
        if response.status_code != 404:
            return response.json()
        if not self._fallback_to_live_candlesticks:
            response.raise_for_status()
        live_path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        return self._get_path(live_path, params=params).json()

    def _get_path(
        self,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        allowed_statuses: set[int] | None = None,
    ) -> httpx.Response:
        return self._get_url(
            f"{self._base_url}{path}",
            params=params,
            allowed_statuses=allowed_statuses,
        )

    def _get_url(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        allowed_statuses: set[int] | None = None,
    ) -> httpx.Response:
        allowed_statuses = allowed_statuses or set()
        attempts = max(1, self._max_retries + 1)
        response: httpx.Response | None = None
        for attempt in range(attempts):
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
                self._retry_sleep(2.0**attempt)
                continue
            if response.status_code != 429:
                if response.status_code not in allowed_statuses:
                    response.raise_for_status()
                return response
            if attempt == attempts - 1:
                response.raise_for_status()
            self._retry_sleep(_retry_delay_seconds(response, attempt))
        assert response is not None
        response.raise_for_status()
        return response


class HistoricalMentionCollector:
    def __init__(
        self,
        *,
        http_client: httpx.Client,
        base_url: str = DEFAULT_BASE_URL,
        status: str | None = None,
        discovery_query: str | None = EARNINGS_MENTION_PREFIX,
        max_pages: int | None = None,
        max_markets: int | None = None,
        snapshot_hours_before_close: int = 8,
        snapshot_lookback_hours: int = 24,
        max_snapshot_staleness_minutes: int | None = None,
        fetch_event_market_pages: bool = False,
    ) -> None:
        self._client = KalshiMentionClient(http_client=http_client, base_url=base_url)
        self._status = status
        self._discovery_query = discovery_query
        self._max_pages = max_pages
        self._max_markets = max_markets
        self._snapshot_hours_before_close = snapshot_hours_before_close
        self._snapshot_lookback_hours = snapshot_lookback_hours
        self._max_snapshot_staleness_seconds = (
            max_snapshot_staleness_minutes * 60
            if max_snapshot_staleness_minutes is not None
            else None
        )
        self._fetch_event_market_pages = fetch_event_market_pages

    def collect(self) -> CollectionResult:
        rows: list[HistoricalMentionMarketRow] = []
        skipped: list[SkippedMarket] = []
        seen_markets: set[str] = set()
        events_seen = 0
        markets_seen = 0

        for series_row in self._client.iter_mention_series(
            status=self._status,
            query=self._discovery_query,
            max_pages=self._max_pages,
        ):
            events_seen += 1
            for raw_market, event_context in self._iter_series_markets(series_row):
                market_ticker = market_ticker_from_payload(raw_market)
                if not market_ticker or market_ticker in seen_markets:
                    continue
                seen_markets.add(market_ticker)
                markets_seen += 1
                if self._max_markets is not None and markets_seen > self._max_markets:
                    return self._build_result(rows, skipped, events_seen, markets_seen - 1)
                if not is_earnings_mention_market(raw_market, event_context):
                    skipped.append(
                        SkippedMarket(
                            market_ticker=market_ticker,
                            event_ticker=str(raw_market.get("event_ticker") or "") or None,
                            reason="non_earnings_mention_market",
                        )
                    )
                    continue
                row = self._build_row(raw_market=raw_market, event_context=event_context)
                if isinstance(row, SkippedMarket):
                    skipped.append(row)
                else:
                    rows.append(row)

        return self._build_result(rows, skipped, events_seen, markets_seen)

    def _iter_series_markets(self, series_row: dict) -> Iterator[tuple[dict, dict[str, str]]]:
        event_ticker = str(series_row.get("event_ticker") or "").strip()
        event_title = str(
            series_row.get("event_title")
            or series_row.get("title")
            or series_row.get("series_title")
            or ""
        )
        series_ticker = str(
            series_row.get("series_ticker") or series_row.get("ticker") or ""
        ).strip()
        context = {
            "event_ticker": event_ticker,
            "event_title": event_title,
            "series_ticker": series_ticker,
        }
        raw_markets = series_row.get("markets", [])
        if isinstance(raw_markets, list) and raw_markets:
            for raw_market in raw_markets:
                if isinstance(raw_market, dict):
                    yield _merge_event_context(raw_market, context), context
            return
        if self._fetch_event_market_pages and event_ticker:
            for raw_market in self._client.iter_event_markets(
                event_ticker=event_ticker,
                status=self._status,
            ):
                yield _merge_event_context(raw_market, context), context

    def _build_row(
        self,
        *,
        raw_market: dict,
        event_context: dict[str, str],
    ) -> HistoricalMentionMarketRow | SkippedMarket:
        market_ticker = market_ticker_from_payload(raw_market)
        event_ticker = str(
            raw_market.get("event_ticker") or event_context.get("event_ticker") or ""
        ).strip()
        raw_market = self._hydrate_market_detail_if_needed(raw_market)
        if not is_mention_market(raw_market):
            return SkippedMarket(
                market_ticker=market_ticker,
                event_ticker=event_ticker or None,
                reason="not_mention_market",
            )
        result = str(
            raw_market.get("result") or raw_market.get("final_result") or ""
        ).strip().lower()
        if result not in {"yes", "no"}:
            return SkippedMarket(
                market_ticker=market_ticker,
                event_ticker=event_ticker or None,
                reason="missing_final_outcome",
            )
        close_time = parse_datetime(raw_market.get("close_time") or raw_market.get("close_ts"))
        if close_time is None:
            return SkippedMarket(
                market_ticker=market_ticker,
                event_ticker=event_ticker or None,
                reason="missing_close_time",
            )
        event_phrase = str(
            raw_market.get("event_title")
            or event_context.get("event_title")
            or raw_market.get("title")
            or ""
        ).strip()
        market_name = str(raw_market.get("title") or event_phrase or market_ticker or "").strip()
        word_said = extract_target_phrase(raw_market)
        if not word_said:
            return SkippedMarket(
                market_ticker=market_ticker,
                event_ticker=event_ticker or None,
                reason="missing_word_said",
            )
        if market_name == event_phrase:
            market_name = f"{event_phrase} - {word_said}" if event_phrase else word_said
        series_ticker = str(
            raw_market.get("series_ticker") or event_context.get("series_ticker") or ""
        ).strip()
        if not series_ticker:
            series_ticker = series_ticker_from_event(event_ticker or market_ticker or "")
        snapshot_target_time = close_time - timedelta(hours=self._snapshot_hours_before_close)
        start_ts = int(
            (snapshot_target_time - timedelta(hours=self._snapshot_lookback_hours)).timestamp()
        )
        end_ts = int(snapshot_target_time.timestamp())

        try:
            candles_payload = self._client.get_market_candlesticks(
                market_ticker=str(market_ticker),
                series_ticker=series_ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1,
            )
        except httpx.HTTPError:
            return SkippedMarket(
                market_ticker=market_ticker,
                event_ticker=event_ticker or None,
                reason="snapshot_fetch_failed",
            )
        snapshot = select_preclose_snapshot(
            candles_payload.get("candlesticks", []),
            target_ts=end_ts,
            max_staleness_seconds=self._max_snapshot_staleness_seconds,
        )
        if snapshot is None:
            return SkippedMarket(
                market_ticker=market_ticker,
                event_ticker=event_ticker or None,
                reason="no_fresh_snapshot_candle",
            )
        return HistoricalMentionMarketRow(
            market_ticker=str(market_ticker),
            event_ticker=event_ticker,
            series_ticker=series_ticker,
            market_category=classify_market_category(
                series_ticker=series_ticker,
                event_title=event_phrase,
                market_title=market_name,
            ),
            event_phrase=event_phrase,
            market_name=market_name,
            word_said=word_said,
            normalized_word_said=normalize_phrase(word_said),
            final_outcome=result,
            status=str(raw_market.get("status") or "") or None,
            close_time=close_time,
            snapshot_target_time=snapshot_target_time,
            preclose_yes_bid=snapshot.yes_bid,
            preclose_yes_ask=snapshot.yes_ask,
            preclose_yes_mid=snapshot.yes_mid,
            candle_end_ts=snapshot.candle_end_ts,
            snapshot_staleness_seconds=snapshot.staleness_seconds,
            settlement_ts=parse_datetime(raw_market.get("settlement_ts")),
            source="kalshi_search_series",
        )

    def _hydrate_market_detail_if_needed(self, raw_market: dict) -> dict:
        market_ticker = market_ticker_from_payload(raw_market)
        if not market_ticker:
            return raw_market
        if raw_market.get("close_time") or raw_market.get("close_ts"):
            return raw_market
        try:
            detail = self._client.get_historical_market(market_ticker)
        except httpx.HTTPError:
            return raw_market
        merged = dict(raw_market)
        merged.update({key: value for key, value in detail.items() if value is not None})
        return merged

    @staticmethod
    def _build_result(
        rows: list[HistoricalMentionMarketRow],
        skipped: list[SkippedMarket],
        events_seen: int,
        markets_seen: int,
    ) -> CollectionResult:
        skip_reasons = Counter(skip.reason for skip in skipped)
        stats: dict[str, int | dict[str, int]] = {
            "events_seen": events_seen,
            "markets_seen": markets_seen,
            "rows_written": len(rows),
            "skipped_count": len(skipped),
            "skip_reasons": dict(sorted(skip_reasons.items())),
        }
        return CollectionResult(rows=rows, skipped_markets=skipped, stats=stats)


def is_mention_market(payload: dict) -> bool:
    ticker = str(payload.get("ticker") or payload.get("market_ticker") or payload.get("id") or "")
    if "MENTION" in ticker.upper():
        return True
    if "SAY" in ticker.upper() and extract_target_phrase(payload):
        return True
    if extract_target_phrase(payload):
        text = " ".join(
            str(payload.get(key) or "")
            for key in ("title", "event_title", "rules_primary", "rules", "settlement_rules")
        )
        return any(pattern.search(text) for pattern in _MENTION_TEXT_PATTERNS)
    return False


def is_earnings_mention_market(payload: dict, event_context: dict[str, str] | None = None) -> bool:
    event_context = event_context or {}
    identifiers = [
        market_ticker_from_payload(payload) or "",
        str(payload.get("event_ticker") or event_context.get("event_ticker") or ""),
        str(payload.get("series_ticker") or event_context.get("series_ticker") or ""),
    ]
    return any(identifier.upper().startswith(EARNINGS_MENTION_PREFIX) for identifier in identifiers)


def extract_target_phrase(payload: dict) -> str:
    custom_strike = payload.get("custom_strike")
    if isinstance(custom_strike, dict):
        for preferred_key in ("Word", "word", "Phrase", "phrase", "Term", "term"):
            phrase = _clean_phrase(custom_strike.get(preferred_key))
            if phrase:
                return phrase
        for value in custom_strike.values():
            phrase = _clean_phrase(value)
            if phrase:
                return phrase
    for key in ("yes_sub_title", "yes_subtitle", "subtitle", "strike"):
        phrase = _clean_phrase(payload.get(key))
        if phrase:
            return phrase
    rules_text = str(
        payload.get("rules_primary")
        or payload.get("rules")
        or payload.get("settlement_rules")
        or ""
    )
    rules_match = re.search(
        r"\bIf\s+(.+?)\s+is\s+said\s+by\b",
        rules_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if rules_match:
        phrase = _clean_phrase(rules_match.group(1))
        if phrase:
            return phrase
    market_ticker = market_ticker_from_payload(payload) or ""
    suffix = market_ticker.rsplit("-", 1)[-1] if "-" in market_ticker else ""
    return _clean_phrase(suffix)


def classify_market_category(
    *,
    series_ticker: str,
    event_title: str,
    market_title: str,
) -> MarketCategory:
    ticker = series_ticker.upper()
    text = f"{series_ticker} {event_title} {market_title}".lower()
    if ticker.startswith("KXEARNINGSMENTION") or "earnings call" in text or "earnings" in text:
        return "earnings"
    if any(term in text for term in _SPORTS_TERMS):
        return "sports"
    if any(term in text for term in _POLITICS_TERMS):
        return "politics"
    return "other"


def select_preclose_snapshot(
    candlesticks: list[dict],
    *,
    target_ts: int,
    max_staleness_seconds: int | None = None,
) -> PrecloseSnapshot | None:
    eligible = [
        candle
        for candle in candlesticks
        if _has_close(candle, "yes_bid")
        and _has_close(candle, "yes_ask")
        and int(candle.get("end_period_ts", target_ts + 1)) <= target_ts
    ]
    if not eligible:
        return None
    candle = max(eligible, key=lambda row: int(row["end_period_ts"]))
    staleness_seconds = target_ts - int(candle["end_period_ts"])
    if max_staleness_seconds is not None and staleness_seconds > max_staleness_seconds:
        return None
    yes_bid = _normalize_price(_candle_close(candle, "yes_bid"))
    yes_ask = _normalize_price(_candle_close(candle, "yes_ask"))
    yes_mid = ((yes_bid + yes_ask) / Decimal("2")).quantize(Decimal("0.01"))
    return PrecloseSnapshot(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_mid=yes_mid,
        candle_end_ts=int(candle["end_period_ts"]),
        staleness_seconds=staleness_seconds,
    )


def parse_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def market_ticker_from_payload(payload: dict) -> str | None:
    value = payload.get("ticker") or payload.get("market_ticker") or payload.get("id")
    if value is None:
        return None
    ticker = str(value).strip()
    return ticker or None


def series_ticker_from_event(value: str) -> str:
    return value.split("-", 1)[0].strip()


def normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _merge_event_context(raw_market: dict, context: dict[str, str]) -> dict:
    normalized = dict(raw_market)
    if not normalized.get("event_ticker") and context.get("event_ticker"):
        normalized["event_ticker"] = context["event_ticker"]
    if not normalized.get("event_title") and context.get("event_title"):
        normalized["event_title"] = context["event_title"]
    if not normalized.get("series_ticker") and context.get("series_ticker"):
        normalized["series_ticker"] = context["series_ticker"]
    return normalized


def _clean_phrase(value: object) -> str:
    if value is None:
        return ""
    phrase = re.sub(r"\s+", " ", str(value)).strip().strip("\"'")
    return phrase


def _has_close(candle: dict, key: str) -> bool:
    value = candle.get(key)
    return isinstance(value, dict) and (
        value.get("close_dollars") is not None or value.get("close") is not None
    )


def _candle_close(candle: dict, key: str) -> object:
    value = candle[key]
    return value.get("close_dollars", value.get("close"))


def _normalize_price(value: object) -> Decimal:
    decimal_value = Decimal(str(value))
    if decimal_value > 1:
        decimal_value = decimal_value / Decimal("100")
    return decimal_value.quantize(Decimal("0.01"))


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(float(retry_after), 8.0)
        except ValueError:
            pass
    return min(2.0**attempt, 8.0)
