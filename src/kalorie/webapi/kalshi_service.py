from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from decimal import Decimal

import httpx


@dataclass(frozen=True)
class WebMentionMarket:
    market_ticker: str
    event_ticker: str
    title: str
    target_phrase: str
    company_symbol: str
    yes_bid: Decimal
    yes_ask: Decimal
    volume: int


class KalshiWebService:
    def __init__(
        self,
        *,
        http_client: httpx.Client,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        market_cache_ttl_seconds: float = 86400.0,
    ) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._search_series_url = f"{self._base_url.split('/trade-api/', 1)[0]}/v1/search/series"
        self._market_cache_ttl_seconds = max(0.0, market_cache_ttl_seconds)
        self._market_cache_lock = threading.Lock()
        self._cached_open_markets: list[WebMentionMarket] = []
        self._cached_open_markets_fetched_at = 0.0

    def list_open_mention_markets(self) -> list[WebMentionMarket]:
        cached = self._get_cached_open_markets()
        if cached is not None:
            return cached

        markets = self._list_open_mention_markets_via_search_series()
        if markets:
            self._set_cached_open_markets(markets)
            return markets

        markets = self._list_open_mention_markets_via_global_scan()
        if markets:
            self._set_cached_open_markets(markets)
            return markets

        markets = self._list_open_mention_markets_via_series_scan()
        if markets:
            self._set_cached_open_markets(markets)
        return markets

    def list_event_mention_markets(self, event_ticker: str) -> list[WebMentionMarket]:
        normalized_event_ticker = event_ticker.strip()
        if not normalized_event_ticker:
            return []

        cached = self._get_cached_open_markets()
        if cached is not None:
            cached_rows = [
                market for market in cached if market.event_ticker == normalized_event_ticker
            ]
            if cached_rows:
                return sorted(cached_rows, key=lambda market: (-market.volume, market.market_ticker))

        response = self._http.get(
            f"{self._base_url}/markets",
            params={
                "status": "open",
                "limit": 200,
                "event_ticker": normalized_event_ticker,
            },
        )
        if response.status_code == 429:
            return []
        response.raise_for_status()
        payload = response.json()
        raw_markets = payload.get("markets", [])
        parsed_rows = self._parse_market_rows(raw_markets)
        return sorted(parsed_rows, key=lambda market: (-market.volume, market.market_ticker))

    def _list_open_mention_markets_via_global_scan(self) -> list[WebMentionMarket]:
        markets: list[WebMentionMarket] = []
        cursor: str | None = None
        while True:
            params: dict[str, str | int] = {
                "status": "open",
                "limit": 200,
                "search": "KXEARNINGSMENTION",
            }
            if cursor:
                params["cursor"] = cursor
            response = self._http.get(f"{self._base_url}/markets", params=params)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    # Return best-effort results instead of failing the entire endpoint.
                    break
                raise
            payload = response.json()
            raw_markets = payload.get("markets", [])
            markets.extend(self._parse_market_rows(raw_markets))
            next_cursor = payload.get("cursor")
            if not next_cursor:
                break
            cursor = str(next_cursor)
        return markets

    def _list_open_mention_markets_via_search_series(self) -> list[WebMentionMarket]:
        params: dict[str, str | int] = {
            "category": "Mentions",
            "status": "open",
            "order_by": "start-time",
            "reverse": "false",
            "page_size": 100,
            "hydrate": "milestones,structured_targets",
            "with_milestones": "true",
        }
        cursor: str | None = None
        markets: list[WebMentionMarket] = []
        seen_tickers: set[str] = set()

        while True:
            request_params = dict(params)
            if cursor:
                request_params["cursor"] = cursor
            response = self._http.get(self._search_series_url, params=request_params)
            if response.status_code in {403, 404, 429}:
                return []
            response.raise_for_status()
            payload = response.json()
            current_page = payload.get("current_page", [])
            if not isinstance(current_page, list):
                return []

            for series_row in current_page:
                if not isinstance(series_row, dict):
                    continue
                event_ticker = str(series_row.get("event_ticker") or "").strip()
                event_title = str(series_row.get("event_title") or series_row.get("series_title") or "")
                raw_markets = series_row.get("markets", [])
                if not isinstance(raw_markets, list):
                    continue
                for raw_market in raw_markets:
                    if not isinstance(raw_market, dict):
                        continue
                    normalized_market = dict(raw_market)
                    normalized_market["event_ticker"] = normalized_market.get("event_ticker") or event_ticker
                    if not normalized_market.get("title"):
                        normalized_market["title"] = event_title
                    if "yes_sub_title" not in normalized_market and "yes_subtitle" in normalized_market:
                        normalized_market["yes_sub_title"] = normalized_market.get("yes_subtitle")
                    if "no_sub_title" not in normalized_market and "no_subtitle" in normalized_market:
                        normalized_market["no_sub_title"] = normalized_market.get("no_subtitle")
                    parsed = self._parse_market(normalized_market)
                    if parsed is None or parsed.market_ticker in seen_tickers:
                        continue
                    seen_tickers.add(parsed.market_ticker)
                    markets.append(parsed)

            next_cursor = payload.get("next_cursor")
            if not next_cursor:
                break
            cursor = str(next_cursor)

        return markets

    def _list_open_mention_markets_via_series_scan(self) -> list[WebMentionMarket]:
        response = self._http.get(f"{self._base_url}/series")
        if response.status_code == 429:
            return []
        response.raise_for_status()
        payload = response.json()
        raw_series = payload.get("series", [])
        series_tickers = [
            str(series.get("ticker"))
            for series in raw_series
            if isinstance(series, dict)
            and str(series.get("ticker") or "").startswith("KXEARNINGSMENTION")
        ]
        markets: list[WebMentionMarket] = []
        seen_tickers: set[str] = set()
        for series_ticker in series_tickers:
            series_response = self._http.get(
                f"{self._base_url}/markets",
                params={
                    "status": "open",
                    "limit": 200,
                    "series_ticker": series_ticker,
                },
            )
            if series_response.status_code == 429:
                # Continue with best-effort results when Kalshi throttles series scans.
                time.sleep(0.15)
                continue
            series_response.raise_for_status()
            series_payload = series_response.json()
            raw_markets = series_payload.get("markets", [])
            for parsed in self._parse_market_rows(raw_markets):
                if parsed.market_ticker in seen_tickers:
                    continue
                seen_tickers.add(parsed.market_ticker)
                markets.append(parsed)
            # Pace requests to avoid triggering Kalshi 429 limits.
            time.sleep(0.15)
        return markets

    def _parse_market(self, payload: dict) -> WebMentionMarket | None:
        market_ticker = str(payload.get("ticker") or payload.get("id") or "").strip()
        event_ticker = str(payload.get("event_ticker") or "").strip()
        if not market_ticker.startswith("KXEARNINGSMENTION"):
            return None
        if not event_ticker:
            return None
        yes_bid = _read_price(payload, "yes_bid")
        yes_ask = _read_price(payload, "yes_ask")
        if yes_ask is None:
            no_bid = _read_price(payload, "no_bid")
            if no_bid is not None:
                yes_ask = (Decimal("1") - no_bid).quantize(Decimal("0.01"))
        if yes_bid is None or yes_ask is None:
            return None
        if yes_ask < yes_bid:
            yes_ask = yes_bid
        return WebMentionMarket(
            market_ticker=market_ticker,
            event_ticker=event_ticker,
            title=str(payload.get("title") or ""),
            target_phrase=_extract_target_phrase(payload=payload, market_ticker=market_ticker),
            company_symbol=_infer_company_symbol(
                event_ticker=event_ticker,
                market_ticker=market_ticker,
            ),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            volume=int(payload.get("volume") or 0),
        )

    def _parse_market_rows(self, raw_markets: object) -> list[WebMentionMarket]:
        rows: list[WebMentionMarket] = []
        if not isinstance(raw_markets, list):
            return rows
        for raw_market in raw_markets:
            parsed = self._parse_market(raw_market)
            if parsed is not None:
                rows.append(parsed)
        return rows

    def _get_cached_open_markets(self) -> list[WebMentionMarket] | None:
        if self._market_cache_ttl_seconds <= 0:
            return None
        now = time.monotonic()
        with self._market_cache_lock:
            if not self._cached_open_markets:
                return None
            age_seconds = now - self._cached_open_markets_fetched_at
            if age_seconds > self._market_cache_ttl_seconds:
                return None
            return list(self._cached_open_markets)

    def _set_cached_open_markets(self, markets: list[WebMentionMarket]) -> None:
        if self._market_cache_ttl_seconds <= 0 or not markets:
            return
        with self._market_cache_lock:
            self._cached_open_markets = list(markets)
            self._cached_open_markets_fetched_at = time.monotonic()


def _read_price(payload: dict, key: str) -> Decimal | None:
    dollars_key = f"{key}_dollars"
    raw_value = payload.get(dollars_key, payload.get(key))
    if raw_value is None:
        return None
    decimal_value = Decimal(str(raw_value))
    if decimal_value > 1:
        decimal_value = decimal_value / Decimal("100")
    return decimal_value.quantize(Decimal("0.01"))


def _infer_company_symbol(*, event_ticker: str, market_ticker: str) -> str:
    for value in (event_ticker, market_ticker):
        match = re.search(r"KXEARNINGSMENTION([A-Z]+)-", value.upper())
        if match:
            return match.group(1)
    return "UNKNOWN"


def _extract_target_phrase(*, payload: dict, market_ticker: str) -> str:
    custom_strike = payload.get("custom_strike")
    if isinstance(custom_strike, dict):
        for value in custom_strike.values():
            phrase = _normalize_phrase_text(value)
            if phrase:
                return phrase

    for key in ("yes_sub_title", "no_sub_title", "subtitle"):
        phrase = _normalize_phrase_text(payload.get(key))
        if phrase:
            return phrase

    rules_primary = str(payload.get("rules_primary") or "")
    rules_match = re.search(r"If (.+?) is said by", rules_primary, flags=re.IGNORECASE)
    if rules_match:
        phrase = _normalize_phrase_text(rules_match.group(1))
        if phrase:
            return phrase

    ticker_suffix = market_ticker.rsplit("-", 1)[-1] if "-" in market_ticker else market_ticker
    phrase = _normalize_phrase_text(ticker_suffix)
    return phrase or "UNKNOWN"


def _normalize_phrase_text(value: object) -> str:
    if value is None:
        return ""
    phrase = str(value).strip()
    if not phrase:
        return ""
    return phrase.strip("\"'")

