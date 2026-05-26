import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import httpx

from kalorie.domain.models import MarketSnapshot, MentionMarketContract
from kalorie.market.markets import MentionMarketParseError, parse_mention_market_target


class KalshiError(RuntimeError):
    pass


class KalshiParseError(KalshiError):
    pass


class KalshiSigner(Protocol):
    def sign_headers(self, method: str, path: str) -> dict[str, str]:
        """Return authenticated request headers without exposing key material."""


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(float(retry_after), 8.0)
        except ValueError:
            pass
    return min(2.0**attempt, 8.0)


class KalshiPublicClient:
    def __init__(
        self,
        http_client: httpx.Client,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        retry_sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
    ) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._retry_sleep = retry_sleep
        self._max_retries = max_retries

    def get_market(self, market_id: str) -> MarketSnapshot:
        path = f"/markets/{market_id}"
        response = self._get_response(path)
        return parse_market_snapshot(response.json(), market_id)

    def get_event_mention_markets(self, event_ticker: str) -> list[MentionMarketContract]:
        path = f"/events/{event_ticker}/markets"
        response = self._get_response(path, allowed_statuses={404})
        if response.status_code == 404:
            payload = self._get_markets_by_event_ticker(event_ticker)
            return parse_mention_market_contracts(payload, event_ticker=event_ticker)
        return parse_mention_market_contracts(response.json(), event_ticker=event_ticker)

    def get_historical_markets(
        self,
        *,
        status: str = "closed",
        search: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        params: dict[str, str | int] = {"status": status, "limit": limit}
        if search:
            params["search"] = search
        if cursor:
            params["cursor"] = cursor
        response = self._get_response("/markets", params=params)
        return response.json()

    def get_market_candlesticks(
        self,
        *,
        series_ticker: str,
        market_id: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> dict:
        path = f"/series/{series_ticker}/markets/{market_id}/candlesticks"
        response = self._get_response(
            path,
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        return response.json()

    def _get_response(
        self,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        allowed_statuses: set[int] | None = None,
    ) -> httpx.Response:
        allowed_statuses = allowed_statuses or set()
        attempts = max(1, self._max_retries + 1)
        response: httpx.Response | None = None
        for attempt in range(attempts):
            response = self._http.get(f"{self._base_url}{path}", params=params)
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

    def _get_markets_by_event_ticker(self, event_ticker: str) -> dict:
        markets: list[dict] = []
        cursor: str | None = None
        while True:
            params: dict[str, str | int] = {"event_ticker": event_ticker, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            response = self._get_response("/markets", params=params)
            payload = response.json()
            page_markets = payload.get("markets", [])
            if isinstance(page_markets, list):
                markets.extend(page_markets)
            cursor = payload.get("cursor")
            if not cursor:
                break
        return {"markets": markets}


class KalshiEarningsMarketsClient(KalshiPublicClient):
    _EARNINGS_MARKER = "KXEARNINGSMENTION"

    def list_company_mention_markets(
        self,
        company_symbol: str,
        *,
        status: str = "open",
        limit: int = 400,
        max_pages: int = 10,
        event_ticker: str | None = None,
    ) -> list[MentionMarketContract]:
        symbol = company_symbol.upper()
        if event_ticker:
            contracts = self.get_event_mention_markets(event_ticker)
            return [
                contract
                for contract in contracts
                if _contract_mentions_symbol(contract, symbol)
            ]
        payload = self._search_earnings_markets(
            company_symbol=symbol,
            status=status,
            limit=limit,
            max_pages=max_pages,
        )
        contracts = parse_mention_market_contracts(
            payload,
            event_ticker=f"{self._EARNINGS_MARKER}{symbol}",
        )
        return [contract for contract in contracts if _contract_mentions_symbol(contract, symbol)]

    def list_company_event_tickers(
        self,
        company_symbol: str,
        *,
        status: str = "open",
        limit: int = 400,
        max_pages: int = 10,
    ) -> list[str]:
        latest_observed_at_by_event: dict[str, datetime] = {}
        for contract in self.list_company_mention_markets(
            company_symbol=company_symbol,
            status=status,
            limit=limit,
            max_pages=max_pages,
        ):
            observed_at = latest_observed_at_by_event.get(contract.event_ticker)
            if observed_at is None or contract.observed_at > observed_at:
                latest_observed_at_by_event[contract.event_ticker] = contract.observed_at
        return [
            event_ticker
            for event_ticker, _ in sorted(
                latest_observed_at_by_event.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    def get_latest_company_event_ticker(
        self,
        company_symbol: str,
        *,
        status: str = "open",
        limit: int = 400,
        max_pages: int = 10,
    ) -> str | None:
        tickers = self.list_company_event_tickers(
            company_symbol=company_symbol,
            status=status,
            limit=limit,
            max_pages=max_pages,
        )
        return tickers[0] if tickers else None

    def _search_earnings_markets(
        self,
        *,
        company_symbol: str,
        status: str,
        limit: int,
        max_pages: int,
    ) -> dict:
        if limit < 1:
            return {"markets": []}
        remaining = limit
        cursor: str | None = None
        markets: list[dict] = []
        marker = f"{self._EARNINGS_MARKER}{company_symbol}"
        for _ in range(max_pages):
            params: dict[str, str | int] = {"search": marker, "limit": min(remaining, 200)}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            response = self._get_response("/markets", params=params)
            payload = response.json()
            page_markets = payload.get("markets", [])
            if isinstance(page_markets, list):
                markets.extend(page_markets)
                remaining = max(0, limit - len(markets))
            cursor = payload.get("cursor")
            if remaining <= 0 or not cursor:
                break
        return {"markets": markets[:limit]}


class KalshiAuthorizedClient(KalshiPublicClient):
    def __init__(
        self,
        http_client: httpx.Client,
        key_id: str | None,
        private_key_path: Path,
        signer: KalshiSigner,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
    ) -> None:
        if not key_id:
            raise ValueError("KALSHI_API_KEY_ID is required for authorized mode")
        if not private_key_path.exists():
            raise ValueError(f"KALSHI_PRIVATE_KEY_PATH does not exist: {private_key_path}")
        super().__init__(http_client=http_client, base_url=base_url)
        self._key_id = key_id
        self._signer = signer

    def get_market(self, market_id: str) -> MarketSnapshot:
        path = f"/markets/{market_id}"
        headers = {"KALSHI-ACCESS-KEY": self._key_id}
        headers.update(self._signer.sign_headers("GET", path))
        response = self._http.get(f"{self._base_url}{path}", headers=headers)
        response.raise_for_status()
        return parse_market_snapshot(response.json(), market_id)


def parse_market_snapshot(payload: dict, requested_market_id: str) -> MarketSnapshot:
    market = payload.get("market", payload)
    market_id = str(market.get("ticker") or market.get("id") or requested_market_id)
    try:
        yes_bid = _market_price(market, "yes_bid")
        yes_ask = _market_price(market, "yes_ask")
        observed_at = _parse_datetime(market.get("observed_at"))
        title = str(market["title"])
    except (KeyError, TypeError, ValueError) as exc:
        raise KalshiParseError(f"Could not parse Kalshi market {market_id}") from exc
    return MarketSnapshot(
        venue="kalshi",
        market_id=market_id,
        title=title,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        observed_at=observed_at,
    )


def parse_mention_market_contracts(payload: dict, event_ticker: str) -> list[MentionMarketContract]:
    markets = payload.get("markets", payload if isinstance(payload, list) else [])
    contracts: list[MentionMarketContract] = []
    for market in markets:
        title = str(market.get("title", ""))
        rules_text = str(
            market.get("rules_primary")
            or market.get("rules")
            or market.get("settlement_rules")
            or ""
        )
        try:
            target_phrase = parse_mention_market_target(title=title, rules_text=rules_text)
        except MentionMarketParseError:
            continue
        market_id = str(market.get("ticker") or market.get("id"))
        contracts.append(
            MentionMarketContract(
                venue="kalshi",
                market_id=market_id,
                event_ticker=str(market.get("event_ticker") or event_ticker),
                title=title,
                rules_text=rules_text,
                target_phrase=target_phrase,
                yes_bid=_market_price(market, "yes_bid", default=0),
                yes_ask=_market_price(market, "yes_ask", default=1),
                observed_at=_parse_datetime(market.get("observed_at")),
            )
        )
    return contracts


def _market_price(
    market: dict,
    price_key: str,
    *,
    default: int | float | str | Decimal | None = None,
) -> Decimal:
    dollars_key = f"{price_key}_dollars"
    value = market.get(dollars_key)
    if value is None:
        value = market.get(price_key, default)
    if value is None:
        raise KeyError(price_key)
    return _normalize_price(value)


def _normalize_price(value: int | float | str | Decimal) -> Decimal:
    decimal_value = Decimal(str(value))
    if decimal_value > 1:
        decimal_value = decimal_value / Decimal("100")
    return decimal_value.quantize(Decimal("0.01"))


def _parse_datetime(value: str | None) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _contract_mentions_symbol(contract: MentionMarketContract, symbol: str) -> bool:
    identifier_text = f"{contract.event_ticker} {contract.market_id} {contract.title}".upper()
    escaped_symbol = re.escape(symbol.upper())
    return bool(
        re.search(rf"(?<![A-Z0-9]){escaped_symbol}(?![A-Z0-9])", identifier_text)
        or re.search(rf"MENTION{escaped_symbol}(?![A-Z0-9])", identifier_text)
        or re.search(rf"MENTION{escaped_symbol}-", identifier_text)
    )
