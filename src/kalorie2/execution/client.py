from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict

from kalorie2.kalshi_account import (
    DEFAULT_KALSHI_BASE_URL,
    _load_private_key,
    _sign_message,
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class MarketQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_ticker: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2.0


class OrderbookDepth(BaseModel):
    """Resting-bid ladders for a market, in integer cents.

    Kalshi returns resting bids per side. A YES *buy* crosses resting NO bids
    (ask price = 100 - no_bid_price), and a NO buy crosses resting YES bids.
    """

    model_config = ConfigDict(extra="forbid")

    market_ticker: str
    yes_bids: list[tuple[int, int]] = []
    no_bids: list[tuple[int, int]] = []

    def ask_levels(self, side: str) -> list[tuple[float, int]]:
        source = self.no_bids if side.upper() == "YES" else self.yes_bids
        levels = [
            (round((100 - price) / 100.0, 4), quantity)
            for price, quantity in source
            if quantity > 0 and 0 < price < 100
        ]
        levels.sort(key=lambda level: level[0])
        return levels

    def best_ask(self, side: str) -> float | None:
        levels = self.ask_levels(side)
        return levels[0][0] if levels else None

    def total_ask_depth(self, side: str, *, max_price: float) -> int:
        return sum(qty for price, qty in self.ask_levels(side) if price <= max_price + 1e-9)


class KalshiExecutionClient:
    """Authenticated Kalshi trading client.

    Reads (quotes, balance, positions, resting orders) retry transient failures.
    Order submissions never retry: a failed POST is reconciled, not blindly
    resent, so we cannot accidentally double-fill.
    """

    def __init__(
        self,
        *,
        http_client: httpx.Client,
        api_key_id: str,
        private_key: Any,
        base_url: str = DEFAULT_KALSHI_BASE_URL,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key_id:
            raise ValueError("KALSHI_API_KEY_ID is required")
        self._http = http_client
        self._api_key_id = api_key_id
        self._private_key = private_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max(0, max_retries)
        self._sleep = sleep

    @classmethod
    def from_env(cls, *, http_client: httpx.Client) -> Self | None:
        api_key_id = os.environ.get("KALSHI_API_KEY_ID") or os.environ.get("KALSHI_API_KEY")
        private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not api_key_id or not private_key_path:
            return None
        path = Path(private_key_path)
        if not path.exists():
            raise ValueError(f"KALSHI_PRIVATE_KEY_PATH does not exist: {private_key_path}")
        return cls(
            http_client=http_client,
            api_key_id=api_key_id,
            private_key=_load_private_key(path),
            base_url=os.environ.get("KALSHI_BASE_URL", DEFAULT_KALSHI_BASE_URL),
        )

    def get_market_quote(self, ticker: str) -> MarketQuote:
        payload = self._request("GET", f"/markets/{ticker}")
        market = payload.get("market", payload)
        return MarketQuote(
            market_ticker=str(market.get("ticker") or ticker),
            yes_bid=_cents_to_probability(market.get("yes_bid")),
            yes_ask=_cents_to_probability(market.get("yes_ask")),
            no_bid=_cents_to_probability(market.get("no_bid")),
            no_ask=_cents_to_probability(market.get("no_ask")),
        )

    def get_orderbook(self, ticker: str, *, depth: int | None = None) -> OrderbookDepth:
        params: dict[str, str | int] | None = {"depth": depth} if depth else None
        payload = self._request("GET", f"/markets/{ticker}/orderbook", params=params)
        book = payload.get("orderbook", payload) or {}
        return OrderbookDepth(
            market_ticker=ticker,
            yes_bids=_parse_levels(book.get("yes")),
            no_bids=_parse_levels(book.get("no")),
        )

    def get_balance(self) -> dict[str, Any]:
        return self._request("GET", "/portfolio/balance")

    def list_positions(self) -> dict[str, Any]:
        positions: list[Any] = []
        cursor: str | None = None
        while True:
            params: dict[str, str | int] = {"limit": 200, "count_filter": "position"}
            if cursor:
                params["cursor"] = cursor
            payload = self._request("GET", "/portfolio/positions", params=params)
            page = payload.get("market_positions", [])
            if isinstance(page, list):
                positions.extend(page)
            next_cursor = payload.get("cursor")
            cursor = str(next_cursor) if next_cursor else None
            if not cursor:
                break
        return {"market_positions": positions}

    def list_resting_orders(self, *, ticker: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, str | int] = {"status": "resting", "limit": 200}
        if ticker:
            params["ticker"] = ticker
        payload = self._request("GET", "/portfolio/orders", params=params)
        orders = payload.get("orders", [])
        return list(orders) if isinstance(orders, list) else []

    def submit_limit_order(
        self,
        *,
        ticker: str,
        action: str,
        side: str,
        limit_price_cents: int,
        count: int,
        client_order_id: str,
    ) -> str:
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action.lower(),
            "type": "limit",
            "side": side.lower(),
            "count": int(count),
            "client_order_id": client_order_id,
        }
        if side.lower() == "yes":
            body["yes_price"] = int(limit_price_cents)
        else:
            body["no_price"] = int(limit_price_cents)
        payload = self._request("POST", "/portfolio/orders", json_body=body, retry=False)
        return str(payload["order"]["order_id"])

    def cancel_order(self, order_id: str) -> bool:
        payload = self._request("DELETE", f"/portfolio/orders/{order_id}")
        return int(payload.get("reduced_by", 0)) > 0

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        json_body: dict[str, Any] | None = None,
        retry: bool = True,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        parsed_path = urlparse(url).path
        max_attempts = self._max_retries if retry else 0

        for attempt in range(max_attempts + 1):
            response = self._http.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=self._headers(method, parsed_path),
            )
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts:
                self._sleep(0.5 * (2**attempt) + random.uniform(0, 0.25))
                continue
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        raise RuntimeError("unreachable retry loop exit")

    def _headers(self, method: str, path: str) -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path}".encode()
        signature = _sign_message(self._private_key, message)
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }


def _cents_to_probability(value: Any) -> float:
    if value is None:
        return 0.0
    return round(float(value) / 100.0, 4)


def _parse_levels(raw: Any) -> list[tuple[int, int]]:
    if not isinstance(raw, list):
        return []
    levels: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, list | tuple) or len(entry) < 2:
            continue
        try:
            price = int(entry[0])
            quantity = int(entry[1])
        except (TypeError, ValueError):
            continue
        if quantity > 0:
            levels.append((price, quantity))
    return levels
