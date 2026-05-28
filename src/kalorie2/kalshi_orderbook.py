from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kalorie2.kalshi_account import _load_private_key, _sign_message

DEFAULT_KALSHI_WEBSOCKET_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
ORDERBOOK_CHANNEL = "orderbook_delta"


@dataclass(frozen=True)
class OrderbookQuote:
    market_ticker: str
    yes_bid: float
    yes_ask: float
    yes_mid: float


class OrderbookState:
    def __init__(self, market_ticker: str) -> None:
        self.market_ticker = market_ticker
        self._yes: dict[int, int] = {}
        self._no: dict[int, int] = {}

    def apply_message(self, message: dict[str, Any]) -> OrderbookQuote | None:
        msg = message.get("msg")
        if not isinstance(msg, dict) or msg.get("market_ticker") != self.market_ticker:
            return None
        message_type = message.get("type")
        if message_type == "orderbook_snapshot":
            self._apply_snapshot(msg)
        elif message_type == "orderbook_delta":
            self._apply_delta(msg)
        else:
            return None
        return self.quote()

    def quote(self) -> OrderbookQuote | None:
        if not self._yes or not self._no:
            return None
        yes_bid_cents = max(self._yes)
        no_bid_cents = max(self._no)
        yes_bid = _cents_to_probability(yes_bid_cents)
        yes_ask = _cents_to_probability(100 - no_bid_cents)
        if yes_bid > yes_ask:
            return None
        return OrderbookQuote(
            market_ticker=self.market_ticker,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            yes_mid=round((yes_bid + yes_ask) / 2.0, 4),
        )

    def _apply_snapshot(self, msg: dict[str, Any]) -> None:
        self._yes = _levels_from_snapshot(msg.get("yes"))
        self._no = _levels_from_snapshot(msg.get("no"))

    def _apply_delta(self, msg: dict[str, Any]) -> None:
        side = str(msg.get("side") or "").lower()
        if side not in {"yes", "no"}:
            return
        price = _int_or_none(msg.get("price"))
        delta = _int_or_none(msg.get("delta"))
        if price is None or delta is None:
            return
        levels = self._yes if side == "yes" else self._no
        next_quantity = levels.get(price, 0) + delta
        if next_quantity <= 0:
            levels.pop(price, None)
        else:
            levels[price] = next_quantity


def build_orderbook_subscription(message_id: int, market_tickers: Iterable[str]) -> dict[str, Any]:
    tickers = sorted({ticker.strip() for ticker in market_tickers if ticker.strip()})
    if not tickers:
        raise ValueError("market_tickers must include at least one ticker")
    return {
        "id": message_id,
        "cmd": "subscribe",
        "params": {
            "channels": [ORDERBOOK_CHANNEL],
            "market_tickers": tickers,
        },
    }


def build_kalshi_websocket_headers(
    *,
    api_key_id: str,
    private_key_path: Path,
    method: str = "GET",
    path: str = "/trade-api/ws/v2",
    timestamp_ms: str | None = None,
) -> dict[str, str]:
    if not api_key_id:
        raise ValueError("KALSHI_API_KEY_ID is required")
    timestamp = timestamp_ms or str(int(time.time() * 1000))
    private_key = _load_private_key(private_key_path)
    signature = _sign_message(private_key, f"{timestamp}{method.upper()}{path}".encode())
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }


class KalshiOrderbookWebSocketClient:
    def __init__(
        self,
        *,
        api_key_id: str,
        private_key_path: Path,
        market_tickers: Iterable[str],
        ws_url: str = DEFAULT_KALSHI_WEBSOCKET_URL,
        connect: Callable[..., Any] | None = None,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self._api_key_id = api_key_id
        self._private_key_path = private_key_path
        self._market_tickers = sorted(
            {ticker.strip() for ticker in market_tickers if ticker.strip()}
        )
        self._ws_url = ws_url
        self._connect = connect
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._states = {ticker: OrderbookState(ticker) for ticker in self._market_tickers}

    async def iter_quotes(self) -> AsyncIterator[OrderbookQuote]:
        if not self._market_tickers:
            return
        backoff_seconds = self._initial_backoff_seconds
        while True:
            try:
                async for quote in self._connect_once():
                    backoff_seconds = self._initial_backoff_seconds
                    yield quote
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(backoff_seconds + random.uniform(0, backoff_seconds / 4.0))
                backoff_seconds = min(backoff_seconds * 2.0, self._max_backoff_seconds)

    async def _connect_once(self) -> AsyncIterator[OrderbookQuote]:
        connect = self._connect or _default_websocket_connect
        path = urlparse(self._ws_url).path or "/trade-api/ws/v2"
        headers = build_kalshi_websocket_headers(
            api_key_id=self._api_key_id,
            private_key_path=self._private_key_path,
            path=path,
        )
        async with connect(
            self._ws_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        ) as websocket:
            await websocket.send(json.dumps(build_orderbook_subscription(1, self._market_tickers)))
            async for raw_message in websocket:
                payload = json.loads(raw_message)
                msg = payload.get("msg")
                if not isinstance(msg, dict):
                    continue
                ticker = msg.get("market_ticker")
                state = self._states.get(str(ticker))
                if state is None:
                    continue
                quote = state.apply_message(payload)
                if quote is not None:
                    yield quote


def _default_websocket_connect(*args: Any, **kwargs: Any) -> Any:
    import websockets

    return websockets.connect(*args, **kwargs)


def _levels_from_snapshot(raw_levels: Any) -> dict[int, int]:
    levels: dict[int, int] = {}
    if not isinstance(raw_levels, list):
        return levels
    for raw_level in raw_levels:
        if not isinstance(raw_level, list | tuple) or len(raw_level) < 2:
            continue
        price = _int_or_none(raw_level[0])
        quantity = _int_or_none(raw_level[1])
        if price is not None and quantity is not None and quantity > 0:
            levels[price] = quantity
    return levels


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cents_to_probability(value: int) -> float:
    return round(value / 100.0, 4)
