from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
PAPER_BANKROLL = 100.0


class AccountSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    source: str
    portfolio_value: float | None = None
    free_cash: float | None = None
    position_exposure: float | None = None
    bankroll: float = PAPER_BANKROLL
    error: str | None = None


class OpenPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_ticker: str
    side: str
    contracts: float
    average_price: float | None = None
    market_value: float | None = None
    exposure: float | None = None
    realized_pnl: float | None = None
    fees_paid: float | None = None


class OpenPositionsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    source: str
    open_position_count: int = 0
    total_contracts: float = 0
    average_price: float | None = None
    total_market_value: float | None = None
    total_exposure: float | None = None
    realized_pnl: float | None = None
    fees_paid: float | None = None
    positions: list[OpenPosition] = Field(default_factory=list)
    error: str | None = None


class KalshiAccountClient:
    def __init__(
        self,
        *,
        http_client: httpx.Client,
        api_key_id: str,
        private_key_path: Path,
        base_url: str = DEFAULT_KALSHI_BASE_URL,
    ) -> None:
        if not api_key_id:
            raise ValueError("KALSHI_API_KEY_ID is required")
        if not private_key_path.exists():
            raise ValueError(f"KALSHI_PRIVATE_KEY_PATH does not exist: {private_key_path}")
        self._http = http_client
        self._api_key_id = api_key_id
        self._base_url = base_url.rstrip("/")
        self._private_key = _load_private_key(private_key_path)

    @classmethod
    def from_env(cls, *, http_client: httpx.Client) -> Self | None:
        api_key_id = os.environ.get("KALSHI_API_KEY_ID") or os.environ.get("KALSHI_API_KEY")
        private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not api_key_id or not private_key_path:
            return None
        return cls(
            http_client=http_client,
            api_key_id=api_key_id,
            private_key_path=Path(private_key_path),
            base_url=os.environ.get("KALSHI_BASE_URL", DEFAULT_KALSHI_BASE_URL),
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
            page_positions = payload.get("market_positions", [])
            if isinstance(page_positions, list):
                positions.extend(page_positions)
            next_cursor = payload.get("cursor")
            cursor = str(next_cursor) if next_cursor else None
            if not cursor:
                break
        return {"market_positions": positions}

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        parsed_path = urlparse(url).path
        response = self._http.request(
            method,
            url,
            params=params,
            headers=self._headers(method, parsed_path),
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

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


def build_account_summary(
    *,
    balance_payload: dict[str, Any] | None,
    positions_payload: dict[str, Any] | None,
    error: str | None = None,
) -> AccountSummary:
    if not balance_payload:
        return AccountSummary(
            available=False,
            source="paper",
            bankroll=PAPER_BANKROLL,
            error=error,
        )

    balance = balance_payload.get("balance", balance_payload)
    balance_obj = balance if isinstance(balance, dict) else {"balance": balance}
    free_cash = _money_from_fields(
        balance_obj,
        ("available_balance_dollars", "dollars"),
        ("available_balance", "cents"),
        ("balance_dollars", "dollars"),
        ("balance", "cents"),
    )
    portfolio_value = _money_from_fields(
        balance_obj,
        ("portfolio_value_dollars", "dollars"),
        ("portfolio_value", "cents"),
        ("balance_dollars", "dollars"),
        ("balance", "cents"),
    )
    position_exposure = _position_exposure(positions_payload)
    return AccountSummary(
        available=True,
        source="kalshi",
        portfolio_value=portfolio_value,
        free_cash=free_cash,
        position_exposure=position_exposure,
        bankroll=free_cash if free_cash is not None else PAPER_BANKROLL,
        error=error,
    )


def build_open_positions_summary(
    positions_payload: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> OpenPositionsSummary:
    if not positions_payload:
        return OpenPositionsSummary(available=False, source="paper", error=error)
    positions = positions_payload.get("market_positions", [])
    if not isinstance(positions, list):
        return OpenPositionsSummary(available=True, source="kalshi", error=error)

    rows: list[OpenPosition] = []
    for position in positions:
        if not isinstance(position, dict):
            continue
        row = _open_position(position)
        if row is not None and row.contracts > 0:
            rows.append(row)

    total_contracts = round(sum(row.contracts for row in rows), 4)
    total_market_value = _sum_optional(row.market_value for row in rows)
    total_exposure = _sum_optional(row.exposure for row in rows)
    realized_pnl = _sum_optional(row.realized_pnl for row in rows)
    fees_paid = _sum_optional(row.fees_paid for row in rows)
    average_price = None
    known_price_contracts = sum(row.contracts for row in rows if row.average_price is not None)
    if known_price_contracts:
        weighted = sum((row.average_price or 0.0) * row.contracts for row in rows)
        average_price = round(weighted / known_price_contracts, 4)

    return OpenPositionsSummary(
        available=True,
        source="kalshi",
        open_position_count=len(rows),
        total_contracts=total_contracts,
        average_price=average_price,
        total_market_value=total_market_value,
        total_exposure=total_exposure,
        realized_pnl=realized_pnl,
        fees_paid=fees_paid,
        positions=rows,
        error=error,
    )


def _position_exposure(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    positions = payload.get("market_positions", [])
    if not isinstance(positions, list):
        return None
    total = 0.0
    seen = False
    for position in positions:
        if not isinstance(position, dict):
            continue
        value = _money_from_fields(
            position,
            ("market_exposure_dollars", "dollars"),
            ("market_exposure", "cents"),
        )
        if value is not None:
            total += abs(value)
            seen = True
    return round(total, 2) if seen else None


def _open_position(payload: dict[str, Any]) -> OpenPosition | None:
    market_ticker = str(
        payload.get("ticker") or payload.get("market_ticker") or payload.get("market_id") or ""
    ).strip()
    position = _float_from_fields(payload, "position_fp", "position", "contracts", "quantity")
    if position is None:
        return None
    if not market_ticker or position == 0:
        return None
    side = "YES" if position > 0 else "NO"
    contracts = abs(position)
    average_price = _money_from_fields(
        payload,
        ("average_price_dollars", "dollars"),
        ("avg_price_dollars", "dollars"),
        ("average_price", "cents"),
        ("avg_price", "cents"),
    )
    if average_price is None:
        total_traded = _money_from_fields(
            payload,
            ("total_traded_dollars", "dollars"),
            ("total_traded", "cents"),
        )
        if total_traded is not None and contracts:
            average_price = round(total_traded / contracts, 4)
    return OpenPosition(
        market_ticker=market_ticker,
        side=side,
        contracts=contracts,
        average_price=average_price,
        market_value=_money_from_fields(
            payload,
            ("market_value_dollars", "dollars"),
            ("market_value", "cents"),
        ),
        exposure=_money_from_fields(
            payload,
            ("market_exposure_dollars", "dollars"),
            ("market_exposure", "cents"),
        ),
        realized_pnl=_money_from_fields(
            payload,
            ("realized_pnl_dollars", "dollars"),
            ("realized_pnl", "cents"),
        ),
        fees_paid=_money_from_fields(
            payload,
            ("fees_paid_dollars", "dollars"),
            ("fees_paid", "cents"),
        ),
    )


def _sum_optional(values: object) -> float | None:
    total = 0.0
    seen = False
    for value in values:
        if value is None:
            continue
        total += float(value)
        seen = True
    return round(total, 2) if seen else None


def _float_from_fields(payload: dict[str, Any], *fields: str) -> float | None:
    for field in fields:
        if field not in payload or payload[field] is None:
            continue
        try:
            return float(payload[field])
        except (TypeError, ValueError):
            continue
    return None


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _money_from_fields(
    payload: dict[str, Any],
    *fields: tuple[str, str],
) -> float | None:
    for field, unit in fields:
        if field not in payload or payload[field] is None:
            continue
        raw_value = payload[field]
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if unit == "cents":
            value /= 100
        return round(value, 2)
    return None


def _load_private_key(private_key_path: Path) -> Any:
    from cryptography.hazmat.primitives import serialization

    return serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)


def _sign_message(private_key: Any, message: bytes) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")
