from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict

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
        return self._request("GET", "/portfolio/positions", params={"limit": 200})

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
